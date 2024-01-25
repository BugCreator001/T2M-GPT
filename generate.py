clip_text = ["pick up stoop Medium speed Forward No turning"]
clip_text = ["equipt walk Fast Right No turning"]


import sys
sys.argv = ['GPT_eval_multi.py']
import options.option_transformer as option_trans
args = option_trans.get_args_parser()

args.dataname = 'wmib'
args.resume_pth = 'output_vqfinal/exp_debug/saved_net_200000.pth'
args.resume_trans = 'output_GPT_Final/exp_debug/net_last.pth'
args.down_t = 2
args.depth = 3
args.block_size = 52
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
from os.path import join as pjoin
from convert_to_bvh import save_motion_to_bvh_file_speed, save_motion_to_bvh_file
warnings.filterwarnings('ignore')

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                embed_dim=512,
                                clip_dim=args.clip_dim,
                                block_size=args.block_size,
                                num_layers=args.num_layers,
                                n_head=8,
                                drop_out_rate=args.drop_out_rate,
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=False)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['net'], strict=False)
trans_encoder.eval()
trans_encoder.cuda()

motion_dir = './dataset/wMIB/data'
f = np.load(pjoin(motion_dir, 'Mean.npz'), allow_pickle=True)
mean = f['mean']
f2 = np.load(pjoin(motion_dir, 'Std.npz'), allow_pickle=True)
std = f2['std']

mean = torch.from_numpy(mean).cuda()
std = torch.from_numpy(std).cuda()

text = clip.tokenize(clip_text, truncate=True).cuda()
print(text)

feat_clip_text = clip_model.encode_text(text).float()
print(feat_clip_text[0:1])

index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
print(index_motion)

pred_pose = net.forward_decoder(index_motion)
print(pred_pose.shape)
pred_pose = pred_pose.reshape(pred_pose.shape[0]*pred_pose.shape[1], pred_pose.shape[2])
pred_pose = pred_pose * std + mean
#from utils.motion_process import recover_from_ric
#pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
# xyz = pred_xyz.reshape(1, -1, 22, 3)

save_motion_to_bvh_file('./motion.bvh', pred_pose.detach().cpu().numpy())
