import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from convert_to_bvh import save_motion_to_bvh_file


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'wmib':
            self.data_root = './dataset/wMIB'
            self.motion_dir = pjoin(self.data_root, 'data')
            # self.text_dir = ''
            self.joints_num = 31

            self.max_motion_length = 196
            self.meta_dir = ''
        
        joints_num = self.joints_num

        self.data = []
        self.lengths = []

        if self.dataset_name != 'wmib':
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))

            split_file = pjoin(self.data_root, 'train.txt')

            id_list = []
            with cs.open(split_file, 'r') as f:
                for line in f.readlines():
                    id_list.append(line.strip())

            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                    if motion.shape[0] < self.window_size:
                        continue
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.data.append(motion)
                except:
                    # Some motion may not exist in KIT dataset
                    pass

            self.mean = mean
            self.std = std
            print("Total number of motions {}".format(len(self.data)))
        else:
            self.data_dict = {}
            with np.load(pjoin(self.motion_dir, 'motion_data.npz'), allow_pickle=True) as motion_data:
                self.motion_dict = dict(motion_data.items())
            names = np.load(pjoin(self.motion_dir, 'motion_name.npz'), allow_pickle=True)
            self.new_name_list = names['new_name_list']
            sum_arr = np.zeros(189)
            sum_square = np.zeros(189)
            for _, name in tqdm(enumerate(self.new_name_list), desc='Loading motion segments',
                                total=len(self.new_name_list)):
                self.data_dict[name] = self.motion_dict[name].item()
                motion = self.data_dict[name]['motion']
                if motion.shape[0] < self.window_size:
                    continue
                self.data.append(motion)
                self.lengths.append(self.data_dict[name]['length'])

            tot_length = np.sum(self.lengths)
            print(tot_length)
            for motion in self.data:
                sum_arr = sum_arr + np.sum(motion, axis=0) / float(tot_length)
                sum_square = sum_square + np.sum(np.square(motion), axis=0) / float(tot_length)

            #f = np.load(pjoin(self.motion_dir, 'Mean.npz'), allow_pickle=True)
            #self.mean = f['mean']
            #f2 = np.load(pjoin(self.motion_dir, 'Std.npz'), allow_pickle=True)
            #self.std = f2['std']
            self.mean = sum_arr
            self.std = np.sqrt(sum_square - np.square(self.mean))

            self.std[0: 3] = self.std[0: 3].mean() / 1.
            self.std[3:] = self.std[3:].mean() / 1.

            np.savez('./dataset/wMIB/data/Mean.npz', mean=self.mean)
            np.savez('./dataset/wMIB/data/Std.npz', std=self.std)

            print(self.mean, self.std)

            # save_motion_to_bvh_file('./motion.bvh', self.data[19])

            print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self):
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]

        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # save_motion_to_bvh_file('./motion.bvh', self.data[25])

        #print(motion)

        return motion

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
