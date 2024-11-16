import os
import json
import warnings
import numpy as np
import random
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from data_utils.WLT import WLT, vis_pc

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ShapeNetDataLoader(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # print("self.classes==>>",self.classes)
        self.class_name = [k for k,v in self.classes.items()]
        # print("self.class_name==>>",self.class_name)
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                label = self.classes[cat]
                label = np.array([label]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                point_set = data[:, 0:3]
                point_set = farthest_point_sample(point_set, self.npoints)
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = label
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

        self.list_of_points = np.stack(self.list_of_points)
        self.list_of_labels = np.stack(self.list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        return point_set, label[0]

    def __len__(self):
        return len(self.datapath)


class BDShapeNetDataLoader(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal/', args=None, split='train', class_choice=None, normal_channel=False):
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.root = root
        self.split = split
        self.args = args
        if split == 'train':
            self.poisoned_rate = args.poisoned_rate
        else:
            self.poisoned_rate = 1.0
        self.target_label = args.target_label
        self.seed = args.seed
        random.seed(self.seed)
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        self.class_name = [k for k,v in self.classes.items()]

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        self.save_path = os.path.join(root, 'shapenet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        if not os.path.exists(self.save_path):
            print('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = [None] * len(self.datapath)
            self.list_of_labels = [None] * len(self.datapath)

            for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                fn = self.datapath[index]
                cat = self.datapath[index][0]
                label = self.classes[cat]
                label = np.array([label]).astype(np.int32)
                data = np.loadtxt(fn[1]).astype(np.float32)
                point_set = data[:, 0:3]
                point_set = farthest_point_sample(point_set, self.npoints)
                self.list_of_points[index] = point_set
                self.list_of_labels[index] = label
            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            print('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)

        t_list_of_points, t_list_of_labels = [], []
        if split == 'test':
            for idx in range(len(self.datapath)):
                if self.list_of_labels[idx] != self.target_label:
                    t_list_of_points.append(self.list_of_points[idx])
                    t_list_of_labels.append(self.list_of_labels[idx])
            self.list_of_points, self.list_of_labels = np.array(t_list_of_points), np.array(t_list_of_labels)

        total_num = len(self.list_of_labels)
        self.poison_num = int(total_num * self.poisoned_rate)
        tmp_list = []
        for k in range(total_num):
            if self.list_of_labels[k] != self.target_label:
                tmp_list.append(k)
        random.shuffle(tmp_list)
        self.poison_set = frozenset(tmp_list[:self.poison_num])
        self.origin_adv_label = None
        if split == 'test':
            self.origin_adv_label = [ self.list_of_labels[j] for j in  self.poison_set]
        print('The size of clean data is %d' % (total_num - len(self.poison_set)))
        print('The size of poison data is %d' % (len(self.poison_set)))
        self.add_WLT_trigger = WLT(args)
        self.origin_data = []
        self.add_trigger()

    def __len__(self):
        return len(self.list_of_labels)
    
    def add_trigger(self):
        tri_list_of_points, tri_list_of_labels = [None] * len(self.list_of_labels), [None] * len(self.list_of_labels)
        for idx in range(len(self.list_of_labels)):
            point_set, lab = self.list_of_points[idx][:, 0:3], self.list_of_labels[idx]
            if idx in self.poison_set:
                self.origin_data.append(point_set)
                _, point_set = self.add_WLT_trigger(point_set)

                if self.args.alltoall:
                    lab = np.array([(lab[0] + 1) % self.num_category]).astype(np.int32)
                else:
                    lab = np.array([self.target_label]).astype(np.int32)
            tri_list_of_points[idx] = point_set
            tri_list_of_labels[idx] = lab
        self.list_of_points, self.list_of_labels = np.array(tri_list_of_points), np.array(tri_list_of_labels)

    def __getitem__(self, index):
        point_set, label = self.list_of_points[index][:, 0:3], self.list_of_labels[index]
        point_set = pc_normalize(point_set)     # shape: (1024, 3)
        return point_set, label[0]

