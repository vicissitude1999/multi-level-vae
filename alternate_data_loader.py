import torch
import random
import pickle
import os
import numpy as np
from PIL import Image

import utils
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.utils import save_image


root_dir = '/media/renyi/HDD/'

'''
n = 10000
ds_train = datasets.CelebA(root=root_dir, download=True, split='train', transform=utils.transform_config)
men = []
women = []
for i in range(n):
    if ds_train[i][1][20] == 1:
        men.append(i)
    else:
        women.append(i)
with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'wb') as f:
    pickle.dump(men, f)
with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'wb') as f:
    pickle.dump(women, f)
'''

'''
n = 1000
ds_train = datasets.CelebA(root=root_dir, download=True, split='test', transform=utils.transform_config)
men = []
women = []
for i in range(n):
    if ds_train[i][1][20] == 1:
        men.append(i)
    else:
        women.append(i)
with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'wb') as f:
    pickle.dump(men, f)
with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'wb') as f:
    pickle.dump(women, f)
'''


class celeba(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        if idx%self.T < self.cps[idx//self.T]:
            label = 2*(idx//self.T)
        else:
            label = 2*(idx//self.T) + 1

        return self.ds_train[self.map[idx]][0], label
    

class celeba_test(Dataset):
    def __init__(self, N, T):
        self.N = N
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='test', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(N)]
        self.cps = cps
        self.men_indices = [i for i in range(N*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(N*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.N*self.T
    
    def __getitem__(self, idx):
        if idx%self.T < self.cps[idx//self.T]:
            label = 2*(idx//self.T)
        else:
            label = 2*(idx//self.T) + 1

        return self.ds_train[self.map[idx]][0], label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X




class celeba_classification(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        if idx in self.men_indices:
            label = 1
        else:
            label = 0

        return self.ds_train[self.map[idx]][0], label


class celeba_test_classification(Dataset):
    def __init__(self, N, T):
        self.N = N
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='test', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(N)]
        self.cps = cps
        self.men_indices = [i for i in range(N*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(N*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.N*self.T
    
    def __getitem__(self, idx):
        if idx in self.men_indices:
            label = 1
        else:
            label = 0

        return self.ds_train[self.map[idx]][0], label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X




class celeba_change_person(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        self.groups_iterated = {}

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if label not in self.groups_iterated:
            idx = random.choice(range(10000))
            self.groups_iterated[label] = idx
        sample = self.ds_train[self.groups_iterated[label]][0]

        return sample, label
    

class celeba_test_change_person(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        self.groups_iterated = {}

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if label not in self.groups_iterated:
            idx = random.choice(range(10000))
            self.groups_iterated[label] = idx
        sample = self.ds_train[self.groups_iterated[label]][0]

        return sample, label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X
