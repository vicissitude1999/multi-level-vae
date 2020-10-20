import torch
import random
import pickle
import os
import numpy as np
from PIL import Image

from torchvision import datasets
import torchvision.transforms as transforms
from utils import transform_config
from torch.utils.data import Dataset


class MNIST_Paired(Dataset):
    def __init__(self, root='mnist', download=True, train=True, transform=transform_config):
        self.mnist = datasets.MNIST(root=root, download=download, train=train, transform=transform)

        self.data_dict = {}

        for i in range(self.__len__()):
            image, label = self.mnist.__getitem__(i)

            try:
                self.data_dict[label.item()]
            except KeyError:
                self.data_dict[label.item()] = []
            self.data_dict[label.item()].append(image)

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        image, label = self.mnist.__getitem__(index)

        # return another image of the same class randomly selected from the data dictionary
        # this is done to simulate pair-wise labeling of data
        return image, random.SystemRandom().choice(self.data_dict[label.item()]), label


class DoubleUniNormal(Dataset):
    def __init__(self, dsname):
        root_dir = os.getcwd()
        file_name = os.path.join(root_dir, 'data', dsname)
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        self.x_train, self.y_train, self.x_test, self.y_test = dataset
        self.T = len(self.x_test[0])
        # self.x_test = self.x_test.astype(float)
    
    def __len__(self):
        return self.x_train.size

    def __getitem__(self, idx):
        _, T = self.x_train.shape # 1500 by 100
        row = idx // T
        column = idx % T

        if column < self.y_train[row]:
            label = 2*row
        else:
            label = 2*row + 1
        
        return (self.x_train[row, column].reshape(1), label)


class DoubleMulNormal(Dataset):
    def __init__(self, dsname):
        root_dir = os.getcwd()
        file_name = os.path.join(root_dir, 'data', dsname)
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        self.x_train, self.y_train, self.x_test, self.y_test = dataset
        self.T = self.x_train.shape[2]
    
    def __len__(self):
        n, d, T = self.x_train.shape
        return n*T

    def __getitem__(self, idx):
        n, d, T = self.x_train.shape
        row = idx // T
        column = idx % T

        if column < self.y_train[row]:
            label = 2*row
        else:
            label = 2*row + 1
        
        # print(self.x_train[row, : , column].shape)
        return (self.x_train[row, : ,column], label)


class experiment3(Dataset):
    def __init__(self, n, T, cp_way):
        self.n = n
        self.T = int(T)
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        self.labels = []
        self.cps = []

        # data is X, label is y
        outputs_to_concat = []
        possible_cps = []
        if cp_way == 1: # fixed cp value
            possible_cps = [T//2]
        elif cp_way == 2: # set pf possible cp values
            possible_cps = [T//2, T//3, T//4]
        elif cp_way == 3: # interval of cp values
            possible_cps = list(range(T//4, 3*T//4+1, 1))
        
        for i in range(n):
            i1, i2 = random.sample(range(10), 2) # sample 2 digits
            cp = random.sample(possible_cps, 1)[0] # sample change point

            data1 = self.mnist.data[self.mnist.targets == i1]
            idx1 = random.sample(range(data1.size(0)), cp)
            part1 = data1.view(data1.size(0), -1)[idx1]
            part1 = torch.transpose(part1, 0, 1) # convert it to 784 by l dimension

            data2 = self.mnist.data[self.mnist.targets == i2]
            idx2 = random.sample(range(data2.size(0)), T-cp)
            part2 = data2.view(data2.size(0), -1)[idx2]
            part2 = torch.transpose(part2, 0, 1)
        
            row = torch.cat((part1, part2), dim=1)
            outputs_to_concat.append(row)

            # i1, i2 if repetitive labels, otherwisse just 2n many
            self.labels.append([2*i, 2*i+1])
            self.cps.append(cp)

        self.sample = torch.stack(outputs_to_concat, dim=0) # get the finally formatted date

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        d1 = idx // self.T
        d2 = idx % self.T
        if d2 < self.cps[d1]:
            label = self.labels[d1][0]
        else:
            label = self.labels[d1][1]

        return (self.sample[d1, :, d2], label)

class clever_change(Dataset):
    def __init__(self, transform):
        dir1 = os.path.join(os.getcwd(), 'nsc_images/')
        all_filenames = os.listdir(dir1)
        self.transform = transform
        self.data_dim = self.transform(Image.open(dir1 + all_filenames[0])).shape

        cps_dict = {}
        for filename in all_filenames:
            _, _, i, t = filename.split('_') # i-th image at time t
            t = int(t[:-4])
            i = int(i)
            cps_dict[i] = cps_dict.get(i, 0) + 1

        self.cps_dict = cps_dict

    def __len__(self):
        return 10*len(self.cps_dict)

    def __getitem__(self, item):
        i, t = item // 10, item % 10

        if t < self.cps_dict[i]:
            label = 2*i
            file_name = 'nsc_images/CLEVR_nonsemantic_'+ str(i).zfill(6)+'_'+\
                        str(t)+'.png'
            img = Image.open(file_name)
        else:
            label = 2*i+1
            file_name = 'sc_images/CLEVR_semantic_'+ str(i).zfill(6)+'_'+\
                        str(t-self.cps_dict[i])+'.png'
            img = Image.open(file_name)

        return self.transform(img), label