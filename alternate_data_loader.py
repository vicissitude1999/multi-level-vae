import torch
import random
import numpy as np

from torchvision import datasets
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


class experiment1(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        # data is X, label is y
        outputs_to_concat = []
        for idx in range(5):
            indices1 = self.mnist.targets == 2*idx
            tmp1 = self.mnist.data[indices1]
            first_5000 = tmp1.view(tmp1.size(0), -1)[0:5000]
            first_5000 = torch.transpose(first_5000, 0, 1)

            indices2 = self.mnist.targets == (2*idx+1)
            tmp2 = self.mnist.data[indices2]
            second_5000 = tmp2.view(tmp2.size(0), -1)[0:5000]
            second_5000 = torch.transpose(second_5000, 0, 1)
        
            row = torch.cat((first_5000, second_5000), dim=1)
            outputs_to_concat.append(row)

        self.sample = torch.stack(outputs_to_concat, dim=0)


    def __len__(self):
        return 50000
    
    def __getitem__(self, idx):
        d1 = idx // 10000
        d2 = idx % 10000
        if d2 < 5000:
            label = 2*d1
        else:
            label = 2*d1 + 1

        return (self.sample[d1, :, d2].view(1, 28, 28), label)


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