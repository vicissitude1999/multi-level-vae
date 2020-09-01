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
    def __init__(self, n, T):
        self.n = n
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        self.labels = []
        self.T = int(T)

        # data is X, label is y
        outputs_to_concat = []
        possible_cps = [T//2, T//4, T//3, T//5]
        cps_to_concat = []
        
        for i in range(n):
            candidates = random.sample(range(10), 2)
            i1 = min(candidates)
            i2 = max(candidates)

            cp = random.sample(possible_cps, 1)

            indices1 = self.mnist.targets == i1
            tmp1 = self.mnist.data[indices1]
            first_5000 = tmp1.view(tmp1.size(0), -1)[0:cp]
            first_5000 = torch.transpose(first_5000, 0, 1)

            indices2 = self.mnist.targets == i2
            tmp2 = self.mnist.data[indices2]
            second_5000 = tmp2.view(tmp2.size(0), -1)[0:T-cp]
            second_5000 = torch.transpose(second_5000, 0, 1)
        
            row = torch.cat((first_5000, second_5000), dim=1)
            outputs_to_concat.append(row)

            self.labels.append([i1, i2])

        self.sample = torch.stack(outputs_to_concat, dim=0)

    def __len__(self):
        return self.T*self.n
    
    def __getitem__(self, idx):
        d1 = idx // self.T
        d2 = idx % self.T
        if d2 < self.T//4:
            label = self.labels[d1][0]
        else:
            label = self.labels[d1][1]

        # print(self.sample[d1, :, d2])
        return (self.sample[d1, :, d2].view(1, 28, 28), label)

class DoubleUniNormal(Dataset):
    def __init__(self, dsname):
        file_name = root_dir + 'data/original/' + dsname + '.pickle'
        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)
        self.x_train, self.y_train, self.x_test, self.y_test = dataset
    
    def __len__(self):
        return self.x_train.size

    def __getitem__(self, idx):
        _, T = self.x_train.shape
        row = idx // T
        column = idx % T
        if column < self.y_train[row]:
            label = 2*row
        else:
            label = 2*row + 1
        return (self.x_train[row][column], label)