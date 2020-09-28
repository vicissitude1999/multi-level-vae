import torch
import random
import pickle
import os

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

