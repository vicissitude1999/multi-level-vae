import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import transform_config, reparameterize
from alternate_data_loader import DoubleUniNormal

class Encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=128):
        super(Encoder, self).__init__()
        self.fc_hidden1 = fc_hidden1, self.fc_hidden2 = fc_hidden2, self.CNN_embed_dim = CNN_embed_dim

        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # latent class mu and sigma
        self.fc3_style_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)
        self.fc3_style_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)
        self.fc3_content_mu = nn.Linear(self.fc_hidden2,self.CNN_embed_dim)
        self.fc3_content_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))

        style_mu, style_logvar = self.fc3_style_mu(x), self.fc3_style_logvar(x)
        content_mu, content_logvar = self.fc3_content_mu(x), self.fc3_content_logvar(x)

        return style_mu, style_logvar, content_mu, content_logvar


class Decoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=128):
        super(Decoder, self).__init__()
        self.fc_hidden1 = fc_hidden1, self.fc_hidden2 = fc_hidden2, self.CNN_embed_dim = CNN_embed_dim

        # CNN architectures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        self.fc4 = nn.Linear(2*self.CNN_embed_dim, self.fc_hidden2)
        self.bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def forward(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

        return x