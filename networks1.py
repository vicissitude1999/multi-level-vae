import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from utils import transform_config, reparameterize

class DFCVAE(nn.Module):
    def __init__(self, z_dim=128, hidden_dims = None, alpha=1.0, beta=0.5):
        super(DFCVAE, self).__init__()
        self.z_dim = z_dim
        self.alpha = alpha
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

                # Build Encoder
        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_style_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_style_var = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_content_mu = nn.Linear(hidden_dims[-1]*4, z_dim)
        self.fc_content_var = nn.Linear(hidden_dims[-1]*4, z_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(2*z_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.feature_network = models.vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        style_mu, style_logvar = self.fc_style_mu(x), self.fc_style_var(x)
        content_mu, content_logvar = self.fc_content_mu(x), self.fc_content_var(x)

        return style_mu, style_logvar, content_mu, content_logvar
    
    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)

        return x
    
    def extract_features(self, input, feature_layers = None):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features


class resnetVAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, z_dim=128):
        super(resnetVAE, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.z_dim = fc_hidden1, fc_hidden2, z_dim

        # encoding components
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.block1 = nn.Sequential(
            nn.Linear(resnet.fc.in_features, self.fc_hidden1),
            nn.BatchNorm1d(self.fc_hidden1, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        # latent class mu and sigma
        self.style_mu = nn.Linear(self.fc_hidden2, self.z_dim)
        self.style_logvar = nn.Linear(self.fc_hidden2, self.z_dim)
        self.content_mu = nn.Linear(self.fc_hidden2,self.z_dim)
        self.content_logvar = nn.Linear(self.fc_hidden2, self.z_dim)

        # decoding components
        self.block3 = nn.Sequential(
            nn.Linear(2*self.z_dim, self.fc_hidden2),
            nn.BatchNorm1d(self.fc_hidden2),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Linear(self.fc_hidden2, 64*4*4),
            nn.BatchNorm1d(64*4*4),
            nn.ReLU(inplace=True)
        )

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()   # restrict the range to (0,1) because input image are RBG colors in (0,1)
        )

        self.feature_network = models.vgg19_bn(pretrained=True)

        # Freeze the pretrained feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()
    
    def encode(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.block2(self.block1(x))

        style_mu, style_logvar = self.style_mu(x), self.style_logvar(x)
        content_mu, content_logvar = self.content_mu(x), self.content_logvar(x)

        return style_mu, style_logvar, content_mu, content_logvar
    
    def decode(self, style_z, content_z):
        x = torch.cat((style_z, content_z), dim=1)
        x = self.block3(x)
        x = self.block4(x).view(-1, 64, 4, 4)
        x = self.convTrans7(self.convTrans6(self.convTrans5(x)))
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

        return x
    
    def extract_features(self, input, feature_layers = None):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features

class z_classifier(nn.Module):
    def __init__(self, z_dim=128):
        super(z_classifier, self).__init__()
        self.z_dim = z_dim
        self.block1 = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(32, 1)
        )
        print(True)
        
    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        return x