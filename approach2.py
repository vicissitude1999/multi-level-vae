import os
import argparse
import numpy as np
from itertools import cycle

import torch
import random
import pickle
from torchvision import datasets
from torch.autograd import Variable
from alternate_data_loader import MNIST_Paired
from utils import accumulate_group_evidence, group_wise_reparameterize, reparameterize

import matplotlib.pyplot as plt
import torch.optim as optim
from utils import transform_config
from networks import Encoder, Decoder
from torch.utils.data import DataLoader
from alternate_data_loader import MNIST_Paired, experiment1, experiment3

from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--reference_data', type=str, default='fixed', help="generate output using random digits or fixed reference")
parser.add_argument('--accumulate_evidence', type=str, default=False, help="accumulate class evidence before producing swapped images")

parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the images")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")

parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")

torch.set_printoptions(precision=8)

FLAGS = parser.parse_args()

if __name__ == '__main__':
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.double()
    decoder.double()

    encoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.encoder_save), map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.decoder_save), map_location=lambda storage, loc: storage))

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')

    # load data set and create data loader instance
    print('Loading MNIST paired dataset...')
    paired_mnist = experiment3(50, 500, 3)
    cps = paired_mnist.cps
    
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    X = torch.FloatTensor(FLAGS.batch_size, 1)

    # test data
    test_data = paired_mnist.sample
    # test_data = torch.from_numpy(paired_mnist.x_test).float()
    cps_hat = []

    for i in range(len(test_data)):
        print('Running X_'+str(i))
        X_i = test_data[i]

        errors = {}

        for eta in range(max(1, cps[i]-20), min(cps[i]+20, paired_mnist.T)):
            g1 = X_i[:, 0:eta].transpose(0, 1)
            g2 = X_i[:, eta:paired_mnist.T].transpose(0, 1)

            total_error = 0

            for g in [g1, g2]:
                style_mu, _, class_mu, class_logvar = encoder(g)
                grouped_mu, _ = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, torch.zeros(g.size(0), 1), False
                )

                decoder_style_input = torch.tensor(style_mu, requires_grad = True)
                decoder_content_input = torch.tensor(grouped_mu[0], requires_grad = True)

                content = decoder_content_input.expand(g.size(0), decoder_content_input.size(0))

                optimizer = optim.Adam(
                    [decoder_style_input, decoder_content_input]
                )

                for iterations in range(500):
                    optimizer.zero_grad()

                    reconstructed = decoder(decoder_style_input, content)
                    reconstruction_error = torch.sum((reconstructed - g).pow(2))
                    # print(reconstruction_error)
                    reconstruction_error.backward(retain_graph = True)

                    optimizer.step()
                
                # print(reconstruction_error)
                total_error += reconstruction_error
            
            errors[eta] = total_error.item()
        
        cp_hat = min(errors, key=errors.get)
        cps_hat.append(cp_hat)

        plt.scatter(list(errors.keys()), list(errors.values()), s=0.9)
        plt.axvline(x=cps[i])
        plt.axvline(x=cp_hat, color='r')
        plt.xlabel('etas')
        plt.ylabel('squared errors')
        plt.savefig('sqerrors/run-1/' + 'Expt3_n=50_T=500_X' + str(i))
        plt.close()

    print(cps)
    print(cps_hat)


'''
            with open('errors.txt', 'w') as f:
                for e in errors:
                    f.write("%s\n" % e.item())
'''