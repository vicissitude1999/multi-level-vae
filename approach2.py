import os
import argparse
import numpy as np
from itertools import cycle

import torch
import random
import pickle
from torchvision import datasets
from torch.autograd import Variable
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
    # encoder.cuda()
    # decoder.cuda()

    encoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.encoder_save), map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.decoder_save), map_location=lambda storage, loc: storage))

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')
    if not os.path.exists('sqerrors'):
        os.makedirs('sqerrors')

    # create test data set and create data loader instance
    print('Creating MNIST paired test dataset...')
    paired_mnist = experiment3(50, 50, 3)
    test_data = paired_mnist.sample
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))
    
    # get the true change points and create list for predicted change points
    cps = paired_mnist.cps
    cps_hat = []
    
    # set up directories. experiment_info is the name of the experiment
    all_dirs = os.listdir('sqerrors/')
    max_dir = max([int(d[3:]) for d in all_dirs])
    new_dir_name = 'run0'+str(max_dir+1) if max_dir <= 8 else 'run'+str(max_dir+1)
    os.makedirs('sqerrors/'+new_dir_name)
    experiment_info = 'Expt3_n=500_T=50'


    # run on each test sample X_i
    for i in range(len(test_data)): # Running X_i
        print('Running X_'+str(i))
        X_i = test_data[i]
        errors = {}

        for eta in range(max(1, cps[i]-20), min(cps[i]+20, paired_mnist.T)):
            # separate into 2 groups
            g1 = X_i[:, 0:eta].transpose(0, 1)
            g2 = X_i[:, eta:paired_mnist.T].transpose(0, 1)

            total_error = 0

            for g in [g1, g2]:
                style_mu, _, class_mu, class_logvar = encoder(g)
                grouped_mu, _ = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, torch.zeros(g.size(0), 1), False
                )
                # decoder_style_input = torch.tensor(style_mu, requires_grad = True)
                # decoder_content_input = torch.tensor(grouped_mu[0], requires_grad = True)
                # above 2 lines incur warnings
                decoder_style_input = style_mu.clone().detach().requires_grad_(True)
                decoder_content_input = grouped_mu[0].clone().detach().requires_grad_(True)
                
                # optimize wrt the above variables
                decoder_style_input.cuda()
                decoder_content_input.cuda()

                content = decoder_content_input.expand(g.size(0), decoder_content_input.size(0))

                optimizer = optim.Adam(
                    [decoder_style_input, decoder_content_input]
                )

                for iterations in range(500):
                    optimizer.zero_grad()

                    reconstructed = decoder(decoder_style_input, content)
                    reconstruction_error = torch.sum((reconstructed - g).pow(2))
                    reconstruction_error.backward(retain_graph = True)

                    optimizer.step()
                
                # sq error from g1 + sq error from g2
                total_error += reconstruction_error
            
            # append the total_error of current splitting
            errors[eta] = total_error.item()
        

        # finished iterating through candidate change points
        # get the argmin t
        cp_hat = min(errors, key=errors.get)
        cps_hat.append(cp_hat)

        plt.scatter(list(errors.keys()), list(errors.values()), s=0.9)
        plt.axvline(x=cps[i])
        plt.axvline(x=cp_hat, color='r')
        plt.xlabel('etas')
        plt.ylabel('squared errors')
        plt.savefig(os.path.join('sqerrors', new_dir_name, experiment_info+'_X'+str(i)))
        plt.close()
    


    
    with open(os.path.join('sqerrors', new_dir_name, experiment_info+'.txt'), 'w') as cps_r:
        for tmp in cps_hat:
            cps_r.write('{} '.format(tmp))
        cps_r.write('\n')
        for tmp in cps:
            cps_r.write('{} '.format(tmp))


    print(cps)
    print(cps_hat)