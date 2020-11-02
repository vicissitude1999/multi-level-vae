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
import alternate_data_loader
import utils

from mpl_toolkits.axes_grid1 import ImageGrid

from multiprocessing import Pool

from functools import partial
print = partial(print, flush=True)

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=False, help="run the following code on a GPU")
parser.add_argument('--reference_data', type=str, default='fixed', help="generate output using random digits or fixed reference")
parser.add_argument('--accumulate_evidence', type=str, default=False, help="accumulate class evidence before producing swapped images")

parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the images")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")

parser.add_argument('--style_dim', type=int, default=10, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=10, help="dimension of common factor latent space")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder_clever', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder_clever', help="model save for decoder")

# The 2 variables at line 114 and 115 should be moved to GPU.
# The code works but I'm not sure if it's actually running on GPU.
# I tested on the GPU of my laptop but it's not faster than CPU.
# I thought moving encoder and decoder (line 58, 59) to GPU can make things faster.
# But an error pops up: an illegal memory access was encountered in utils.py
# It's really confusing why a function in utils.py can cause this error.
torch.set_printoptions(precision=8)
FLAGS = parser.parse_args()

def extract_reconstructions(encoder_input, style_mu, class_mu, class_logvar):
    grouped_mu, _ = accumulate_group_evidence(
        class_mu.data, class_logvar.data, torch.zeros(style_mu.size(0), 1)
    )
    decoder_style_input = style_mu.clone().detach().requires_grad_(True)
    decoder_content_input = grouped_mu[0].clone().detach().requires_grad_(True)

    # optimize wrt the above variables
    # decoder_style_input.cuda()
    # decoder_content_input.cuda()

    content = decoder_content_input.expand(style_mu.size(0), decoder_content_input.size(0))

    optimizer = optim.Adam(
        [decoder_style_input, decoder_content_input]
    )

    for iterations in range(50):
        optimizer.zero_grad()

        reconstructed = decoder(decoder_style_input, content)
        reconstruction_error = torch.sum((reconstructed - encoder_input).pow(2))
        reconstruction_error.backward()

        optimizer.step()

    return reconstructed, reconstruction_error

def get_eta_error(eta, X, encoder, maxlen):
    # separate into 2 groups
    g1 = X[0:eta]
    g2 = X[eta:maxlen]

    style_mu_bef, _, class_mu_bef, class_logvar_bef = encoder(g1)
    style_mu_aft, _, class_mu_aft, class_logvar_aft = encoder(g2)

    _, bef_reconstruction_error = extract_reconstructions(g1, style_mu_bef, class_mu_bef, class_logvar_bef)
    _, aft_reconstruction_error = extract_reconstructions(g2, style_mu_aft, class_mu_aft, class_logvar_aft)

    # sq error from g1 + sq error from g2
    total_error = bef_reconstruction_error + aft_reconstruction_error

    return total_error.item()


if __name__ == '__main__':
    # make necessary directories
    cwd = os.getcwd()
    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')
    if not os.path.exists('sqerrors'):
        os.makedirs('sqerrors')

    all_dirs = os.listdir(os.path.join(cwd, 'sqerrors'))
    max_dir = 0
    if all_dirs:
        max_dir = max([int(d[3:]) for d in all_dirs])
    new_dir_name = 'run' + str(max_dir+1).zfill(2)
    directory_name = os.path.join(cwd, 'sqerrors', new_dir_name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    gpu_ids = []
    for ii in range(6):
        try:
            torch.cuda.get_device_properties(ii)
            print(str(ii), flush=True)
            if not gpu_ids:
                gpu_ids = [ii]
            else:
                gpu_ids.append(ii)
        except AssertionError:
            print('Not ' + str(ii) + "!", flush=True)

    print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
    gpu_ids = [int(x) for x in gpu_ids]
    # device management
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_dataparallel = len(gpu_ids) > 1
    print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

    # model definition
    encoder = Encoder()
    decoder = Decoder()
    # load saved parameters of encoder and decoder
    encoder.load_state_dict(torch.load(os.path.join(cwd, 'checkpoints', 'encoder_clever'),
                                       map_location=lambda storage, loc: storage))
    decoder.load_state_dict(torch.load(os.path.join(cwd, 'checkpoints', 'decoder_clever'),
                                       map_location=lambda storage, loc: storage))
    encoder=encoder.to(device=device)
    decoder=decoder.to(device=device)

    # loading dataset
    print('Loading CLEVR test data...')
    ds = alternate_data_loader.clever_change(utils.transform_config2)
    _, test_indices = utils.subset_sampler(ds, 10, 0.3, True, random_seed=42)


    # get the true change points and create list for predicted change points
    cps = ds.cps_dict
    cps_hat = []


    # run on each test sample X_i
    for i in test_indices: # Running X_i
        print('Running X_'+str(i))
        X = torch.empty(size=(10,) + ds.data_dim)
        for j in range(10):
            X[j] = ds[10*i + j][0]
        X = X.to(device)

        errors = {}
        minimum_eta = 2
        maximum_eta = 8

        # partial is awesome
        eta_error_calc = partial(get_eta_error, encoder=encoder, X=X.detach(), maxlen=10)
        for eta in range(minimum_eta, maximum_eta):
            total_error = eta_error_calc(eta)
            errors[eta] = total_error

        # finished iterating through candidate change points
        # get the argmin t
        cp_hat = min(errors, key=errors.get)
        cps_hat.append(cp_hat)

        plt.scatter(list(errors.keys()), list(errors.values()), s=0.9)
        plt.axvline(x=cps[i])
        plt.axvline(x=cp_hat, color='r')
        plt.xlabel('etas')
        plt.ylabel('squared errors')
        plt.savefig(os.path.join('sqerrors', new_dir_name, 'clevr_X'+str(i)+'.jpg'))
        plt.close()




    with open(os.path.join('sqerrors', new_dir_name, 'clevr.txt'), 'w') as cps_r:
        for tmp in cps_hat:
            cps_r.write('{} '.format(tmp))
        cps_r.write('\n')
        for tmp in cps:
            cps_r.write('{} '.format(tmp))