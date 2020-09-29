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
from alternate_data_loader import MNIST_Paired, DoubleUniNormal

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

parser.add_argument('--style_dim', type=int, default=1, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=1, help="dimension of common factor latent space")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder_1_var_reparam', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder_1_var_reparam', help="model save for decoder")

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

    content = decoder_content_input.expand(style_mu.size(0), decoder_content_input.size(0)).double()

    optimizer = optim.Adam(
        [decoder_style_input, decoder_content_input]
    )

    for iterations in range(100):
        optimizer.zero_grad()

        reconstructed = decoder(decoder_style_input, content)
        reconstruction_error = torch.log(torch.sum((reconstructed - encoder_input).pow(2)))
        reconstruction_error.backward()

        optimizer.step()

    return reconstructed, reconstruction_error

def get_eta_error(eta, X, encoder, maxlen):
    # separate into 2 groups
    g1 = X[0: eta].view(eta, -1)
    g2 = X[eta: maxlen].view(maxlen-eta, -1)

    style_mu_bef, _, class_mu_bef, class_logvar_bef = encoder(g1)
    style_mu_aft, _, class_mu_aft, class_logvar_aft = encoder(g2)

    _, bef_reconstruction_error = extract_reconstructions(g1, style_mu_bef, class_mu_bef, class_logvar_bef)

    _, aft_reconstruction_error = extract_reconstructions(g2, style_mu_aft, class_mu_aft, class_logvar_aft)

    # sq error from g1 + sq error from g2
    total_error = bef_reconstruction_error + aft_reconstruction_error

    return total_error.item()


if __name__ == '__main__':
    """
    model definitions
    """
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

    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.double()
    decoder.double()

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')
    if not os.path.exists('sqerrors'):
        os.makedirs('sqerrors')

    cwd = os.getcwd()
    dirs = os.listdir(os.path.join(cwd, 'data'))
    print('Loading double univariate normal time series test data...')

    for dsname in dirs:
        params = dsname.split('_')
        if params[2] in ('theta=10', 'theta=20'):
            # load saved parameters of encoder and decoder
            encoder.load_state_dict(torch.load(os.path.join(cwd, 'checkpoints', 'encoder_'+dsname),
            map_location=lambda storage, loc: storage))
            decoder.load_state_dict(torch.load(os.path.join(cwd, 'checkpoints', 'decoder_'+dsname),
            map_location=lambda storage, loc: storage))
            encoder=encoder.to(device=device)
            decoder=decoder.to(device=device)


            paired_mnist = DoubleUniNormal(dsname)
            loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))
            test_data = torch.from_numpy(paired_mnist.x_test)

            ### Would recommend using this pattern
            use_gpu = torch.cuda.is_available()
            
            # get the true change points and create list for predicted change points
            cps = paired_mnist.y_test
            cps_hat = []
            
            # set up directories. experiment_info is the name of the experiment
            all_dirs = os.listdir(os.path.join(cwd, 'sqerrors'))
            try:
                max_dir = max([int(d[3:]) for d in all_dirs])
            except ValueError:
                max_dir = 0
            new_dir_name = 'run0'+str(max_dir+1) if max_dir <= 8 else 'run'+str(max_dir+1)
            directory_name = os.path.join(cwd, 'sqerrors', new_dir_name)
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            experiment_info = dsname
            # run02: n=200 T=50 seed=10
            # run03: n=500 T=50 seed=10
            # run04: n=500 T=50 seed=100
            # run05: n=500 T=50 no seed

            # run06: n=500 T=50 seed=700
            # run07: n=200 T=50 seed=700
            # run08: n=1000 T=50 seed=700

            # run09: double uninormal


            # run on each test sample X_i
            for i in range(len(test_data)): # Running X_i
                print('Running X_'+str(i))
                X_i = test_data[i].to(device=device)
                
                errors = {}

                minimum_eta = max(1, cps[i]-20)
                maximum_eta = min(cps[i]+20, paired_mnist.T)

                # partial is awesome
                eta_error_calc = partial(get_eta_error, encoder=encoder, X=X_i.detach(), maxlen=paired_mnist.T)

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
                plt.savefig(os.path.join('sqerrors', new_dir_name, experiment_info+'_X'+str(i)+'.jpg'))
                plt.close()
            


            
            with open(os.path.join('sqerrors', new_dir_name, experiment_info+'.txt'), 'w') as cps_r:
                for tmp in cps_hat:
                    cps_r.write('{} '.format(tmp))
                cps_r.write('\n')
                for tmp in cps:
                    cps_r.write('{} '.format(tmp))


            print(cps)
            print(cps_hat)
