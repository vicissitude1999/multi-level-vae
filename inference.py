import os
import argparse
import numpy as np
from itertools import cycle

import torch
import random
import pickle
from torchvision import datasets
from torch.autograd import Variable
from torch.distributions import Normal
import math
from alternate_data_loader import MNIST_Paired
from alternate_data_loader import DoubleUniNormal
import torch.optim as optim
from utils import accumulate_group_evidence, group_wise_reparameterize, reparameterize, mse_loss

import matplotlib.pyplot as plt
from utils import transform_config
from networks import Encoder, Decoder
from torch.utils.data import DataLoader

from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser()
torch.autograd.set_detect_anomaly(True)

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
parser.add_argument('--reference_data', type=str, default='fixed', help="generate output using random digits or fixed reference")
parser.add_argument('--accumulate_evidence', type=str, default=False, help="accumulate class evidence before producing swapped images")
parser.add_argument('--batch_size', type=int, default=1, help="batch size for training")
parser.add_argument('--image_size', type=int, default=28, help="height and width of the image")
parser.add_argument('--num_channels', type=int, default=1, help="number of channels in the images")
parser.add_argument('--num_classes', type=int, default=10, help="number of classes in the dataset")

parser.add_argument('--style_dim', type=int, default=1, help="dimension of varying factor latent space")
parser.add_argument('--class_dim', type=int, default=1, help="dimension of common factor latent space")

# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")

torch.set_printoptions(precision=8)

def run_through_network(X, labels_batch):
    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
    grouped_mu, grouped_logvar = accumulate_group_evidence(
        class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
    )
    importance_sampling(X[0], style_mu[0], style_logvar[0], class_mu[0], class_logvar[0])

    # reconstruct samples
    style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
    class_latent_embeddings = group_wise_reparameterize(
        training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
    )

    reconstructed_images = decoder(style_latent_embeddings, class_latent_embeddings)

    return reconstructed_images

def importance_sampling(sample, style_mu, style_logvar, grouped_mu, grouped_logvar):
    print(style_mu)
    print(style_logvar.exp())

    print(grouped_mu)
    print(grouped_logvar.exp())
    #std = logvar.mul(0.5).exp_()
    eps_s = torch.FloatTensor(128, 1)
    eps_s.normal_()

    eps_g = torch.FloatTensor(128, 1)
    eps_g.normal_()

    q_s = Normal(0, 1).log_prob(eps_s).exp()
    print(q_s)
    q_g = Normal(0, 1).log_prob(eps_g).exp()

    reparam_s = eps_s.mul(style_logvar.mul(0.5).exp()).add(style_mu)
    reparam_g = eps_g.mul(grouped_logvar.mul(0.5).exp()).add(grouped_mu)

    p_s = Normal(0, 1).log_prob(reparam_s).exp()
    p_g = Normal(0, 1).log_prob(reparam_g).exp()

    reconstructed = decoder(reparam_s, reparam_g)

    px_cond = Normal(0, 1).log_prob(sample - reconstructed).exp()

    summand = px_cond * p_s * p_g / q_s / q_g
    print(summand)

    likelihood = torch.sum(summand) / summand.size(0)
    



FLAGS = parser.parse_args()

if __name__ == '__main__':
    """
    model definitions
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)

    encoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.encoder_save), map_location=lambda storage, loc: storage))
    decoder.load_state_dict(
        torch.load(os.path.join('checkpoints', FLAGS.decoder_save), map_location=lambda storage, loc: storage))

    encoder.cuda()
    decoder.cuda()

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')

    # load data set and create data loader instance
    '''
    print('Loading MNIST paired dataset...')
    paired_mnist = MNIST_Paired(root='mnist', download=True, train=False, transform=transform_config)
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))
    image_array = []
    for i in range(0, 11):
        image_array.append([])

    # add a blank image in the top row
    image_array[0].append(np.zeros((28, 28, 3), dtype=np.float32))

    sampled_classes = list(paired_mnist.data_dict.keys())
    sampled_classes.sort()

    if FLAGS.reference_data == 'random':
        count = 1
        # fill the top row and first column of the grid with reference images
        for class_sample in sampled_classes:
            class_image = random.SystemRandom().choice(paired_mnist.data_dict[class_sample])
            class_image = np.transpose(class_image.numpy(), (1, 2, 0))
            class_image = np.concatenate((class_image, class_image, class_image), axis=2)

            # add image in the top row
            image_array[0].append(class_image)

            # add image in the first column
            image_array[count].append(class_image)

            count += 1

    elif FLAGS.reference_data == 'fixed':
        with open("reference_images.pkl", "rb") as fp:
            reference_images = pickle.load(fp)

        count = 1
        # fill the top row and first column of the grid with reference images
        for i in range(0, 10):
            # add image in the top row
            image_array[0].append(reference_images[i])

            # add image in the first column
            image_array[count].append(reference_images[i])

            count += 1

    # get class specified label for entire top row at once
    specified_factor_images = []
    for i in range(1, 11):
        specified_factor_images.append(image_array[0][i][:, :, 0])

    specified_factor_images = np.asarray(specified_factor_images)
    specified_factor_images = np.expand_dims(specified_factor_images, axis=3)
    specified_factor_images = np.transpose(specified_factor_images, (0, 3, 1, 2))
    specified_factor_images = torch.FloatTensor(specified_factor_images)
    specified_factor_images = specified_factor_images.contiguous()

    if FLAGS.accumulate_evidence:
        # sample a big batch, accumulate evidence and use that for class embeddings
        image_batch, _, labels_batch = next(loader)
        _, __, class_mu, class_logvar = encoder(Variable(image_batch))

        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
        )

        accumulated_class_latent_embeddings = group_wise_reparameterize(
            training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
        )

        # select class latent embeddings for 10 digits sorted by class labels
        class_latent_embeddings = []
        for i in range(0, 10):
            index = np.where(labels_batch.data.numpy() == i)[0][0]
            class_latent_embeddings.append(accumulated_class_latent_embeddings[index])

        class_latent_embeddings = torch.stack(class_latent_embeddings)
    else:
        # simply use 10 images selected from grid to produce class embeddings (no evidence accumulation)
        _, __, class_mu, class_logvar = encoder(Variable(specified_factor_images))

        labels_batch = torch.LongTensor(range(0, 10))

        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
        )

        class_latent_embeddings = group_wise_reparameterize(
            training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
        )

    # generate all possible combinations using the encoder and decoder architecture in the grid
    for row in range(1, 11):
        style_image = image_array[row][0]
        style_image = np.transpose(style_image, (2, 0, 1))
        style_image = torch.FloatTensor(style_image)
        style_image = style_image.contiguous()
        style_image = style_image[0, :, :]
        style_image = style_image.view(1, 1, 28, 28)

        style_mu, style_logvar, _, __ = encoder(Variable(style_image))
        style_latent_embeddings = reparameterize(training=False, mu=style_mu, logvar=style_logvar)

        for col in range(1, 11):
            specified_factor_temp = class_latent_embeddings[col - 1]
            specified_factor_temp = specified_factor_temp.view(1, FLAGS.class_dim)
            reconstructed_x = decoder(style_latent_embeddings, specified_factor_temp)
            reconstructed_x = np.transpose(reconstructed_x.data.numpy(), (0, 2, 3, 1))[0]
            reconstructed_x = np.concatenate((reconstructed_x, reconstructed_x, reconstructed_x), axis=2)

            image_array[row].append(reconstructed_x)

    # plot
    image_array = np.asarray(image_array)
    print(image_array.shape)

    fig = plt.figure(1)

    grid = ImageGrid(fig, 111, nrows_ncols=[11, 11], axes_pad=0.)

    temp_list = []

    for i in range(0, 11):
        for j in range(0, 11):
            temp_list.append(image_array[i][j])

    for i in range(121):
        grid[i].axis('off')
        grid[i].imshow(temp_list[i])

    plt.savefig('reconstructed_images/inference.png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.clf()
    '''

    print('Loading double uninormal dataset...')
    paired_mnist = DoubleUniNormal('DoubleUniNormal_theta=1_n=1500')
    loader = cycle(DataLoader(paired_mnist, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))
    X = torch.FloatTensor(FLAGS.batch_size, 1)

    # test data 
    test_data = torch.from_numpy(paired_mnist.x_test).float()

    for i in range(len(test_data)):
        if i == 1: # Running on the i-th test data
            print('Running X_'+str(i))
            print(paired_mnist.y_test[i])
            print()
            X_i = test_data[i] # 1-d vector
            l = X_i.size(0)

            errors = []

            for eta in range(10, 90):
                print('\tRunning eta =', eta)
                g1 = X_i[0:eta].view(eta, -1).cuda()
                g2 = X_i[eta:l].view(l-eta, -1).cuda()

                total_error = 0
                for g in [g1, g2]:
                    style_mu, _, class_mu, class_logvar = encoder(g)
                    grouped_mu, _ = accumulate_group_evidence(
                        class_mu.data, class_logvar.data, torch.zeros(g.size(0), 1)
                    )

                    decoder_style_input = torch.tensor(style_mu, requires_grad = True, device='cuda')
                    decoder_content_input = torch.tensor(grouped_mu[0], requires_grad = True, device='cuda')

                    content = decoder_content_input.expand(g.size(0), 1)

                    optimizer = optim.Adam(
                        [decoder_style_input, decoder_content_input],
                        lr = 0.01 # this may be an important parameter
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
                
                errors.append(total_error)

                with open('errors.txt', 'w') as f:
                    for e in errors:
                        f.write("%s\n" % e.item())


            
    '''
    image_batch, labels_batch = next(loader)

    X.copy_(image_batch)

    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
    
    grouped_mu, grouped_logvar = accumulate_group_evidence(
        class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
    )

    # reconstruct samples
    style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
    class_latent_embeddings = group_wise_reparameterize(
        training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
    )

    reconstructed_images = decoder(style_latent_embeddings, class_latent_embeddings)

    print(X.view(1, -1))
    print(reconstructed_images.view(1, -1))
    '''