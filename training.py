import os
import numpy as np
from itertools import cycle
import pickle
import random

import torch
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from utils import weights_init
from utils import transform_config
from networks import Encoder, Decoder
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence
from alternate_data_loader import DoubleUniNormal, DoubleMulNormal, experiment3

def training_procedure(FLAGS):
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    """
    variable definition
    """
    X = torch.FloatTensor(FLAGS.batch_size, 784)

    '''
    run on GPU if GPU is available
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device=device)
    decoder.to(device=device)
    X = X.to(device=device)
    
    """
    optimizer definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    """
    
    """
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL_divergence_loss\tClass_KL_divergence_loss\n')

    # load data set and create data loader instance
    dirs = os.listdir(os.path.join(os.getcwd(), 'data'))
    print('Loading double multivariate normal time series data...')
    for dsname in dirs:
        params = dsname.split('_')
        if params[2] in ('theta=-1'):
            print('Running dataset ', dsname)
            ds = DoubleMulNormal(dsname)
            # ds = experiment3(1000, 50, 3)
            loader = cycle(DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True))

            # initialize summary writer
            writer = SummaryWriter()

            for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
                print()
                print('Epoch #' + str(epoch) + '........................................................')

                # the total loss at each epoch after running iterations of batches
                total_loss = 0

                for iteration in range(int(len(ds) / FLAGS.batch_size)):
                    # load a mini-batch
                    image_batch, labels_batch = next(loader)
                    
                    # set zero_grad for the optimizer
                    auto_encoder_optimizer.zero_grad()

                    X.copy_(image_batch)

                    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(X))
                    grouped_mu, grouped_logvar = accumulate_group_evidence(
                        class_mu.data, class_logvar.data, labels_batch, FLAGS.cuda
                    )

                    # kl-divergence error for style latent space
                    style_kl_divergence_loss = FLAGS.kl_divergence_coef * (
                            - 0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
                    )
                    style_kl_divergence_loss /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
                    style_kl_divergence_loss.backward(retain_graph=True)

                    # kl-divergence error for class latent space
                    class_kl_divergence_loss = FLAGS.kl_divergence_coef * (
                            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
                    )
                    class_kl_divergence_loss /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
                    class_kl_divergence_loss.backward(retain_graph=True)

                    # reconstruct samples
                    """
                    sampling from group mu and logvar for each image in mini-batch differently makes
                    the decoder consider class latent embeddings as random noise and ignore them 
                    """
                    style_latent_embeddings = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
                    class_latent_embeddings = group_wise_reparameterize(
                        training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
                    )

                    reconstructed_images = decoder(style_latent_embeddings, class_latent_embeddings)

                    reconstruction_error = FLAGS.reconstruction_coef * mse_loss(reconstructed_images, Variable(X))
                    reconstruction_error.backward()

                    total_loss += style_kl_divergence_loss + class_kl_divergence_loss + reconstruction_error

                    auto_encoder_optimizer.step()

                    if (iteration + 1) % 50 == 0:
                        print('\tIteration #' + str(iteration))
                        print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                        print('Style KL loss: ' + str(style_kl_divergence_loss.data.storage().tolist()[0]))
                        print('Class KL loss: ' + str(class_kl_divergence_loss.data.storage().tolist()[0]))

                    # write to log
                    with open(FLAGS.log_file, 'a') as log:
                        log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                            epoch,
                            iteration,
                            reconstruction_error.data.storage().tolist()[0],
                            style_kl_divergence_loss.data.storage().tolist()[0],
                            class_kl_divergence_loss.data.storage().tolist()[0]
                        ))

                    # write to tensorboard
                    writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                                    epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)
                    writer.add_scalar('Style KL-Divergence loss', style_kl_divergence_loss.data.storage().tolist()[0],
                                    epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)
                    writer.add_scalar('Class KL-Divergence loss', class_kl_divergence_loss.data.storage().tolist()[0],
                                    epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)

                    if epoch == 0 and (iteration+1) % 50 == 0:
                        torch.save(encoder.state_dict(), os.path.join('checkpoints', 'encoder_'+dsname))
                        torch.save(decoder.state_dict(), os.path.join('checkpoints', 'decoder_'+dsname))

                # save checkpoints after every 10 epochs
                if (epoch + 1) % 10 == 0 or (epoch + 1) == FLAGS.end_epoch:
                    torch.save(encoder.state_dict(), os.path.join('checkpoints', 'encoder_'+dsname))
                    torch.save(decoder.state_dict(), os.path.join('checkpoints', 'decoder_'+dsname))
                
                print('Total loss at current epoch: ', total_loss.item())