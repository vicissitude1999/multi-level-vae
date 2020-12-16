import os
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks1 import Encoder, Decoder
import utils
from utils import reparameterize, group_wise_reparameterize, accumulate_group_evidence
import alternate_data_loader

def training_procedure(FLAGS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data set and create data loader instance
    print('Loading CLEVR-change training data...')
    ds = alternate_data_loader.clever_change(utils.transform_config1)
    train_sampler, _ = utils.subset_sampler(ds, 10, 0.3, True, random_seed=42)
    train_loader = DataLoader(ds, batch_size=FLAGS.batch_size, sampler=train_sampler, drop_last=True)

    # model definition
    encoder = Encoder()
    decoder = Decoder()
    # encoder.apply(utils.weights_init)
    # decoder.apply(utils.weights_init)
    encoder.to(device=device)
    decoder.to(device=device)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    # optimizer definition
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    # create dirs, log file, etc
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL\tContent_KL\n')
    # initialize summary writer
    writer = SummaryWriter()


    # start training
    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print()
        print('Epoch #' + str(epoch) + '........................................................')

        # the total loss at each epoch after running iterations of batches
        total_loss = 0
        iteration = 0

        for i in range(300):
            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()

            # load a mini-batch
            X = torch.empty(size=(2,) + ds.data_dim)
            y = torch.empty(size=(2,))
            X[0] = ds[32][0]
            X[1] = ds[32][0]
            y[0] = ds[32][1]
            y[1] = ds[32][1]
            X = X.to(device=device)
            y = y.to(device=device)

            style_mu, style_logvar, content_mu, content_logvar = encoder(X)
            # put all content stuff into group in the grouping/evidence-accumulation stage
            group_mu, group_logvar = accumulate_group_evidence(
                content_mu.data, content_logvar.data, y, FLAGS.cuda
            )

            # kl-divergence error for style latent space
            style_kl = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            style_kl /= FLAGS.batch_size * np.prod(ds.data_dim)
            style_kl.backward(retain_graph=True)

            # kl-divergence error for content/group latent space
            content_kl = -0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
            content_kl /= FLAGS.batch_size * np.prod(ds.data_dim)
            content_kl.backward(retain_graph=True)

            # reconstruct samples
            """
            sampling from group mu and logvar for each image in mini-batch differently makes
            the decoder consider content latent embeddings as random noise and ignore them 
            """
            # training param means calling the method when training
            style_z = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
            content_z = group_wise_reparameterize(
                training=True, mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=FLAGS.cuda
            )

            reconstruction = decoder(style_z, content_z)

            reconstruction_error = utils.mse_loss(reconstruction, X)
            reconstruction_error.backward()

            auto_encoder_optimizer.step()

            total_loss += style_kl.detach().item() + content_kl.detach().item() + reconstruction_error.detach().item()

            # print losses
            if (iteration + 1) % 50 == 0:
                print('\tIteration #' + str(iteration))
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('Style KL loss: ' + str(style_kl.data.storage().tolist()[0]))
                print('Content KL loss: ' + str(content_kl.data.storage().tolist()[0]))
            iteration += 1

            # write to log
            with open(FLAGS.log_file, 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                    epoch,
                    iteration,
                    reconstruction_error.data.storage().tolist()[0],
                    style_kl.data.storage().tolist()[0],
                    content_kl.data.storage().tolist()[0]
                ))

            # write to tensorboard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                            epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Style KL-Divergence loss', style_kl.data.storage().tolist()[0],
                            epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('content KL-Divergence loss', content_kl.data.storage().tolist()[0],
                            epoch * (int(len(ds) / FLAGS.batch_size) + 1) + iteration)

        print('Total KL loss: ' + str(total_loss))

        # save checkpoints after at every epoch
        torch.save(encoder.state_dict(), os.path.join('checkpoints', FLAGS.encoder_save))
        torch.save(decoder.state_dict(), os.path.join('checkpoints', FLAGS.decoder_save))