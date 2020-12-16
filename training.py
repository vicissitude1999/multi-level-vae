import os
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks1 import DFCVAE
import utils
from utils import reparameterize, group_wise_reparameterize, accumulate_group_evidence
import alternate_data_loader

def training_procedure(FLAGS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data set and create data loader instance
    print('Loading CelebA training data...')
    ds = alternate_data_loader.celeba_test_change_person(1000, 10)
    train_loader = DataLoader(ds, batch_size=FLAGS.batch_size, drop_last=True)

    # model definition
    model = DFCVAE()
    model.to(device=device)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    # optimizer definition
    auto_encoder_optimizer = optim.Adam(
        model.parameters(),
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

        for batch_index, (X, y) in enumerate(train_loader):
            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()

            # move to cuda
            X = X.to(device=device)
            y = y.to(device=device)

            style_mu, style_logvar, content_mu, content_logvar = model.encode(X)
            # put all content stuff into group in the grouping/evidence-accumulation stage
            group_mu, group_logvar = accumulate_group_evidence(
                content_mu.data, content_logvar.data, y
            )

            # KL-divergence errors
            style_kl = -0.5*torch.sum(1+style_logvar-style_mu.pow(2)-style_logvar.exp())
            content_kl = -0.5*torch.sum(1+group_logvar-group_mu.pow(2)-group_logvar.exp())
            style_kl /= FLAGS.batch_size * np.prod(ds.data_dim)
            content_kl /= FLAGS.batch_size * np.prod(ds.data_dim)
            """
            sampling from group mu and logvar for each image in mini-batch differently makes
            the decoder consider content latent embeddings as random noise and ignore them 
            """
            # reconstruction error
            style_z = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
            content_z = group_wise_reparameterize(
                training=True, mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=FLAGS.cuda
            )
            reconstruction = model.decode(style_z, content_z)
            reconstruction_error = utils.mse_loss(reconstruction, X)
            # feature loss
            reconstruction_features = model.extract_features(reconstruction)
            input_features = model.extract_features(X)
            feature_loss = 0.0
            for (r, i) in zip(reconstruction_features, input_features):
                feature_loss += utils.mse_loss(r, i)

            # total_loss
            loss = 1*(reconstruction_error+feature_loss) + 1*(style_kl+content_kl)
            loss.backward()

            auto_encoder_optimizer.step()

            total_loss += loss.detach()

            # print losses
            if (iteration + 1) % 10 == 0:
                print('\tIteration #' + str(iteration))
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('Style KL loss: ' + str(style_kl.data.storage().tolist()[0]))
                print('Content KL loss: ' + str(content_kl.data.storage().tolist()[0]))
                print('Feature loss: ' + str(feature_loss.data.storage().tolist()[0]))
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
        torch.save(model.state_dict(), os.path.join('/media/renyi/HDD/checkpoints', FLAGS.model_save))