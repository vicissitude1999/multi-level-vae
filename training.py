import os
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils import weights_init
import torchvision.transforms as transforms
from networks import Encoder, Decoder
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence
import alternate_data_loader

def training_procedure(FLAGS):
    # define data transformer
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load data set and create data loader instance
    print('Loading CLEVER-change data...')
    ds = alternate_data_loader.clever_change(transform)
    test_split = 0.3 # percentage of test data
    shuffle = True
    random_seed = 42

    indices = list(range(len(ds)))
    split = int(np.floor(test_split*len(ds)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    train_loader = cycle(DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=True, sampler=train_sampler, drop_last=True))
    # test_loader = cycle(DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=True, sampler=test_sampler, drop_last=True))


    # model definition
    encoder = Encoder()
    encoder.apply(weights_init)
    decoder = Decoder()
    decoder.apply(weights_init)

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

    # variable definition and move to GPU if available
    X = torch.FloatTensor(FLAGS.batch_size, ds.data_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device=device)
    decoder.to(device=device)
    X = X.to(device=device)

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
        num_iterations = int(len(ds)) / FLAGS.batch_size

        for iteration in range(num_iterations):
            # set zero_grad for the optimizer
            auto_encoder_optimizer.zero_grad()

            # load a mini-batch
            image_batch, labels_batch = next(train_loader)
            X.copy_(image_batch)

            style_mu, style_logvar, content_mu, content_logvar = encoder(Variable(X))
            # put all content stuff into group in the grouping/evidence-accumulation stage
            group_mu, group_logvar = accumulate_group_evidence(
                content_mu.data, content_logvar.data, labels_batch, FLAGS.cuda
            )

            # kl-divergence error for style latent space
            style_kl = FLAGS.kl_divergence_coef * (
                    - 0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            )
            style_kl /= FLAGS.batch_size
            for i in range(len(ds.data_dim)):
                style_kl /= ds.data_dim[i]
            style_kl.backward(retain_graph=True)

            # kl-divergence error for content/group latent space
            content_kl = FLAGS.kl_divergence_coef * (
                    - 0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
            )
            style_kl /= FLAGS.batch_size
            for i in range(len(ds.data_dim)):
                style_kl /= ds.data_dim[i]
            content_kl.backward(retain_graph=True)

            # reconstruct samples
            """
            sampling from group mu and logvar for each image in mini-batch differently makes
            the decoder consider content latent embeddings as random noise and ignore them 
            """
            # training param means calling the method when training
            style_z = reparameterize(training=True, mu=style_mu, logvar=style_logvar)
            content_z = group_wise_reparameterize(
                training=True, mu=group_mu, logvar=group_logvar, labels_batch=labels_batch, cuda=FLAGS.cuda
            )

            reconstruction = decoder(style_z, content_z)

            reconstruction_error = FLAGS.reconstruction_coef * mse_loss(reconstruction, Variable(X))
            reconstruction_error.backward()

            auto_encoder_optimizer.step()

            total_loss += style_kl + content_kl + reconstruction_error

            # print losses
            if (iteration + 1) % (num_iterations / 5) == 0:
                print('\tIteration #' + str(iteration))
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('Style KL loss: ' + str(style_kl.data.storage().tolist()[0]))
                print('content KL loss: ' + str(content_kl.data.storage().tolist()[0]))
                print('Total loss ' + total_loss.item())

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

        # save checkpoints after every epochs
        torch.save(encoder.state_dict(), os.path.join('checkpoints', 'encoder_clever'))
        torch.save(decoder.state_dict(), os.path.join('checkpoints', 'decoder_clever'))