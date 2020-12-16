import os
import argparse
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Pool
from functools import partial

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid

import utils
import alternate_data_loader
from networks1 import DFCVAE, z_classifier






# settings
torch.set_printoptions(precision=8)
print = partial(print, flush=True)

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--accumulate_evidence', type=str, default=False, help="accumulate class evidence before producing swapped images")
parser.add_argument('--model_save', type=str, default='dfcvae', help="model save for DFCVAE") # paths to saved models
parser.add_argument('--test_run', type=int) # test run number
parser.add_argument('--save_original', type=bool, default=True) # save the original images
FLAGS = parser.parse_args()


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc


# model definition
model = DFCVAE()
model.load_state_dict(torch.load(os.path.join('/media/renyi/HDD/checkpoints', FLAGS.model_save),
                                    map_location=lambda storage, loc: storage))
model_classify = z_classifier(256)

# use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# move to gpu
model = model.to(device=device)
model_classify = model_classify.to(device=device)

optimizer = torch.optim.Adam(model_classify.parameters(), lr = 0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# load dataset
print('Loading CelebA training data...')
ds = alternate_data_loader.celeba_classification(1000, 10)
loader = DataLoader(dataset=ds, batch_size=128, shuffle=True)

# iterate over test samples X_1, X_2, etc...
for e in range(20):
    epoch_loss = 0
    epoch_correct_pred = 0

    for X, y in loader:
        X = X.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            style_mu, style_logvar, content_mu, content_logvar = model.encode(X)
            style_z = utils.reparameterize(True, style_mu, style_logvar)
            content_z = utils.reparameterize(True, content_mu, content_logvar)
            X_encoded = torch.cat((style_z, content_z), dim=1)
        
        optimizer.zero_grad()
        y_hat = model_classify.forward(X_encoded)
        y_hat = y_hat.view(y_hat.size(0))


        loss = criterion(y_hat, torch.tensor(y, dtype=float).cuda())
        number_corrects = torch.sum(y == torch.round(torch.sigmoid(y_hat)))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_correct_pred += number_corrects.item()
    
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(loader):.5f} | Acc: {epoch_correct_pred/len(loader):.3f}')


ds_test = alternate_data_loader.celeba_test_classification(50, 10)
test_loader = DataLoader(dataset=ds_test, batch_size=128, shuffle=True)
y_pred = []
model_classify.eval()

acc_total = 0

with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device=device)
        y = y.to(device=device)
    
        style_mu, style_logvar, content_mu, content_logvar = model.encode(X)
        style_z = utils.reparameterize(True, style_mu, style_logvar)
        content_z = utils.reparameterize(True, content_mu, content_logvar)
        X_encoded = torch.cat((style_z, content_z), dim=1)
        y_hat = model_classify.forward(X_encoded)
        y_hat = y_hat.view(y_hat.size(0))

        number_corrects = torch.sum(y == torch.round(torch.sigmoid(y_hat)))
        acc_total += number_corrects

print(acc_total // len(test_loader))