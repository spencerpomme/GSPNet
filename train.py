'''
Copyright <2019> <COPYRIGHT Pingcheng Zhang>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

##============================================================================##

'''

import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from glob import glob, iglob


# helper functions
def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)


def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)


def forward_back_prop(model, optimizer, criterion, inp, target, hidden, clip):
    """
    Forward and backward propagation on the neural network.

    Args:
        model:     The PyTorch Module that holds the neural network
        optimizer: The PyTorch optimizer for the neural network
        criterion: The PyTorch loss function
        inp:       A batch of input to the neural network
        target:    The target output for the batch of input
        hidden:    Hidden state
        clip:      Clip the overly large gradient

    Returns:
        The loss and the latest hidden state Tensor
    """
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in hidden])

    # zero accumulated gradients
    model.zero_grad()

    # print(f'input shape: {inp}, target shape: {target}')
    # get the output from the model
    output, h = model(inp, h)

    # perform backpropagation and optimization
    # calculate the loss and perform backprop
    loss = criterion(output, target)
    loss.backward()

    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs / LSTMs
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h


def train_rnn(model, batch_size, optimizer, criterion, n_epochs,
              train_loader, valid_loader, clip=5, stop_criterion=5,
              show_every_n_batches=100):
    '''
    Train a RNN model with the given hyperparameters.

    Args:
        model:
        batch_size:
        optimizer:
        criterion:
        n_epochs:
        train_loader:
        valid_loader:
        clip:
        show_every_batches:

    Returns:
        A trained model
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training started at {time.ctime()}')
    # validation constants
    early_stop_count = 0
    valid_loss_min = np.inf

    train_losses = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        hidden = model.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if(batch_i > n_batches):
                break

            # forward, back prop
            # print(f'inputs shape: {inputs.shape} labels shape: {labels.shape}')
            # print(f'inputs dtype: {inputs[0][0][0].dtype} label shape: {labels[0][0].dtype}')
            inputs, labels = inputs.cuda(), labels.cuda()

            loss, hidden = forward_back_prop(
                model, optimizer, criterion, inputs, labels, hidden, clip
            )

            # record loss
            train_losses.append(loss)

            # print loss every show_every_n_batches batches
            # including validation loss
            if batch_i % show_every_n_batches == 0:
                # get validation loss
                val_h = model.init_hidden(batch_size)
                valid_losses = []

                # switch to validation mode
                model.eval()

                for v_inputs, v_labels in valid_loader:

                    v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    v_output, val_h = model(v_inputs, val_h)
                    val_loss = criterion(v_output, v_labels)

                    valid_losses.append(val_loss.item())

                model.train()
                avg_val_loss = np.mean(valid_losses)
                # printing loss stats
                print(
                    f'Epoch: {epoch_i:>4}/{n_epochs:<4}  Loss: {np.mean(batch_losses)}  Val Loss {avg_val_loss}')

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:
                    print(
                        f'Valid Loss {valid_loss_min:4f} -> {avg_val_loss:4f}. Saving...')
                    torch.save(model.state_dict(),
                               './trained_models/best_model.pt')
                train_losses = []
                valid_losses = []

    # returns a trained model
    return model


if __name__ == '__main__':

    print('In main of train.py')
