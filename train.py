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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Training methods defined here.
A part of GSPNet project.

'''

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import os
import re
import time
import torch

from torch import nn, optim
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import SubsetRandomSampler, SequentialSampler
from glob import iglob, glob
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm

# import models, loss functions and datasets
import models
from models import *
from losses import *
from datasets import *

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


# helper functions
def save_model(model, dest: str, hyps: dict):
    '''
    Save model to local file.

    Args:
        model: trained model
        dest: folder to save trained model
        hyps: hyperparameters of the trained model
    '''
    name = f'mn{hyps["mn"]}-is{hyps["is"]}-os{hyps["os"]}-sl{hyps["sl"]}-bs{hyps["bs"]}-hd{hyps["hd"]}-lr{hyps["lr"]}-nl{hyps["nl"]}-dp{hyps["dp"]}.pt'
    torch.save(model.state_dict(), dest + '/' + name)


# data feeder, type 2, deprecated
def batch_dataset(datadir, seq_len):
    '''
    Batch the neural network data using DataLoader.

    Args:
        datadir: Directory storing tensor data
        seq_len: The sequence length of each batch
    Return:
        DataLoader with batched data
    '''
    # WARNING: this function is deprecated, will remove after 2019 May 1st
    data_iter = iglob(datadir + '/*')

    states = []

    print('Loading dataset...')
    print('Loading training set...')
    for state in tqdm(data_iter, ascii=True):
        state = torch.load(state).numpy()
        states.append(state)

    states = np.array(states)
    states = states.reshape((len(states), -1))
    states = states.astype('float32')
    num_batches = len(states) // seq_len

    # only full batches
    states = states[: num_batches * seq_len]
    features, targets = [], []

    for idx in range(0, (len(states) - seq_len)):
        features.append(states[idx: idx + seq_len])
        targets.append(states[idx + seq_len])

    data = TensorDataset(torch.from_numpy(np.array(features)),
                         torch.from_numpy(np.array(targets)))

    data_loader = torch.utils.data.DataLoader(
        data, shuffle=False, batch_size=batch_size, num_workers=0)

    return data_loader


def check_encoder_dim(mode: str, model, dataset):
    '''
    Check whether the convolutional autoencoder architecture matches data dimension.

    Args:
        mode: `pnf` or `od` mode
        model: convencoder model instance
        dataset: dataset object
    Returns:
        if_match: bool
    '''
    loader = DataLoader(dataset,
                        batch_size=1, num_workers=0, drop_last=True)
    iterator = iter(loader)
    X, y = iterator.next()
    if mode == 'od':
        assert X.size(1) == 1, f'Mode `od`: X expect channel size 1 but get {X.size(1)}.'
    elif mode == 'pnf':
        assert X.size(1) == 3, f'Mode `pnf`: X expect channel size 3 but get {X.size(1)}'


# training function of CNN classification
def train_classifier(model, optimizer, criterion, n_epochs,
                     train_loader, valid_loader, hyps,
                     stop_criterion=20, device='cuda:0',
                     show_every_n_batches=100):
    '''
    Train a CNN classifier with the given hyperparameters.

    Args:
        model:              The PyTorch Module that holds the neural network
        optimizer:          The PyTorch optimizer for the neural network
        criterion:          The PyTorch loss function
        n_epochs:           Total go through of entire dataset
        train_loader:       Training data loader
        valid_loader:       Validation data loader
        hyps:               A dict containing hyperparameters
        stop_criterion:     Early stop variable
        device:             Training device
        show_every_batches: Display loss every this number of time steps
    Returns:
        A trained model. The best model will also be saved locally.
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training on device {device} started at {time.ctime()}')
    # validation constants
    early_stop_count = 0
    valid_loss_min = np.inf

    train_losses = []
    valid_losses = []

    # for plot training loss and validation loss
    tl = []
    vl = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # early stop mechanism:
        if early_stop_count >= stop_criterion:
            print(f'Early stop triggered after {stop_criterion} epochs.')
            break

        for data, label in train_loader:

            # forward, back prop
            if TRAIN_ON_MULTI_GPUS:
                data, label = data.cuda(), label.cuda()
            elif torch.cuda.is_available():
                data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # record loss
            train_losses.append(loss.item() * data.size(0))

            model.eval()

        for v_data, v_label in valid_loader:

            v_data, v_label = v_data.cuda(), v_label.cuda()

            v_output = model(v_data)
            val_loss = criterion(v_output, v_label)

            valid_losses.append(val_loss.item() * data.size(0))

        model.train()
        avg_val_loss = np.mean(valid_losses)
        avg_tra_loss = np.mean(train_losses)

        tl.append(avg_tra_loss)
        vl.append(avg_val_loss)
        # printing loss stats
        print(
            f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Min Val: {valid_loss_min:.4f}',
            flush=True)

        # decide whether to save model or not:
        if avg_val_loss < valid_loss_min:

            print(f'Valid Loss {valid_loss_min:.4f} -> {avg_val_loss:.4f}. \
                    Saving...', flush=True)

            torch.save(model.state_dict(),
                       f'mn{hyps["mn"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-nc{hyps["nc"]}-dp{hyps["dp"]}.pt')

            valid_loss_min = avg_val_loss
            early_stop_count = 0

        else:
            early_stop_count += 1

        # clear
        train_losses = []
        valid_losses = []

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model, (tl, vl)


def run_classifier_training(model_name, epochs, nc, vs, rs,
                            lr=0.001, bs=128, dp=0.5, device='cuda:0'):
    '''
    Main function of cnn classifier training.

    Args:
        model_name: model name
        epochs: number of epochs to train
        nc: number of classes
        vs: validation size, proportion of validation data set
        rs: random seed
        lr: learning_rate
        bs: batch_size
        dp: drop_prob
    '''
    # Training parameters
    epochs = epochs
    learning_rate = 0.001
    batch_size = bs

    # Model parameters
    input_size = 69 * 69 * 3  # <- don't change this value
    drop_prob = 0.5
    # Show stats for every n number of batches
    senb = 5000

    # wrap essential info into dictionary:
    hyps = {
        'mn': model_name,
        'bs': batch_size,
        'lr': learning_rate,
        'nc': nc,
        'dp': drop_prob
    }

    # Initialize data loaders
    data_dir = 'data/2018_15min/tensors'

    # LSTM data loader
    data_set = SnapshotClassificationDatasetRAM(data_dir)

    # split data for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(vs * num_train))

    # shuffle
    np.random.seed(rs)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(data_set, sampler=train_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    valid_loader = DataLoader(data_set, sampler=valid_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    # initialize model
    model = models.__dict__[model_name](n_classes=nc)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.to(device)
    else:
        print('Training on CPU, very long training time is expectable.')

    # optimizer and criterion(loss function)
    if TRAIN_ON_MULTI_GPUS:
        optimizer = optim.SGD(model.module.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # start training
    trained_model, tlvl = train_classifier(model, optimizer, criterion, epochs,
                                           train_loader, valid_loader, hyps)

    # loss plot
    tl, vl = tlvl
    x = np.arange(len(tl))
    # for model 3 of classification only
    x, tl, vl = x[1:], tl[1:], vl[1:]
    train_curve, = plt.plot(x, tl, 'r-', label='train loss')
    valid_curve, = plt.plot(x, vl, 'b-', label='valid loss')
    plt.legend(handler_map={train_curve: HandlerLine2D(numpoints=1)})

    plt.savefig(
        'trained_models' +
        f'/mn{hyps["mn"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-nc{hyps["nc"]}-dp{hyps["dp"]}.png'
    )
    plt.show()


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
    if type(hidden) == tuple:
        h = tuple([each.data for each in hidden])
    else:
        h = hidden.data

    # zero accumulated gradients
    if TRAIN_ON_MULTI_GPUS:
        model.module.zero_grad()
    else:
        model.zero_grad()

    # print(f'input shape: {inp}, target shape: {target}')
    # get the output from the model
    output, h = model(inp, h)

    # perform backpropagation and optimization
    # calculate the loss and perform backprop
    loss = criterion(output, target)
    loss.backward()

    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs
    if TRAIN_ON_MULTI_GPUS:
        nn.utils.clip_grad_norm_(model.module.parameters(), clip)
    else:
        nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h


# training function for sequential prediction
def train_recurrent(model, batch_size, optimizer, criterion,
                    n_epochs, train_loader, valid_loader, hyps, clip=5,
                    stop_criterion=20, show_every_n_batches=1, multi_gpus=True,
                    device='cuda:0'):
    '''
    Train a LSTM model with the given hyperparameters.

    Args:
        model:              The PyTorch Module that holds the neural network
        batch_size:         batch size, integer
        optimizer:          The PyTorch optimizer for the neural network
        criterion:          The PyTorch loss function
        n_epochs:           Total go through of entire dataset
        train_loader:       Training data loader
        valid_loader:       Validation data loader
        hyps:               A dict containing model parameters
        clip:               Clip the overly large gradient
        show_every_batches: Display loss every this number of time steps
        multi_gpus:         Whether have multiple GPUs
        device:             location to put tensor/model
    Returns:
        A trained model. The best model will also be saved locally.
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training on device {device} started at {time.ctime()}')
    # validation constants
    early_stop_count = 0
    valid_loss_min = np.inf

    train_losses = []

    # for plot training loss and validation loss
    tl = []
    vl = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        if TRAIN_ON_MULTI_GPUS and multi_gpus:
            hidden = model.module.init_hidden(batch_size)
        else:
            hidden = model.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # early stop mechanism:
            if early_stop_count >= stop_criterion:
                print(f'Early stop triggered after {stop_criterion} epochs.')
                break

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if batch_i > n_batches:
                break

            # forward, back prop
            # print(f'inputs shape: {inputs.shape} labels shape: {labels.shape}')
            # print(f'inputs dtype: {inputs[0][0][0].dtype} label shape: {labels[0][0].dtype}')
            if TRAIN_ON_MULTI_GPUS and multi_gpus:
                inputs, labels = inputs.cuda(), labels.cuda()
            elif torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)

            #  print(f'Input shape: {inputs.shape}')
            loss, hidden = forward_back_prop(
                model, optimizer, criterion, inputs, labels, hidden, clip
            )

            # record loss
            train_losses.append(loss)

            # print loss every show_every_n_batches batches
            # including validation loss
            if batch_i % show_every_n_batches == 0:
                # get validation loss
                if TRAIN_ON_MULTI_GPUS:
                    val_h = model.module.init_hidden(batch_size)
                else:
                    val_h = model.init_hidden(batch_size)
                valid_losses = []

                # switch to validation mode
                model.eval()

                for v_inputs, v_labels in valid_loader:

                    if TRAIN_ON_MULTI_GPUS and multi_gpus:
                        v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()
                    elif torch.cuda.is_available():
                        v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    # if type is tuple, then the model is LSTM
                    if type(val_h) == tuple:
                        val_h = tuple([each.data for each in val_h])
                    else:
                        val_h = val_h.data

                    v_output, val_h = model(v_inputs, val_h)
                    val_loss = criterion(v_output, v_labels)

                    valid_losses.append(val_loss.item())

                model.train()
                avg_val_loss = np.mean(valid_losses)
                avg_tra_loss = np.mean(train_losses)

                tl.append(avg_tra_loss)
                vl.append(avg_val_loss)
                # printing loss stats
                print(
                    f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.4f} ' +
                    f'| Val Loss: {avg_val_loss:.4f} | Min Val: {valid_loss_min:.4f}',
                    flush=True)

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:

                    print(f'Valid Loss {valid_loss_min:.4f} -> {avg_val_loss:.4f}. Saving...', flush=True)

                    # saving state_dict of model
                    save_model(model, 'trained_models', hyps)

                    valid_loss_min = avg_val_loss
                    early_stop_count = 0

                else:
                    early_stop_count += 1

                train_losses = []
                valid_losses = []

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:.2f} seconds.')
    return model, (tl, vl)


# run functions of this module
def run_recursive_training(model_name, epochs, sl=12, bs=64,
                           lr=0.001, hd=256, nl=2, dp=0.5, device='cuda:0'):
    '''
    Main function of RNNs training.

    Args:
        model_name: model name
        epochs: number of epochs to train
        sl: sequence_length
        bs: batch_size
        lr: learning_rate
        hd: hidden_dim
        nl: n_layers
        dp: drop_prob
    '''
    # LSTM Model Data params
    sequence_length = sl  # number of time slices in a sequence
    clip = 5

    # Training parameters
    epochs = epochs
    learning_rate = lr
    batch_size = bs

    # Model parameters
    input_size = 69 * 69 * 3  # <- don't change this value
    output_size = input_size
    hidden_dim = hd
    # Number of RNN Layers
    n_layers = nl
    drop_prob = dp
    # Show stats for every n number of batches
    senb = 5000

    # wrap essential info into dictionary:
    hyps = {
        'mn': model_name,
        'is': input_size,
        'os': output_size,
        'sl': sequence_length,
        'bs': batch_size,
        'lr': learning_rate,
        'hd': hidden_dim,
        'nl': n_layers,
        'dp': drop_prob
    }

    data_dir = 'data/2018_15min/tensors'
    data_set = S2FDatasetRAM(data_dir, sequence_length)

    # split dataset for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(0.8 * num_train))  # hard coded to 0.8

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_loader = DataLoader(data_set, sampler=train_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    valid_loader = DataLoader(data_set, sampler=valid_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    # initialize model
    model = models.__dict__[model_name](input_size, output_size, hidden_dim,
                                        n_layers=n_layers, drop_prob=drop_prob,
                                        device=device)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.to(device)
    else:
        print('Training on CPU, very long training time is expectable.')

    # optimizer and criterion(loss function)
    if TRAIN_ON_MULTI_GPUS:
        optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # loss:
    # criterion = nn.MSELoss()
    criterion = dich_mse_loss
    # start training
    trained_model, tlvl = train_recurrent(model, batch_size, optimizer,
                                          criterion, epochs, train_loader,
                                          valid_loader, hyps, device=device)

    # loss plot
    tl, vl = tlvl
    x = np.arange(len(tl))

    train_curve, = plt.plot(x, tl, 'r-', label='train loss')
    valid_curve, = plt.plot(x, vl, 'b-', label='valid loss')
    plt.legend(handler_map={train_curve: HandlerLine2D(numpoints=1)})

    plt.savefig(
        'trained_models' + '/' +
        f'mn{hyps["mn"]}-is{hyps["is"]}-os{hyps["os"]}-sl{hyps["sl"]}-bs{hyps["bs"]}-hd{hyps["hd"]}-lr{hyps["lr"]}-nl{hyps["nl"]}-dp{hyps["dp"]}.png'
    )
    plt.show()


def train_encoder(model, optimizer, criterion, n_epochs,
                  loader, hyps, device='cuda:0', show_every_n_batches=100):
    '''
    Train an auto encoder with the given hyperparameters.

    Args:
        model:              The PyTorch Module that holds the neural network
        optimizer:          The PyTorch optimizer for the neural network
        criterion:          The PyTorch loss function
        n_epochs:           Total go through of entire dataset
        loader:             Training data loader
        hyps:               A dict containing hyperparameters
        device:             Training device
        show_every_batches: Display loss every this number of time steps
    Returns:
        A trained model. The best model will also be saved locally.
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training on device {device} started at {time.ctime()}')
    # validation constants
    valid_loss_min = np.inf

    losses = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        for data, label in loader:
            # forward, back prop
            if TRAIN_ON_MULTI_GPUS:
                data, label = data.cuda(), label.cuda()
            elif torch.cuda.is_available():
                data, label = data.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # record loss
            losses.append(loss.item() * data.size(0))

        avg_loss = np.mean(losses)
        # printing loss stats
        print(
            f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_loss:.4f}', flush=True)
        # clear
        losses = []

    torch.save(model.state_dict(), 'trained_models' + '/' +
               f'mn{hyps["mn"]}-is{hyps["is"]}-os{hyps["os"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-hd{hyps["hd"]}-md{hyps["md"]}.pt')

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model


def run_encoder_training(model_name, epochs, data_dir, mode='od',
                         hd=512, lr=0.001, bs=64, dp=0.5, device='cuda:0'):
    '''
    Main function of auto encoder.

    Args:
        model_name: model name
        epochs: number of epochs to train
        data_dir: location of training data
        hd: hidden dim
        lr: learning_rate
        bs: batch_size
    '''
    # Training parameters
    epochs = epochs
    learning_rate = lr
    batch_size = bs

    # Model parameters
    if mode == 'od':
        input_size = 69 * 69 * 1
    elif mode == 'pnf':
        input_size = 69 * 69 * 3
    else:
        raise ValueError('Only `od` and `pnf` are supported.')
    output_size = input_size
    hidden_dim = hd

    # wrap essential info into dictionary:
    hyps = {
        'is': input_size,
        'os': output_size,
        'mn': model_name,
        'hd': hidden_dim,
        'bs': batch_size,
        'lr': learning_rate,
        'md': mode
    }

    # Initialize data loaders
    # LSTM data loader
    if model_name in ['ConvAutoEncoder', 'ConvAutoEncoderShallow', 'VAE']:
        data_set = ConvEncoderDatasetRAM(data_dir)
    else:
        data_set = EncoderDatasetRAM(data_dir)

    # split dataset for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(0.8 * num_train))  # hard coded to 0.8

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    loader = DataLoader(data_set, sampler=train_sampler,
                        batch_size=batch_size, num_workers=0, drop_last=True)

    valid_loader = DataLoader(data_set, sampler=valid_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    # initialize model
    if model_name in ['ConvAutoEncoder', 'ConvAutoEncoderShallow']:
        model = models.__dict__[model_name](hyps['is'], hyps['os'], mode=hyps['md'])
    elif model_name == 'VAE':
        model = models.__dict__[model_name](hyps['md'], h_dim=hyps['hd'], z_dim=32)
    else:
        model = models.__dict__[model_name](hyps['is'], hyps['os'], hidden_dim=hyps['hd'])
    print(model)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.to(device)
    else:
        print('Training on CPU, very long training time is expectable.')

    check_encoder_dim(mode, model, data_set)

    # optimizer and criterion(loss function)
    if TRAIN_ON_MULTI_GPUS:
        optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # criterion = nn.L1Loss()
    # criterion = dich_mse_loss
    criterion = nn.MSELoss()

    # start training
    trained_model = train_encoder(model, optimizer, criterion, epochs, loader,
                                  hyps, device=device)

    return trained_model


if __name__ == '__main__':

    # run_recursive_training('VanillaStateGRU', 5, sl=24, bs=128, lr=0.001,
    #                         hd=4096, nl=2, dp=0.5, device='cuda:1')
    # run_classifier_training(100, 2, 0.1, 0, lr=0.001, bs=1024, dp=0.1)

    data_dir = 'data/2018/15min/tensors'
    run_encoder_training('VAE', 1000, data_dir,
                         mode='od', lr=0.001, hd=256, device='cuda:1')
