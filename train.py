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
def create_dir(directory: str):
    '''
    Helper function to create directory
    Args:
        directory: a string describing the to be created dir
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

    except OSError:
        print('Error: Creating directory. ' + directory)
        raise OSError


def save_model(model, dest: str, hyps: dict):
    '''
    Save model to local file.

    Args:
        model: trained model
        dest: folder to save trained model
        hyps: hyperparameters of the trained model
    '''
    name = ''
    mn = hyps["mn"]
    name += f'mn{hyps["mn"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder',
              'ConvAutoEncoder', 'ConvAutoEncoderShallow']:
        name += f'-os{hyps["os"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder']:
        name += f'-is{hyps["is"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder',
              'SparseAutoEncoder', 'SparseConvAutoEncoder']:
        name += f'-hd{hyps["hd"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN']:
        name += f'-nl{hyps["nl"]}-dp{hyps["dp"]}-sl{hyps["sl"]}'
    if mn in ['ConvClassifier', 'MLPClassifier']:
        name += f'-nc{hyps["nc"]}'
    if mn in ['ConvAutoEncoder', 'ConvAutoEncoderShallow', 'VAE',
              'SparseAutoEncoder', 'SparseConvAutoEncoder']:
        name += f'-md{hyps["md"]}'
    if mn in ['VAE']:
        name += f'-zd{hyps["z_dim"]}'
    if mn in ['GAN']:
        name += f'-zs{hyps["zs"]}'
        name += f'-ss{hyps["ss"]}'
        name += f'-cd{hyps["cd"]}'
        name += f'-vs{hyps["vs"]}'
        name += f'-md{hyps["md"]}'

    name += f'-bs{hyps["bs"]}-lr{hyps["lr"]}.pt'
    create_dir(dest)
    dest = dest + '/' + name
    torch.save(model.state_dict(), dest)


def get_curve_name(dest: str, hyps: dict):
    '''
    Generate training loss curve image name.

    Args:
        dest: folder to save trained model
        hyps: hyperparameters of the trained model
    '''
    name = ''
    mn = hyps["mn"]
    name += f'mn{hyps["mn"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder',
              'ConvAutoEncoder', 'ConvAutoEncoderShallow']:
        name += f'-os{hyps["os"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder']:
        name += f'-is{hyps["is"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN', 'AutoEncoder',
              'SparseAutoEncoder']:
        name += f'-hd{hyps["hd"]}'
    if mn in ['VanillaLSTM', 'VanillaGRU', 'EmbedRNN']:
        name += f'-nl{hyps["nl"]}-dp{hyps["dp"]}-sl{hyps["sl"]}'
    if mn in ['ConvClassifier', 'MLPClassifier']:
        name += f'-nc{hyps["nc"]}'
    if mn in ['ConvAutoEncoder', 'ConvAutoEncoderShallow', 'VAE',
              'SparseAutoEncoder']:
        name += f'-md{hyps["md"]}'
    if mn in ['VAE']:
        name += f'-zd{hyps["z_dim"]}'

    name += f'-bs{hyps["bs"]}-lr{hyps["lr"]}.png'
    return dest + '/' + name


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
            train_losses.append(loss.item())

            model.eval()

        for v_data, v_label in valid_loader:

            v_data, v_label = v_data.to(device), v_label.to(device)

            v_output = model(v_data)
            val_loss = criterion(v_output, v_label)

            valid_losses.append(val_loss.item())

        model.train()
        avg_val_loss = np.mean(valid_losses)
        avg_tra_loss = np.mean(train_losses)

        tl.append(avg_tra_loss)
        vl.append(avg_val_loss)
        # printing loss stats
        print(
            f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Min Val: {valid_loss_min:.6f}',
            flush=True)

        # decide whether to save model or not:
        if avg_val_loss < valid_loss_min:

            print(f'Valid Loss {valid_loss_min:.6f} -> {avg_val_loss:.6f}. \
                    Saving...', flush=True)

            save_model(model, 'trained_models', hyps)

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


def run_classifier_training(model_name, data_dir, epochs, bs, vs, lr, nc,
                            dp=0.5, device='cuda:0'):
    '''
    Main function of cnn classifier training.

    Args:
        model_name: model name
        data_dir: data source location
        epochs: number of epochs to train
        bs: batch_size
        vs: validation size, proportion of validation data set
        lr: learning_rate
        nc: number of classes
        dp: drop_prob
        device: GPU or CPU
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

    # LSTM data loader
    data_set = SnapshotClassificationDatasetRAM(data_dir)

    # split data for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(vs * num_train))

    # shuffle
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
                                           train_loader, valid_loader, hyps, device=device)

    # loss plot
    tl, vl = tlvl
    x = np.arange(len(tl))
    # for model 3 of classification only
    x, tl, vl = x[1:], tl[1:], vl[1:]
    train_curve, = plt.plot(x, tl, 'r-', label='train loss')
    valid_curve, = plt.plot(x, vl, 'b-', label='valid loss')
    plt.legend(handler_map={train_curve: HandlerLine2D(numpoints=1)})

    curve_name = get_curve_name('trained_models', hyps)
    plt.savefig(curve_name)
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
                    f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.6f} ' +
                    f'| Val Loss: {avg_val_loss:.6f} | Min Val: {valid_loss_min:.6f}',
                    flush=True)

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:

                    print(f'Valid Loss {valid_loss_min:.6f} -> {avg_val_loss:.6f}. Saving...', flush=True)

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
def run_recursive_training(model_name, data_dir, epochs, bs, vs, lr, sl=12,
                           hd=256, nl=2, dp=0.5, device='cuda:0'):
    '''
    Main function of RNNs training.

    Args:
        model_name: model name
        data_dir: data source location
        epochs: number of epochs to train
        bs: batch_size
        vs: validation proportion
        lr: learning_rate
        sl: sequence_length
        hd: hidden_dim
        nl: n_layers
        dp: drop_prob
        device: training hardware, GPU or CPU
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

    data_set = S2FDatasetRAM(data_dir, sequence_length)

    # split dataset for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(vs * num_train))  # hard coded to 0.8

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
    curve_name = get_curve_name('trained_models', hyps)
    plt.savefig(curve_name)
    plt.show()


def train_encoder(model, optimizer, criterion, n_epochs, loader, hyps,
                  early_stop_count=20, device='cuda:0', show_every_n_epochs=10):
    '''
    Train an auto encoder with the given hyperparameters.

    Args:
        model:              The PyTorch Module that holds the neural network
        optimizer:          The PyTorch optimizer for the neural network
        criterion:          The PyTorch loss function
        n_epochs:           Total go through of entire dataset
        early_stop_count:   Early stop number
        loader:             Training data loader
        hyps:               A dict containing hyperparameters
        device:             Training device
        show_every_batches: Display loss every this number of time steps
    Returns:
        A trained model. The best model will also be saved locally.
    '''

    torch.cuda.empty_cache()
    start = time.time()
    print(f'Training on device {device} started at {time.ctime()}')

    loss_min = np.inf
    losses = []
    stop = 0
    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        if stop >= early_stop_count:
            print(f'Stop converging for {stop} epochs, early stop triggered.')
            break

        for data, _ in loader:
            if TRAIN_ON_MULTI_GPUS:
                data = data.cuda()
            elif torch.cuda.is_available():
                data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            # print(f'output.shape -> {output.shape} | data.shape -> {data.shape}')
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch_i % show_every_n_epochs == 0:
            avg_loss = np.mean(losses)
            print(f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_loss:.6f}')

            if avg_loss < loss_min:

                print(f'Valid Loss {loss_min:.6f} -> {avg_loss:.6f}. Saving...')

                # saving state_dict of model
                save_model(model, 'trained_models', hyps)

                loss_min = avg_loss
                stop = 0
            else:
                stop += 1

            losses = []

    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model


def train_vae(model, optimizer, criterion, n_epochs,
              loader, hyps, device='cuda:0', show_every_n_batches=100):
    '''
    Train an VAE with the given hyperparameters.

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
    bces = []
    klds = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        for data, _ in loader:
            # forward, back prop
            if TRAIN_ON_MULTI_GPUS:
                data = data.cuda()
            elif torch.cuda.is_available():
                data = data.to(device)

            recon_images, mu, logvar = model(data)
            # print(f'label.shape -> {label.shape} | recon_images.shape -> {recon_images.shape}')
            loss, bce, kld = criterion(recon_images, data, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            losses.append(loss.item())
            bces.append(bce.item())
            klds.append(kld.item())

        al = np.mean(losses)
        ab = np.mean(bces)
        ak = np.mean(klds)
        # printing loss stats
        print(
            f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {al:.6f} | BCE: {ab:.6f} | KLD: {ak:.6f}', flush=True)
        # clear
        losses = []

    save_model(model, 'trained_models', hyps)

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model


def run_encoder_training(model_name, data_dir, epochs, bs, vs, lr, mode='od',
                         hd=512, device='cuda:0'):
    '''
    Main function of auto encoder.

    Args:
        model_name: model name
        data_dir: location of training data
        epochs: number of epochs to train
        bs: batch_size
        vs: validation size
        lr: learning rate
        mode: pnf or od
        hd: hidden dim
        device: where to train the model
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
    if model_name in ['ConvAutoEncoder',
                      'ConvAutoEncoderShallow',
                      'VAE',
                      'SparseAutoEncoder',
                      'SparseConvAutoEncoder']:
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
    # This part is currently very mixed. Change in the future.
    if model_name in ['ConvAutoEncoder', 'ConvAutoEncoderShallow']:
        model = models.__dict__[model_name](hyps['os'], mode=hyps['md'])
    elif model_name == 'VAE':
        model = models.__dict__[model_name](hyps['md'], hidden_dim=hyps['hd'], z_dim=32)
    elif model_name in ['SparseConvAutoEncoder', 'SparseAutoEncoder']:
        model = models.__dict__[model_name](hyps['md'], hidden_dim=hyps['hd'])
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

    if model_name == 'VAE':
        criterion = vae_loss
        trained_model = train_vae(model, optimizer, criterion, epochs, loader,
                                  hyps, device=device)

    else:
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = dich_mse_loss
        # start training
        trained_model = train_encoder(
            model, optimizer, criterion, epochs, loader, hyps, device=device
        )

    return trained_model


def train_GAN(D, G, d_optimizer, g_optimizer, n_epochs, z_size,
              train_loader, valid_loader, sample_size, hyps, device='cuda:0',
              print_every=100):
    '''
    GAN training function.

    Args:
        D:
        G:
        d_optimizer:
        g_optimizer:
        n_epochs:
        z_size:
        train_loader:
        valid_loader:
        sample_size:
        hyps:
        device:
        print_every:
    Returns:
        trained model: G and D
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training on device {device} started at {time.ctime()}')

    # keep track of loss and generated, "fake" samples
    samples = []
    truths = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()

    # train the network
    for epoch in range(n_epochs):

        for batch_i, (real_images, _) in enumerate(train_loader):

            batch_size = real_images.size(0)
            # important rescaling step
            real_images = scale(real_images)

            # ============================================ #
            #            TRAIN THE DISCRIMINATOR           #
            # ============================================ #
            d_optimizer.zero_grad()

            # 1. Train with real images
            # Compute the discriminator losses on real images
            if TRAIN_ON_MULTI_GPUS:
                real_images = real_images.cuda()
            elif torch.cuda.is_available():
                real_images = real_images.to(device)

            D_real = D(real_images)
            d_real_loss = real_loss(D_real)

            # 2. Train with fake images
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            # move x to GPU, if available
            if TRAIN_ON_MULTI_GPUS:
                z = z.cuda()
            elif torch.cuda.is_available():
                z = z.to(device)

            fake_images = G(z)

            # Compute the discriminator losses on fake images
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake)

            # add up loss and perform backprop
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ========================================= #
            #            TRAIN THE GENERATOR            #
            # ========================================= #
            g_optimizer.zero_grad()

            # 1. Train with fake images and flipped labels
            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if TRAIN_ON_MULTI_GPUS:
                z = z.cuda()
            elif torch.cuda.is_available():
                z = z.to(device)
            fake_images = G(z)

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake)  # use real loss to flip labels

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print(f'Epoch [{epoch+1:5d}/{n_epochs:5d}] |' +
                      f' d_loss: {d_loss.item():6.4f} | g_loss: {g_loss.item():6.4f}')

        # AFTER EACH EPOCH
        # generate and save sample, fake images
        G.eval()  # for generating samples
        if TRAIN_ON_MULTI_GPUS:
            fixed_z = fixed_z.cuda()
        elif torch.cuda.is_available():
            fixed_z = fixed_z.to(device)

        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    save_model(D, 'trained_models/GAN/D', hyps)
    save_model(G, 'trained_models/GAN/G', hyps)

    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')

    with open('train_samples.pkl', 'rb') as f:
        samples = pkl.load(f)
        # print(f'samples.shape -> {len(samples)} {[item.shape for item in samples]}')
    _ = view_samples(-1, samples, mode=hyps['md'])
    return G, D


def run_GAN_training(data_dir, epochs, bs, vs, lr=0.0002, z_size=128, sample_size=16,
                     conv_dim=64, beta1=0.5, beta2=0.999, mode='od', device='cuda:0'):
    '''
    Main function of GAN.

    Args:
        data_dir: location of training data
        epochs: number of epochs to train
        bs: batch_size
        vs: validation size
        lr: learning rate
        mode: pnf or od
        conv_dim: convolutional layer dimension
        device: where to train the model
    '''
    # Training parameters
    epochs = epochs
    learning_rate = lr
    batch_size = bs
    valid_size = vs

    # wrap essential info into dictionary:
    hyps = {
        'mn': 'GAN',
        'bs': batch_size,
        'vs': valid_size,
        'lr': learning_rate,
        'zs': z_size,
        'ss': sample_size,
        'cd': conv_dim,
        'md': mode
    }

    data_set = ConvEncoderDatasetRAM(data_dir)

    # split dataset for training and validation
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))  # hard coded to 0.8

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_loader = DataLoader(data_set, sampler=train_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    valid_loader = DataLoader(data_set, sampler=valid_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    # define discriminator and generator
    D = Discriminator(conv_dim, mode=hyps['md'])
    G = Generator(z_size=z_size, conv_dim=conv_dim, mode=hyps['md'])

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        D = nn.DataParallel(D).cuda()
        G = nn.DataParallel(G).cuda()
    elif torch.cuda.is_available():
        D = D.to(device)
        G = G.to(device)
    else:
        print('Training on CPU, very long training time is expectable.')

    # params
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999  # default value

    # Create optimizers for the discriminator and generator
    d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
    g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

    # Create optimizers for the discriminator and generator
    if TRAIN_ON_MULTI_GPUS:
        d_optimizer = optim.Adam(D.module.parameters(), lr, [beta1, beta2])
        g_optimizer = optim.Adam(G.module.parameters(), lr, [beta1, beta2])
    else:
        d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
        g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

    G, D = train_GAN(D, G, d_optimizer, g_optimizer, epochs, z_size,
                     train_loader, valid_loader, sample_size, hyps,
                     device=device, print_every=100)

    return G, D


if __name__ == '__main__':

    # dataset folders
    datasets = {
        "od1815": '2018/15min',
        "od1812": '2018/12min',
        "pnf1815": '2018_15min',
        "pnf1812": '2018_12min',
        "od1715": '2017/15min',
        "od1712": '2017/12min',
        "pnf1715": '2017_15min',
        "pnf1712": '2017_12min'
    }

    mode = 'pnf'
    data_dir = f'data/{datasets[mode + "1715"]}/tensors'

    # run_recursive_training()
    # run_classifier_training('ConvClassifier', data_dir, 50, 128, 0.8, 0.001, 2, device='cuda:1')
    run_encoder_training('ConvAutoEncoderShallow', data_dir, 1000, 1024, 0.8, 0.1,
                         mode=mode, hd=256, device='cuda:0')

    # run_GAN_training(data_dir, 100, 64, 0.8, z_size=100, conv_dim=256, mode='pnf')
