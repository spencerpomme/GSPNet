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

Training methods and customized datset classes defined here.
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
from glob import iglob, glob
from matplotlib import pyplot as plt
from tqdm import tqdm

# import models
from models import *


# Customized RNN/LSTM datasets when dataset are to big to load at once into RAM
# Data feeders, type 1 (using classes)
class F2FDataset(data.Dataset):
    '''
    Frame to frame dataset.
    '''

    def __init__(self, datadir, seq_len):
        '''
        Initialization
        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
        '''
        self.paths = glob(datadir + '/*.pkl')
        # only want full seq_len sized length numbers
        self.seq_len = seq_len
        self.length = len(self.paths) // seq_len * seq_len
        self.idict = {}
        for i in range(self.length):
            self.idict[i] = self.paths[i]

        self.input_ids = self.idict.keys()[:-1]
        self.label_ids = self.idict.keys()[1:]

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Load data and get label
        X = torch.load(self.idict[self.input_ids[index]])
        y = torch.load(self.idict[self.label_ids[index]])

        return X, y


class S2FDataset(data.Dataset):
    '''
    Sequence of Frames to one frame dataset.
    '''

    def __init__(self, datadir, seq_len, batch_size):
        '''
        Initialization
        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
            batch_size: divide a sequence to n_batch sequences
        '''
        self.paths = glob(datadir + '/*.pkl')
        path_num = len(self.paths)
        n_batches = path_num // batch_size
        # only want full size batches
        self.paths = self.paths[: n_batches * batch_size]

        self.seq_len = seq_len
        self.length = len(self.paths) // seq_len * seq_len
        self.idict = {}
        for i in range(self.length):
            self.idict[i] = self.paths[i]

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Load data and get label
        X = []
        for i in range(self.seq_len):
            x = torch.load(self.idict[index + i]).numpy()
            X.append(x)
        X = np.array(X).astype('float32')
        X = X.reshape((len(X), -1))
        X = torch.from_numpy(X)
        y = torch.load(self.idict[index + self.seq_len]).type(torch.float32)

        # flatten y to be the same dimention as X
        y = y.flatten()
        # print('#'*20)
        # print(f'X shape is: {X.shape}')  # <- X is ok
        # print('#' * 20)

        return X, y


class S2FDatasetRAM(data.Dataset):
    '''
    Sequence of Frames to one frame dataset.
    Load all data into RAM at once.
    '''

    def __init__(self, datadir, seq_len, batch_size):
        '''
        Initialization
        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
            batch_size: divide a sequence to n_batch sequences
        '''
        self.paths = glob(datadir + '/*.pkl')
        path_num = len(self.paths)
        n_batches = path_num // batch_size
        # only want full size batches
        self.paths = self.paths[: n_batches * batch_size]

        self.seq_len = seq_len
        self.length = len(self.paths) // seq_len * seq_len
        self.idict = {}
        for i in range(self.length):
            self.idict[i] = self.paths[i]

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Load data and get label
        X = []
        for i in range(self.seq_len):
            x = torch.load(self.idict[index + i]).numpy()
            X.append(x)
        X = np.array(X).astype('float32')
        X = X.reshape((len(X), -1))
        X = torch.from_numpy(X)
        y = torch.load(self.idict[index + self.seq_len]).type(torch.float32)

        # flatten y to be the same dimention as X
        y = y.flatten()
        # print('#'*20)
        # print(f'X shape is: {X.shape}')  # <- X is ok
        # print('#' * 20)

        return X, y


# Classification dataset
class SnapshotClassificationDataset(data.Dataset):
    '''
    A dataset that divide time snapshots into n classes, where n is the number
    of snapshots per day(default) or other specified number. For example, the
    several snapshots (15min) can be recognized as one class of hour X.
    '''
    def __init__(self, datadir: str, combine_fact: int=1):
        '''
        Initialization method.
        Args:
            datadir: directory containing `tensors` and `viz_images` folder
            combine_fact: combine factor, number of adjacent snapshots to be
                          classified as same class.
        Example:
            if combine_fact == 2, then:
            snap1, snap2, ..., snapN of one day is classified to N/2 classes,
            (snap1, snap2) is of one class, (snap3, snap4) is of the second
            class, etc. Note: N % combine_fact should be 0!
        '''
        self.datadir = datadir
        self.combine_fact = combine_fact

        # capture time unit from dir string
        dir_pattern = re.compile('(?<=_)\d+(?=min)')
        interval = int(dir_pattern.findall(self.datadir)[0])

        # number of snapshots in a day: raw_clss
        raw_clss = 60 / interval * 24
        assert raw_clss == int(raw_clss), 'raw_clss should be an whole number'
        assert raw_clss % self.combine_fact == 0, 'raw_clss not divisible'

        # actual classes in this setting
        self.n_classes = raw_clss / self.combine_fact
        self.paths = glob(self.datadir + '/tensors/*.pkl')
        self.pattern = re.compile('(?<=-)\d+(?=.pkl)')

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return len(self.paths)

    def __getitem__(self, index):
        '''
         Generates one sample of data
         Decide class of the sample on the fly.
        '''
        path = self.paths[index]
        X = torch.load(path)
        y = int(self.pattern.findall(path)[0]) % self.n_classes

        return X, y


# helper functions
def save_model(filename: str, model):
    '''
    Save model to local file.

    Args:
        filename: file name string
        model: trained model
    '''
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(model, save_filename)


def load_model(filename: str):
    '''
    Load trained model.

    Args:
        filename: file name string

    Returns:
        loaded torch model
    '''
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)


# data feeder, type 2
def batch_dataset(states, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader

    Args:
        states:
        sequence_length: The sequence length of each batch
        batch_size: The size of each batch; the number of sequences in a batch

    Return:
        DataLoader with batched data
    """
    num_batches = len(states) // batch_size

    # only full batches
    states = states[: num_batches * batch_size]

    # TODO: Implement function
    features, targets = [], []

    for idx in range(0, (len(states) - sequence_length)):
        features.append(states[idx: idx + sequence_length])
        targets.append(states[idx + sequence_length])

    data = TensorDataset(torch.from_numpy(np.array(features)),
                         torch.from_numpy(np.array(targets)))

    data_loader = torch.utils.data.DataLoader(
        data, shuffle=False, batch_size=batch_size, num_workers=0)

    # return a dataloader
    return data_loader


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


# training function for sequential prediction
def train_lstm(model, batch_size, optimizer, criterion, n_epochs,
               train_loader, valid_loader, hyps,
               clip=5, stop_criterion=20,
               show_every_n_batches=1):
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

    Returns:
        A trained model. The best model will also be saved locally.
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

    # for plot training loss and validation loss
    tl = []
    vl = []

    model.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        hidden = model.init_hidden(batch_size)

        # early stop mechanism:
        if early_stop_count >= stop_criterion:
            print(
                f'Validation loss stops decresing for {stop_criterion} epochs, early stop triggered.')
            break

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if batch_i > n_batches:
                break

            # forward, back prop
            # print(f'inputs shape: {inputs.shape} labels shape: {labels.shape}')
            # print(f'inputs dtype: {inputs[0][0][0].dtype} label shape: {labels[0][0].dtype}')
            inputs, labels = inputs.cuda(), labels.cuda()

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
                avg_tra_loss = np.mean(train_losses)

                tl.append(avg_tra_loss)
                vl.append(avg_val_loss)
                # printing loss stats
                print(f'Epoch: {epoch_i:>4}/{n_epochs:<4}  Loss: {avg_tra_loss:4f}  Val Loss {avg_val_loss:4f}')

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:
                    print(f'Valid Loss {valid_loss_min:4f} -> {avg_val_loss:4f}. Saving...')
                    valid_loss_min = avg_val_loss
                    early_stop_count = 0

                else:
                    early_stop_count += 1
                    torch.save(model.state_dict(),
                               f'trained_models/LSTM-sl{hyps["sl"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-nl{hyps["nl"]}-dp{hyps["dp"]}.pt')

                train_losses = []
                valid_losses = []

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model, (tl, vl)


# training function of CNN classification
# TODO: this function is not done.
def train_classifier(model, batch_size, optimizer, criterion, n_epochs,
                     train_loader, valid_loader, hyps,
                     stop_criterion=20,
                     show_every_n_batches=1000):
    '''
    Train a CNN classifier with the given hyperparameters.

    Args:
        model:              The PyTorch Module that holds the neural network
        batch_size:         batch size, integer
        optimizer:          The PyTorch optimizer for the neural network
        criterion:          The PyTorch loss function
        n_epochs:           Total go through of entire dataset
        train_loader:       Training data loader
        valid_loader:       Validation data loader
        chyps:               A dict containing hyperparameters
        show_every_batches: Display loss every this number of time steps

    Returns:
        A trained model. The best model will also be saved locally.
    '''
    # clear cache
    torch.cuda.empty_cache()
    # start timing
    start = time.time()
    print(f'Training classifier started at {time.ctime()}')
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

        # early stop mechanism:
        if early_stop_count >= stop_criterion:
            print(
                f'Validation loss stops decresing for {stop_criterion} epochs, early stop triggered.')
            break

        for data, label in train_loader:

            # forward, back prop
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # record loss
            train_losses.append(loss.item() * data.size(0))

            # print loss every show_every_n_batches batches including validation loss
            if batch_i % show_every_n_batches == 0:
                # get validation loss
                valid_losses = []

                # switch to validation mode
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
                    f'Epoch: {epoch_i:>4}/{n_epochs:<4}  Loss: {avg_tra_loss}  Val Loss {avg_val_loss}')

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:
                    print(
                        f'Valid Loss {valid_loss_min:4f} -> {avg_val_loss:4f}. Saving...')
                    valid_loss_min = avg_val_loss
                    early_stop_count = 0

                else:
                    early_stop_count += 1
                    torch.save(model.state_dict(),
                               f'trained_models/CNN-lr{hyps["lr"]}.pt')

                train_losses = []
                valid_losses = []

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model, (tl, vl)


if __name__ == '__main__':

    # LSTM Model Data params
    sequence_length = 12  # number of time slices in a sequence
    batch_size = 16       # how many sequences processed at a time
    clip = 5

    # Training parameters
    epochs = 20
    learning_rate = 0.001
    # (use `sample_per_batch` to distinguish from batch_size in RNN context)
    sample_per_batch = 128

    # Model parameters
    input_size = 69*69*3
    output_size = input_size
    hidden_dim = 1024
    # Number of RNN Layers
    n_layers = 2
    drop_prob = 0.4
    # Show stats for every n number of batches
    senb = 1

    # wrap essential info into dictionary:
    hyps = {
        'sl': sequence_length,
        'bs': batch_size,
        'lr': learning_rate,
        'hd': hidden_dim,
        'nl': n_layers,
        'dp': drop_prob
    }

    # Environment parameter
    train_on_gpu = torch.cuda.is_available()

    # Initialize data loaders
    train_dir = 'tensor_dataset/full_year_2018_15min/tensors'
    valid_dir = 'tensor_dataset/nn_test_15min_val/tensors'

    # train_iter = iglob(train_dir + '/*')
    # valid_iter = iglob(valid_dir + '/*')

    # train_states = []
    # valid_states = []

    # print('Loading dataset...')
    # print('Loading training set...')
    # for state in tqdm(train_iter, ascii=True):
    #     state = torch.load(state).numpy()
    #     train_states.append(state)
    # print('Loading validation set...')
    # for state in tqdm(valid_iter, ascii=True):
    #     state = torch.load(state).numpy()
    #     valid_states.append(state)

    # train_states = np.array(train_states)
    # valid_states = np.array(valid_states)

    # train_states = train_states.reshape((len(train_states), -1))
    # valid_states = valid_states.reshape((len(valid_states), -1))
    # train_states = train_states.astype('float32')
    # valid_states = valid_states.astype('float32')

    # train_loader = batch_dataset(train_states, sequence_length, batch_size)
    # valid_loader = batch_dataset(valid_states, sequence_length, batch_size)
    # print('Dataset Loaded.')

    # LSTM data loader
    train_set = S2FDataset(train_dir, sequence_length, batch_size)
    valid_set = S2FDataset(valid_dir, sequence_length, batch_size)
    train_loader = DataLoader(train_set, shuffle=False,
                              batch_size=sample_per_batch, num_workers=0)

    valid_loader = DataLoader(valid_set, shuffle=False,
                              batch_size=sample_per_batch, num_workers=0)

    # initialize model
    model = VanillaStateRNN(input_size, output_size, hidden_dim,
                            n_layers=n_layers, drop_prob=drop_prob)

    if train_on_gpu:
        model = model.cuda()

    # optimizer and criterion(loss function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # start training
    trained_model, tlvl = train_lstm(model, sample_per_batch, optimizer, criterion,
                                     epochs, train_loader, valid_loader, hyps)

    # loss plot
    tl, vl = tlvl
    x = np.arange(len(tl))

    plt.plot(x, tl, 'r-')
    plt.plot(x, vl, 'b-')
    plt.show()
