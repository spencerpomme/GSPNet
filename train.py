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
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from glob import iglob, glob
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm

# import models
from models import *

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


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


class F2FDatasetRAM(data.Dataset):
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

    def __init__(self, datadir, seq_len):
        '''
        Initialization
        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
        '''
        self.paths = glob(datadir + '/*.pkl')
        path_num = len(self.paths)
        n_batches = path_num // seq_len
        # only want full size batches
        self.paths = self.paths[: n_batches * seq_len]

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

    def __init__(self, datadir, seq_len):
        '''
        Initialization
        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
            batch_size: divide a sequence to n_batch sequences
        '''
        self.paths = glob(datadir + '/*.pkl')
        path_num = len(self.paths)
        full_seq = path_num // seq_len * seq_len
        # only want full size batches
        self.paths = self.paths[: full_seq]

        self.seq_len = seq_len
        # total length of all tensors
        self.length = len(self.paths)

        # load all tensor into RAM
        tensors = []
        for path in tqdm(self.paths, total=self.length, ascii=True):
            tensor = torch.load(path).numpy()
            tensors.append(tensor)
        tensors = np.array(tensors).astype('float32')
        self.tensors = tensors.reshape((self.length, -1))

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        X = self.tensors[index: index+self.seq_len]
        y = self.tensors[index+self.seq_len]
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y


# Classification dataset
class SnapshotClassificationDataset(data.Dataset):
    '''
    A dataset that divide time snapshots into n classes, where n is the number
    of snapshots per day(default) or other specified number. For example, the
    several snapshots (15min) can be recognized as one class of hour X.
    '''
    def __init__(self, datadir: str):
        '''
        Initialization method.
        Args:
            datadir: directory containing `tensors` and `viz_images` folder
        '''
        self.datadir = datadir
        self.paths = glob(self.datadir + '/*.pkl')
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
        y = decide_label(path)

        return X, y


class SnapshotClassificationDatasetRAM(data.Dataset):
    '''
    The same dataset as SnapshotClassificationDataset, but load all data into
    memory.
    '''

    def __init__(self, datadir: str):
        '''
        Initialization method.
        Args:
            datadir: directory containing `tensors` and `viz_images` folder
        Explaination:
            The n_classes is actually fixed. If the time unit of tensor
            generation is 15min, then there would be 96 classes. The reason
            why n_classes is still provided as an argument is to double check
            the user knows (or, remembers) what he/she is doing.
        '''
        self.datadir = datadir

        # capture time unit (10min, 15min, etc) from dir string
        dir_pattern = re.compile('(?<=_)\d+(?=min)')
        interval = int(dir_pattern.findall(self.datadir)[0])

        # Patterns to extract key number from tensor path, which is used to
        # determine the class of that tensor. This is possible thanks to
        # naming rule of tensors.
        self.paths = glob(self.datadir + '/*.pkl')
        assert len(self.paths) != 0, 'glob error!'
        self.pattern = re.compile('(?<=-)\d+(?=.pkl)')

        # load tensor and labels into RAM
        self.Xs = []
        self.ys = []
        for path in tqdm(self.paths, total=len(self.paths), ascii=True):
            X = torch.load(path).type(torch.FloatTensor)
            X = X.permute(2, 1, 0)
            y = decide_label(path)
            self.Xs.append(X)
            self.ys.append(y)

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return len(self.paths)

    def __getitem__(self, index):
        '''
         Generates one sample of data
        '''
        X = self.Xs[index]
        y = self.ys[index]

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


# data feeder, type 2, deprecated
def batch_dataset(datadir, seq_len):
    '''
    Batch the neural network data using DataLoader
    Args:
        datadir: Directory storing tensor data
        seq_len: The sequence length of each batch
    Return:
        DataLoader with batched data
    '''
    # TODO: merger below commented lines into this function.
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

    # 'clip_grad_norm' helps prevent the exploding gradient problem in RNNs / LSTMs
    if TRAIN_ON_MULTI_GPUS:
        nn.utils.clip_grad_norm_(model.module.parameters(), clip)
    else:
        nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), h


# training function for sequential prediction
def train_lstm(model, batch_size, optimizer, criterion, n_epochs,
               train_loader, valid_loader, hyps, clip=5, stop_criterion=90,
               show_every_n_batches=1, multi_gpus=True):
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

        if TRAIN_ON_MULTI_GPUS:
            hidden = model.module.init_hidden(batch_size)
        else:
            hidden = model.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # early stop mechanism:
            if early_stop_count >= stop_criterion:
                print(
                    f'Validation loss stops decresing for {stop_criterion} epochs, early stop triggered.')
                break

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
                if TRAIN_ON_MULTI_GPUS:
                    val_h = model.module.init_hidden(batch_size)
                else:
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
                print(
                    f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.4f} | Val Loss {avg_val_loss:.4f} | Min Val {valid_loss_min:.4f}',
                    flush=True)

                # decide whether to save model or not:
                if avg_val_loss < valid_loss_min:

                    print(f'Valid Loss {valid_loss_min:.4f} -> {avg_val_loss:.4f}. Saving...', flush=True)
                    # saving state_dict of model
                    torch.save(model.state_dict(),
                               f'trained_models/LSTM-sl{hyps["sl"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-nl{hyps["nl"]}-dp{hyps["dp"]}.pt')
                    valid_loss_min = avg_val_loss
                    early_stop_count = 0

                else:
                    early_stop_count += 1

                train_losses = []
                valid_losses = []

    # returns a trained model
    end = time.time()
    print(f'Training ended at {time.ctime()}, took {end-start:2f} seconds.')
    return model, (tl, vl)


# training function of CNN classification
def train_classifier(model, optimizer, criterion, n_epochs,
                     train_loader, valid_loader, hyps,
                     stop_criterion=20,
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
    valid_losses = []

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
            f'Epoch: {epoch_i:>4}/{n_epochs:<4} | Loss: {avg_tra_loss:.4f} | Val Loss {avg_val_loss:.4f} | Min Val {valid_loss_min:.4f}',
            flush=True)

        # decide whether to save model or not:
        if avg_val_loss < valid_loss_min:

            print(f'Valid Loss {valid_loss_min:.4f} -> {avg_val_loss:.4f}. Saving...', flush=True)
            torch.save(model.state_dict(),
                        f'trained_models/Classifier-bs{hyps["bs"]}-lr{hyps["lr"]}-nc{hyps["nc"]}-dp{hyps["dp"]}.pt')

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


# run functions of this module
def run_lstm_training(epochs, sl=12, bs=64, lr=0.001, hd=256, nl=2, dp=0.5):
    '''
    Main function of lstm training.
    Args:
        epochs: number of epochs to train
        sl: sequence_length,
        bs: batch_size,
        lr: learning_rate,
        hd: hidden_dim,
        nl: n_layers,
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
        'sl': sequence_length,
        'bs': batch_size,
        'lr': learning_rate,
        'hd': hidden_dim,
        'nl': n_layers,
        'dp': drop_prob
    }

    # Initialize data loaders
    train_dir = 'tensor_dataset/full_15min_train/tensors'
    valid_dir = 'tensor_dataset/full_15min_valid/tensors'

    # LSTM data loader
    train_set = S2FDatasetRAM(train_dir, sequence_length)
    valid_set = S2FDatasetRAM(valid_dir, sequence_length)
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size,
                              num_workers=0, drop_last=True)

    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=batch_size,
                              num_workers=0, drop_last=True)

    # initialize model
    model = VanillaStateRNN(input_size, output_size, hidden_dim,
                            n_layers=n_layers, drop_prob=drop_prob)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        print('Training on CPU, very long training time is expectable.')

    # optimizer and criterion(loss function)
    if TRAIN_ON_MULTI_GPUS:
        optimizer = optim.Adam(model.module.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # start training
    trained_model, tlvl = train_lstm(model, batch_size, optimizer, criterion,
                                     epochs, train_loader, valid_loader, hyps)

    # loss plot
    tl, vl = tlvl
    x = np.arange(len(tl))

    train_curve, = plt.plot(x, tl, 'r-', label='train loss')
    valid_curve, = plt.plot(x, vl, 'b-', label='valid loss')
    plt.legend(handler_map={train_curve: HandlerLine2D(numpoints=1)})

    plt.savefig(
        'trained_models' +
        f'/LSTM-sl{hyps["sl"]}-bs{hyps["bs"]}-lr{hyps["lr"]}-nl{hyps["nl"]}-dp{hyps["dp"]}.png'
    )
    plt.show()


def run_classifier_training(epochs, nc, vs, rs, lr=0.001, bs=128, dp=0.5):
    '''
    Main function of cnn classifier training.
    Args:
        epochs: number of epochs to train
        nc: n classes
        vs: valida size, proportion of validation data set
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
        'bs': batch_size,
        'lr': learning_rate,
        'nc': nc,
        'dp': drop_prob
    }

    # Initialize data loaders
    data_dir = 'tensor_dataset/full_15min/tensors'

    # LSTM data loader
    data_set = SnapshotClassificationDatasetRAM(data_dir)

    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(np.floor(vs * num_train))

    # shuffle
    np.random.seed(rs)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(data_set, sampler=train_sampler, batch_size=batch_size,
                              num_workers=0, drop_last=True)

    valid_loader = DataLoader(data_set, sampler=valid_sampler, batch_size=batch_size,
                              num_workers=0, drop_last=True)

    # initialize model
    model = PeriodClassifier3(n_classes=nc)

    # model training device
    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()
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
        f'/Classifier-bs{hyps["bs"]}-lr{hyps["lr"]}-nc{hyps["nc"]}-dp{hyps["dp"]}.png'
    )
    plt.show()


if __name__ == '__main__':

    run_lstm_training(20, sl=12, bs=256, lr=0.001, hd=512, nl=3, dp=0.5)
    # run_classifier_training(100, 2, 0.1, 0, lr=0.001, bs=1024, dp=0.1)
