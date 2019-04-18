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

Neural Network Models, Datasets defined here.
A part of GSPNet project.

'''

import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import re
import time
import torch
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from glob import glob, iglob


# models
# RNN models to do sequential prediction:
class VanillaStateLSTM(nn.Module):
    '''
    The baseline model.

    A simple LSTM model, without any preprocessing to the inputs.
    '''
    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=2,
                 drop_prob=0.5, train_on_gpu=True):
        '''
        LSTM model initialization.

        Args:
            input_size:     dimention of state vector (flattened 3d tensor)
            output_size:    the same shape of input_size, a 3d tensor
                            with shape (69, 69, 3) to generate visual image
            hidden_dim:     hidden size of lstm layers
            n_layers:       number of lstm layers
            drop_prob:      drop out rate
            lr:             learning rate
        '''
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.train_on_gpu = train_on_gpu

        # define the LSTM
        self.lstm = nn.LSTM(input_size, self.hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(self.drop_prob)

        # define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x, hidden):
        '''
        Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`.

        Args:
            x:      input state vector (flattened)
            hidden: hidden state of t-1
        Returns:
            out:    output of current time step
            hidden: hidden state of t
        '''
        batch_size = x.size(0)
        # The below three lines are for debug use, DO NOT remove.
        # print('*' * 20)
        # print(f'x shape: {x.shape}')
        # print(f'hidden[0] size is: {hidden[0].shape} | expected hidden[0] is {(2, x.size(0), 1024)}')

        # reshape hidden state, because using multiple GPUs
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])

        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get the last output, because we decide the output traffic state is
        # caused by previous N (N >= 2) states.
        out = out[:, -1]

        # reshape hidden state, because using multiple GPUs
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initializes hidden state.

        Args:
            batch_size: divide the traffic state sequence into batch_size equally long
                        sub-sequences, for parallelization.
        Returns:
            hidden:     initialized hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().cuda(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_())

        return hidden


# TODO: finish this model
class EmbedStateRNN(nn.Module):
    '''
    The baseline model.

    A simple LSTM model, without any preprocessing to the inputs.
    '''

    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=2,
                 drop_prob=0.5, train_on_gpu=True):
        '''
        LSTM model initialization.

        Args:
            input_size:     dimention of state vector (flattened 3d tensor)
            output_size:    the same shape of input_size, a 3d tensor
                            with shape (69, 69, 3) to generate visual image
            hidden_dim:     hidden size of lstm layers
            n_layers:       number of lstm layers
            drop_prob:      drop out rate
            lr:             learning rate
        '''
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.train_on_gpu = train_on_gpu

        # define the LSTM
        self.lstm = nn.LSTM(input_size, self.hidden_dim,
                            n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(self.drop_prob)

        # embedding layer
        self.embed = nn.Embedding()

        # define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x, hidden):
        '''
        Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`.

        Args:
            x:      input state vector (flattened)
            hidden: hidden state of t-1
        Returns:
            out:    output of current time step
            hidden: hidden state of t
        '''
        batch_size = x.size(0)
        # The below three lines are for debug use, DO NOT remove.
        # print('*' * 20)
        # print(f'x shape: {x.shape}')
        # print(f'hidden[0] size is: {hidden[0].shape} | expected hidden[0] is {(2, x.size(0), 1024)}')

        # reshape hidden state, because using multiple GPUs
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])

        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get the last output, because we decide the output traffic state is
        # caused by previous N (N >= 2) states.
        out = out[:, -1]

        # reshape hidden state, because using multiple GPUs
        hidden = tuple([h.permute(1, 0, 2).contiguous() for h in hidden])

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        '''
        Initializes hidden state.

        Args:
            batch_size: divide the traffic state sequence into batch_size
                        equally long sub-sequences, for parallelization.
        Returns:
            hidden:     initialized hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().cuda(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_())

        return hidden


# classification model(s)
class PeriodClassifier1(nn.Module):
    '''
    A Convolutional Neural Network based classifier.
    Determines whether a snapshot is temporally distinguishable by viz.
    '''
    def __init__(self, n_classes):
        '''
        Initialization.

        Args:
            n_classes: number of classes
        '''
        super(PeriodClassifier1, self).__init__()
        self.n_classes = n_classes  # (4 x 24) snapshots per day
        # define conv layers
        # in: (69 x 69) out: (33 x 33)
        self.conv1 = nn.Conv2d(3, 33, 5, 2)
        # in: (33 x 33) out: (16 x 16)
        self.conv2 = nn.Conv2d(33, 64, 3, 2)
        # in: (16 x 16) out: (7 x 7)
        self.conv3 = nn.Conv2d(64, 128, 4, 2)

        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

        # batch norm layers
        self.conv_bn1 = nn.BatchNorm2d(33)
        self.conv_bn2 = nn.BatchNorm2d(64)
        self.conv_bn3 = nn.BatchNorm2d(128)

        # fully connected layers
        # in (7 x 7 x 128) out (1024)
        self.fc1 = nn.Linear(7*7*128, 1024)
        self.fc2 = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        '''
        Forward behavior of the network.

        Args:
            x: input tensor
        Returns:
            y: prediction probability vector sized (n_classes, 1)
        '''
        # add sequence of convolutional layers
        x = F.relu(self.conv1(x))
        x = self.conv_bn1(x)
        x = F.relu(self.conv2(x))
        x = self.conv_bn2(x)
        x = F.relu(self.conv3(x))
        x = self.conv_bn3(x)

        # flatten tensor input
        x = x.view(-1, 128 * 7 * 7)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PeriodClassifier2(nn.Module):
    '''
    A Convolutional Neural Network based classifier.
    Determines whether a snapshot is temporally distinguishable by viz.
    '''

    def __init__(self, n_classes):
        '''
        Initialization.

        Args:
            n_classes: number of classes
        '''
        super(PeriodClassifier2, self).__init__()
        self.n_classes = n_classes  # (4 x 24) snapshots per day
        # define conv layers
        # in: (69 x 69) out: (33 x 33)
        self.conv1 = nn.Conv2d(3, 33, 5, 2)
        # in: (33 x 33) out: (16 x 16)
        self.conv2 = nn.Conv2d(33, 64, 3, 2)

        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

        # fully connected layers
        # in (7 x 7 x 128) out (1024)
        self.fc1 = nn.Linear(16*16*64, 1024)
        self.fc2 = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        '''
        Forward behavior of the network.

        Args:
            x: input tensor
        Returns:
            y: prediction probability vector sized (n_classes, 1)
        '''
        # add sequence of convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten tensor input
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PeriodClassifier3(nn.Module):
    '''
    A Convolutional Neural Network based classifier.
    Determines whether a snapshot is temporally distinguishable by viz.
    '''

    def __init__(self, n_classes):
        '''
        Initialization.

        Args:
            n_classes: number of classes
        '''
        super(PeriodClassifier3, self).__init__()
        self.n_classes = n_classes  # (4 x 24) snapshots per day

        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

        # fully connected layers
        # in (7 x 7 x 128) out (1024)
        self.fc1 = nn.Linear(69*69*3, 1024)
        self.fc2 = nn.Linear(1024, self.n_classes)

    def forward(self, x):
        '''
        Forward behavior of the network.

        Args:
            x: input tensor
        Returns:
            y: prediction probability vector sized (n_classes, 1)
        '''
        # flatten tensor input
        x = x.view(-1, 3 * 69 * 69)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# util functions
def decide_label(file: str):
    '''
    This function hard codes classification criteria to label the tensors.

    Args:
        file: a file name string
        nc: number of classes
    Returns:
        label: the class label of a given tensor file
    '''
    pattern = re.compile(
        '^(\d{4})-([0-1]\d)-([0-3]\d)_([0-1]\d|[2][0-3]);([0-5]\d);([0-5]\d)-(\d{4})-([0-1]\d)-([0-3]\d)_([0-1]\d|[2][0-3]);([0-5]\d);([0-5]\d)')

    file = file.split('\\')[1]
    i = int(pattern.findall(file)[0][3])
    # 3-hour-a-class
    labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label = labels[i]
    return label

if __name__ == '__main__':

    print('In main of models.py')
