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

from PIL import Image
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from glob import glob, iglob

from losses import L1Penality


# models
# RNN models to do sequential prediction:
class VanillaLSTM(nn.Module):
    '''
    The baseline model.

    A simple LSTM model, without any preprocessing to the inputs.
    '''
    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=2,
                 drop_prob=0.5, train_on_gpu=True, device='cuda:0'):
        '''
        LSTM model initialization.

        Args:
            input_size:     dimention of state vector (flattened 3d tensor)
            output_size:    the same shape of input_size, a 3d tensor
                            with shape (69, 69, 3) to generate visual image
            hidden_dim:     hidden size of lstm layers
            n_layers:       number of lstm layers
            drop_prob:      drop out rate
            train_on_gpu:   whether use GPU or not
            device:         where to put the model
        '''
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.train_on_gpu = train_on_gpu
        self.dvc = device

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
            batch_size: divide the traffic state sequence into batch_size
                        equally long sub-sequences, for parallelization.
        Returns:
            hidden:     initialized hidden state
        '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().to(self.dvc),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().to(self.dvc))
        else:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_())

        return hidden


class VanillaGRU(nn.Module):
    '''
    The baseline model.

    A simple GRU model, without any preprocessing to the inputs.
    '''

    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=2,
                 drop_prob=0.5, train_on_gpu=True, device='cuda:0'):
        '''
        GRU model initialization.

        Args:
            input_size:     dimention of state vector (flattened 3d tensor)
            output_size:    the same shape of input_size, a 3d tensor
                            with shape (69, 69, 3) to generate visual image
            hidden_dim:     hidden size of gru layers
            n_layers:       number of gru layers
            drop_prob:      drop out rate
            train_on_gpu:   whether use GPU or not
            device:         where to put the model
        '''
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.train_on_gpu = train_on_gpu
        self.dvc = device

        # define the gru
        self.gru = nn.GRU(input_size, self.hidden_dim, n_layers,
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
        hidden = hidden.permute(1, 0, 2).contiguous()

        gru_out, hidden = self.gru(x, hidden)
        gru_out = gru_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(gru_out)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)
        # get the last output, because we decide the output traffic state is
        # caused by previous N (N >= 2) states.
        out = out[:, -1]

        # reshape hidden state, because using multiple GPUs
        hidden = hidden.permute(1, 0, 2).contiguous()

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
        # initialized to zero, for hidden state and cell state of gru
        weight = next(self.parameters()).data

        if self.train_on_gpu:
            hidden = weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().to(self.dvc)
        else:
            hidden = weight.new(batch_size, self.n_layers, self.hidden_dim).zero_()

        return hidden


# TODO: finish this model
class EmbedRNN(nn.Module):
    '''
    The baseline model.

    A simple LSTM model, without any preprocessing to the inputs.
    '''

    def __init__(self, input_size, output_size, hidden_dim=256, n_layers=2,
                 drop_prob=0.5, train_on_gpu=True, device='cuda:0'):
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
        self.dvc = device

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
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().to(self.dvc),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_().to(self.dvc))
        else:
            hidden = (weight.new(batch_size, self.n_layers, self.hidden_dim).zero_(),
                      weight.new(batch_size, self.n_layers, self.hidden_dim).zero_())

        return hidden


# TODO: finish this `yakkaina` model
class MultiDimLSTM(nn.Module):
    '''
    '''
    def __init__(self):
        '''
        '''
        pass


# classification model(s)
class ConvClassifier(nn.Module):
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
        super(ConvClassifier, self).__init__()
        self.is_conv = True
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


class MLPClassifier(nn.Module):
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
        super(MLPClassifier, self).__init__()
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


# GAN model:
# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    '''
    Creates a convolutional layer, with optional batch normalization.
    '''
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)

    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    # using Sequential container
    return nn.Sequential(*layers)


# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    '''
    Creates a transposed-convolutional layer, with optional batch normalization.
    '''
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


def scale(x, feature_range=(-1, 1)):
    '''
    Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.
    '''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range
    x = x * (max - min) + min
    return x


def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        # nn.init.constant_(m.weight.data, 0.45)


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples, mode='pnf'):
    if mode == 'pnf':
        chan = 3
    elif mode == 'od':
        chan = 1
    else:
        raise ValueError(f'Arg 3 `mode` expect string value pnf or od, but {mode} was provided.')

    fig, axes = plt.subplots(figsize=(16, 4), nrows=2, ncols=8, sharey=True,
                             sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img = img.reshape((69, 69, chan))

        # img = img.resize((345, 345, chan))
        # print(f'type of img: {type(img)} | shape of img: {img.shape}')
        if mode == 'pnf':
            im = ax.imshow(img)
        elif mode == 'od':
            img = Image.fromarray(img, mode='L')
            img = np.asarray(img)
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            raise ValueError(
                f'Arg 3 `mode` expect string value pnf or od, but {mode} was provided.')
    plt.show()


class Discriminator(nn.Module):
    '''
    Discriminator of GAN.
    '''

    def __init__(self, conv_dim=32, mode='pnf'):
        '''
        Args:
            conv_dim: conv layer base number
            mode: pnf or od
        '''
        super(Discriminator, self).__init__()
        self.mode = mode
        self.is_conv = True
        self.conv_dim = conv_dim
        if self.mode == 'pnf':
            self.chan = 3
        elif self.mode == 'od':
            self.chan = 1
        else:
            raise ValueError(f'Arg 3 `mode` expect string value pnf or od, but {mode} was provided.')

        # 69x69 input
        # first layer, no batch_norm
        self.conv1 = conv(self.chan, conv_dim, 5, padding=0, stride=4, batch_norm=False)
        # 17x17 out
        self.conv2 = conv(conv_dim, conv_dim*2, 3, padding=0)
        # 8x8 out
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, padding=0)
        # 3x3 out

        # final, fully-connected layer
        self.fc = nn.Linear(conv_dim*4*3*3, 1)

    def forward(self, x):
        # all hidden layers + leaky relu activation
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)

        # flatten
        out = out.view(-1, self.conv_dim*4*3*3)

        # final output layer
        out = self.fc(out)
        return out


class Generator(nn.Module):
    '''
    Generator of GAN
    '''

    def __init__(self, z_size, conv_dim=32, mode='pnf'):
        super(Generator, self).__init__()
        self.is_conv = True
        self.conv_dim = conv_dim
        self.mode = mode
        if self.mode == 'pnf':
            self.chan = 3
        elif self.mode == 'od':
            self.chan = 1
        else:
            raise ValueError(f'Arg 3 `mode` expect string value pnf or od, but {mode} was provided.')
        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim*4*3*3)

        # transpose conv layers
        self.t_conv1 = deconv(self.conv_dim*4, self.conv_dim*2, 4, padding=0)
        self.t_conv2 = deconv(self.conv_dim*2, self.conv_dim, 3, padding=0)
        self.t_conv3 = deconv(self.conv_dim, self.chan, 5, padding=0,
                              stride=4, batch_norm=False)

    def forward(self, x):
        # fully-connected + reshape
        out = self.fc(x)
        out = out.view(-1, self.conv_dim*4, 3, 3)  # (batch_size, depth, 4, 4)

        # hidden transpose conv layers + relu
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))

        # last layer + tanh activation
        out = self.t_conv3(out)
        out = torch.tanh(out)

        return out


class AutoEncoder(nn.Module):
    '''
    An autoencoder model, without any preprocessing to the inputs.
    '''

    def __init__(self, input_size, output_size, hidden_dim=512,
                 train_on_gpu=True, device='cuda:0'):
        '''
        Auto encoder initialization.

        Args:
            input_size:     dimention of state vector (flattened 3d tensor)
            output_size:    the same shape of input_size
            hidden_dim:     hidden size
            train_on_gpu:   whether use GPU or not
            device:         where to put the model
        '''
        super().__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.train_on_gpu = train_on_gpu
        self.dvc = device

        # define the layers
        self.encoder = nn.Linear(input_size, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.output_size)

    def forward(self, x):
        '''
        Pass tensor into the encoder and get it out from the decoder.

        Args:
            x:      input state vector (flattened)
        Returns:
            out:    output of current time step
        '''
        batch_size = x.size(0)
        x = F.relu(self.encoder(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        out = self.decoder(x)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)

        return out


class ConvAutoEncoderShallow(nn.Module):
    '''
    A CNN autoencoder model, without any preprocessing to the inputs.
    '''

    def __init__(self, output_size, mode):
        '''
        Auto encoder initialization.

        Args:
            input_size:     dimention of state vector
            output_size:    the same shape of input_size
            mode:           either `od`(greyscale) or `pnf`(rgb)
            train_on_gpu:   whether use GPU or not
            device:         where to put the model
        '''
        super(ConvAutoEncoderShallow, self).__init__()
        self.is_conv = True
        self.output_size = output_size

        # define encode and decode layers
        if mode == 'od':
            self.conv1 = nn.Conv2d(1, 16, 7, stride=2)
            self.conv_t1 = nn.ConvTranspose2d(
                16, 1, 2, stride=2, output_padding=1)
        elif mode == 'pnf':
            self.conv1 = nn.Conv2d(3, 16, 7, stride=2)
            self.conv_t1 = nn.ConvTranspose2d(
                16, 3, 2, stride=2, output_padding=1)

        self.conv2 = nn.Conv2d(16, 4, 4, stride=2, padding=1)
        self.conv_t2 = nn.ConvTranspose2d(4, 16, 4, stride=2)

    def forward(self, x):
        '''
        Pass tensor into the encoder and get it out from the decoder.

        Args:
            x:      input state vector (flattened)
        Returns:
            out:    output of current time step
        '''
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv_t2(x))
        out = self.conv_t1(x)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)

        return out


class ConvAutoEncoder(nn.Module):
    '''
    A CNN autoencoder model, without any preprocessing to the inputs.
    '''

    def __init__(self, output_size, mode):
        '''
        Auto encoder initialization.

        Args:
            output_size:    the same shape of input_size
            mode:           either `od`(greyscale) or `pnf`(rgb)
        '''
        super(ConvAutoEncoder, self).__init__()
        self.is_conv = True
        self.output_size = output_size

        # define encode and decode layers
        if mode == 'od':
            self.conv1 = nn.Conv2d(1, 16, 7, stride=2)             # od mode
            self.t_conv1 = nn.ConvTranspose2d(16, 1, 7, stride=2)  # pnf mode
        elif mode == 'pnf':
            self.conv1 = nn.Conv2d(3, 16, 7, stride=2)             # od mode
            self.t_conv1 = nn.ConvTranspose2d(16, 3, 7, stride=2)  # pnf mode

        self.conv2 = nn.Conv2d(16, 4, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4, 2, 4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(2, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 16, 2, stride=2)

    def forward(self, x):
        '''
        Pass tensor into the encoder and get it out from the decoder.
        '''
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv2(x))
        out = self.t_conv1(x)

        # reshape to be batch_size first
        out = out.view(batch_size, -1, self.output_size)

        return out


class VAE(nn.Module):
    '''
    Variational Auto Encoder
    '''

    def __init__(self, mode='pnf', z_dim=32):
        '''
        Initialization of VAE model.

        Args:
            mode: pnf or od
            z_dim: sample dimension
        '''
        if mode == 'pnf':
            self.img_chans = 3
        elif mode == 'od':
            self.img_chans = 1
        else:
            raise ValueError('Wrong mode. Only pnf and od are supported.')
        super(VAE, self).__init__()
        self.is_conv = True

        self.conv1 = nn.Conv2d(self.img_chans, 32,
                               kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=4, stride=2, padding=1)

        self.fc1 = nn.Linear(8*16*16, z_dim)
        self.fc2 = nn.Linear(8*16*16, z_dim)
        self.fc3 = nn.Linear(z_dim, 8*16*16)

        self.t_conv2 = nn.ConvTranspose2d(8, 32, kernel_size=2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(32, self.img_chans, kernel_size=7, stride=2)

    def forward(self, x):
        '''
        Forward pass
        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        h = x.view(x.size(0), -1)

        mu, logvar = self.fc1(h), self.fc2(h)
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        z = self.fc3(z)
        z = z.view(z.size(0), 8, 16, 16)
        z = F.relu(self.t_conv2(z))
        z = torch.sigmoid(self.t_conv1(z))

        return z, mu, logvar


class SparseConvAutoEncoder(nn.Module):
    '''
    Sparse autoencoder.
    '''

    def __init__(self, mode='pnf', hidden_dim=1024):
        '''
        Initialization of VAE model.
        '''
        if mode == 'pnf':
            self.img_chans = 3
        elif mode == 'od':
            self.img_chans = 1
        else:
            raise ValueError('Wrong mode. Only pnf and od are supported.')

        self.hidden_dim = hidden_dim
        self.is_conv = True

        super(SparseConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_chans, 16, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flattencv(),
            nn.Linear(32*16*16, self.hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 32*16*16),
            nn.ReLU(),
            UnFlattencv(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.img_chans, kernel_size=7, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Forward pass.
        '''
        x = self.encoder(x)
        x = L1Penality.apply(x, 0.5)
        x = self.decoder(x)
        return x


class Flattencv(nn.Module):
    def forward(self, input):
        # print(f'flatten input shape -> {input.shape}')
        return input.view(input.size(0), -1)


class UnFlattencv(nn.Module):
    def forward(self, input):
        # print(f'input.shape -> {input.shape}')
        return input.view(input.size(0), 32, 16, 16)


class SparseAutoEncoder(nn.Module):
    '''
    Sparse autoencoder.
    '''

    def __init__(self, mode='pnf', hidden_dim=1024):
        '''
        Initialization of VAE model.
        '''
        if mode == 'pnf':
            self.img_chans = 3
        elif mode == 'od':
            self.img_chans = 1
        else:
            raise ValueError('Wrong mode. Only pnf and od are supported.')

        self.hidden_dim = hidden_dim
        self.is_conv = True

        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(self.img_chans*69*69, self.hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.img_chans*69*69),
            UnFlatten()
        )

    def forward(self, x):
        '''
        Forward pass.
        '''
        x = self.encoder(x)
        x = L1Penality.apply(x, 0.1)
        x = self.decoder(x)
        return x


class Flatten(nn.Module):
    def forward(self, input):
        print(f'flatten input shape -> {input.shape}')
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        print(f'input.shape -> {input.shape}')
        return input.view(input.size(0), 1, 69, 69)


if __name__ == '__main__':

    print('In main of models.py')
