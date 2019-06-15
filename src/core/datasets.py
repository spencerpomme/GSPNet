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

Customized datset classes defined here.
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

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


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


# Customized RNN/LSTM datasets when dataset are to big to load at once into RAM
# Data feeders, type 1 (using classes)
class F2FDataset(data.Dataset):
    '''
    Frame to frame dataset.
    '''

    def __init__(self, datadir, seq_len):
        '''
        Initialization.

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

        self.input_ids = list(self.idict.keys())[:-1]
        self.label_ids = list(self.idict.keys())[1:]

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
        Initialization.

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

        self.input_ids = list(self.idict.keys())[:-1]
        self.label_ids = list(self.idict.keys())[1:]

        # load all tensor into RAM
        tensors = []
        for path in tqdm(self.paths, total=self.length, ascii=True):
            tensor = torch.load(path).numpy()
            tensors.append(tensor)
        # pad one at the end of the sequence with first state
        pad_tensor = torch.load(self.paths[0]).numpy()
        tensors.append(pad_tensor)

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
        # Load data and get label
        X = self.tensors[index]
        y = self.tensors[index+1]
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y


class S2FDataset(data.Dataset):
    '''
    Sequence of Frames to one frame dataset.
    '''

    def __init__(self, datadir, seq_len):
        '''
        Initialization.

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
        Denotes the total number of samples.
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data.
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

        return X, y


class S2FDatasetRAM(data.Dataset):
    '''
    Sequence of Frames to one frame dataset.
    Load all data into RAM at once.
    '''

    def __init__(self, datadir, seq_len):
        '''
        Initialization.

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
        Denotes the total number of samples.
        '''
        return self.length - self.seq_len

    def __getitem__(self, index):
        '''
        Generates one sample of data.
        '''
        X = self.tensors[index: index+self.seq_len]
        y = self.tensors[index+self.seq_len]
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y


# Classification dataset
# TODO: Please clean unused variables
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
        Denotes the total number of samples.
        '''
        return len(self.paths)

    def __getitem__(self, index):
        '''
         Generates one sample of data.
         Decide class of the sample on the fly.
        '''
        path = self.paths[index]
        X = torch.load(path)
        y = decide_label(path)

        return X, y


# TODO: Please clean unused variables
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
        Denotes the total number of samples.
        '''
        return len(self.paths)

    def __getitem__(self, index):
        '''
         Generates one sample of data.
        '''
        X = self.Xs[index]
        y = self.ys[index]

        return X, y


class EncoderDatasetRAM(data.Dataset):
    '''
    Auto encoder dataset.
    '''

    def __init__(self, datadir):
        '''
        Initialization.

        Args:
            datadir: directory of serialized tensors
            seq_len: timestep length of tensors
        '''
        self.paths = glob(datadir + '/*.pkl')
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
        return self.length - 1

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Load data and get label
        tensor = self.tensors[index]
        X = torch.from_numpy(tensor)
        y = torch.from_numpy(tensor)
        return X, y


class ConvEncoderDatasetRAM(data.Dataset):
    '''
    Convolutional Auto encoder dataset.
    '''

    def __init__(self, datadir):
        '''
        Initialization.

        Args:
            datadir: directory of serialized tensors
        '''
        self.paths = glob(datadir + '/*.pkl')
        self.length = len(self.paths)

        # load all tensor into RAM
        tensors = []
        for path in tqdm(self.paths, total=self.length, ascii=True):
            tensor = torch.load(path).numpy()
            tensors.append(tensor)

        tensors = np.array(tensors).astype('float32')
        self.tensors = tensors
        # numpy image to pytorch image need to swap axes
        self.tensors = np.swapaxes(self.tensors, 1, 3)
        print(f'Conv Autoencoder data shape: {tensors.shape}')

    def __len__(self):
        '''
        Denotes the total number of samples
        '''
        return self.length - 1

    def __getitem__(self, index):
        '''
        Generates one sample of data
        '''
        # Load data and get label
        X = self.tensors[index]
        X = torch.from_numpy(X)
        y = self.tensors[index]
        # y = y.reshape(1, -1)
        y = torch.from_numpy(y)
        # print(f'X.shape -> {X.shape} || y.shape -> {y.shape}')
        return X, y


if __name__ == '__main__':

    print('In module datasets.py')
