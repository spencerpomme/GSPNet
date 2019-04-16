import torch.nn.functional as F
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

Generate future trafic states.
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
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


# TODO: finish this class
def predict(model, states, hidden, dir):
    '''
    Generate future states using the trained neural network.

    Args:
        model:   The PyTorch Module that holds the trained neural network
        states:  The first N states to start predicting furture states,
                      each of them is of shape (3, 69, 69). A numpy ndarray.
        hidden:  hidden states
        dir:     A diretory to save generated tensor
    Returns:
        The generated traffic states
    '''
    states = states.cuda()

    gen_states = []
    hidden = tuple([each.data for each in hidden])

    out, hidden = model(states, hidden)

    return gen_states, hidden


def sample(path, prime_dir, seq_len=None, size=4):
    '''
    Sample from the trained prediction model.
    Args:
        path: path to the trained model file
        prime_dir: location containing test tensor data, containing
                   /tensor and /viz_images.
        size: number of snapshots to predict
    Returns:
        predictions: an array of predicted states
    '''
    # decide sequence length from model dir name
    if not seq_len:
        pattern = re.compile('(?<=sl)\d+(?=-)')
        seq_len = int(pattern.findall(path)[0])

    # load model
    model = torch.load(path)
    # tensor paths
    paths = glob(prime_dir + '/*.pkl')
    primes = []

    # load seq_len paths as the initial states, i.e. prime
    for p in paths[:seq_len]:
        tensor = torch.load(p).numpy()
        primes.append(tensor)
    primes = np.array(primes).astype('float32')
    primes = primes.reshape((seq_len, -1))
    states = torch.from_numpy(primes)

    model = model.cuda()
    model.eval()
    hidden = model.init_hidden(seq_len)

    # start sampling
    for i range(size):
        prediction, hidden = predict(model, states, hidden, '/predicted_states')
        torch.save(prediction, tensor_path)


def sample(model, init_states, length, dest):
    '''
    Generate a length of future traffic state tensors (snapshots).
    Args:
        model: trained LSTM model, loaded from serialized file
        init_states: the initial states
        length: number of to be generated future states
        dest: saving destination
    '''
    seq_len = len(init_states)
    batch_size = seq_len

    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
        hidden = model.module.init_hidden(batch_size)
    elif torch.cuda.is_available():
        model = model.cuda()
    # initialize hidden state
    hidden = model.init_hidden(batch_size)
    states = init_states

    # start sampling
    preds = []
    for i in range(length):

        pred, hidden = predict(states, hidden)
        preds.append(pred)
        save(pred, dest)


if __name__ == '__main__':
    print('main')
