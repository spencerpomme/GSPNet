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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

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
        model:   PyTorch Module that holds the trained neural network
        states:  first N states to start predicting furture states,
                 each of them is of shape (1, 69*69)
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


def sample(model, states, size, dest):
    '''
    Generate a size of future traffic state tensors (snapshots).
    Args:
        model: trained LSTM model, loaded from serialized file
        states: the initial states, shaped (1, 69*69)
        size: number of to be generated future states
        dest: saving destination
    Returns:
        preds: generated future traffic state tensors
    '''
    seq_len = len(states)
    batch_size = seq_len

    if TRAIN_ON_MULTI_GPUS:
        model = nn.DataParallel(model).cuda()
        hidden = model.module.init_hidden(batch_size)
    elif torch.cuda.is_available():
        model = model.cuda()
    # initialize hidden state
    hidden = model.init_hidden(batch_size)

    # start sampling
    preds = []
    for i in range(size):

        pred, hidden = predict(states, hidden)
        preds.append(pred)
        save_to(pred, dest, False, i)  # false means it's not real future
    return preds


def load(prime_dir: str, size: int, seq_len: int = None):
    '''
    Load seq_len adjacent tensors from a random place.
    Args:
        prime_dir: path to folder holding tensors
        size: number of predictions to be generated
        seq_len: sequence length of lstm
    Returns:
        states: tensor of initial states, a tensor of shape (seq_len, 69*69*3)
        truths: real future, a list of (69,69,3) numpy arrays
    '''
    if not seq_len:
        pattern = re.compile('(?<=sl)\d+(?=-)')
        seq_len = int(pattern.findall(path)[0])

    # tensor paths
    paths = glob(prime_dir + '/*.pkl')
    primes = []
    truths = []

    # select a random place to start draw the prime
    start = np.random.randint(0, len(primes)-seq_len)

    # load seq_len paths as the initial states, i.e. prime
    for sp in paths[start: start+seq_len]:
        prime_tensor = torch.load(sp).numpy()
        primes.append(prime_tensor)

    # load actual truths states, for test prediction accuracy visually
    for fp in paths[start+seq_len: start+seq_len+size]:
        truths_tensor = torch.load(fp)
        truths.append(truths_tensor)

    primes = np.array(primes).astype('float32')
    primes = primes.reshape((seq_len, -1))
    states = torch.from_numpy(primes)

    return states, truths


def save_to(tensor: torch.Tensor, dest: str, real: bool, id: int):
    '''
    Save a tensor and its visualization image to specified destination.
    Args:
        tensor: predicted traffic state tensor, of shape (1, 69*69)
        dest: destination path of saving
        real: boolean value, indicating real future or predicted
        id: id of generated tensor/image
    '''
    tensor = tensor.reshape((69, 69, 3))
    image_dest = dest + '/viz_images' + f'{"r" if real else "p"}-{id}.png'
    tensor_dest = dest + '/tensors' + f'{"r" if real else "p"}-{id}.pkl'

    # save tensor
    torch.save(tensor, image_dest)
    # tensor to image
    tensor = tensor.cpu()

    # simple normalize
    tensor *= (255 // tensor.max())
    tensor = tensor.astype('uint8')
    image = Image.fromarray(tensor)
    image.save(image_dest)


def run(model_path, data_path, dest_path, size):
    '''
    Main function of predict module.
    Args:
        model_path: path to the trained model
        data_path: path to where init states at
        dest_path: destination to save prediction
        size: size of predicted future states
    '''
    model = torch.load(model_path)
    states, truths = load(data_path, size)
    sample(model, states, size, dest_path)
    # save actual future
    for i, truth in enumerate(truths):
        save_to(truth, dest_path, True, i)


if __name__ == '__main__':

    print('Start predicting future states...')
    model_path = 'trained_models/LSTM-sl25-bs512-lr0.001-nl2-dp0.5.pt'
    data_path = 'tensor_dataset/full_15min_valid/tensors'
    dest_path = 'future_states'
