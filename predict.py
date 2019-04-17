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

from PIL import Image
from torch import nn, optim
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from glob import iglob, glob
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm

from models import *

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


# TODO: finish this class
def predict(model, states, hidden):
    '''
    Generate future states using the trained neural network.

    Args:
        model:   PyTorch Module that holds the trained neural network
        states:  first N states to start predicting furture states,
                 each of them is of shape (1, 69*69), CUDA tensor.
        hidden:  hidden states
    Returns:
        The generated traffic states
    '''
    # reshape states to shape (batch_size, 1, flattened_dim)
    # here, batch_size == seq_len of intial states
    states = states.reshape(states.shape[0], 1, states.shape[1])
    # detach hidden from history
    hidden = tuple([each.data for each in hidden])
    out, hidden = model(states, hidden)
    # only want the last output
    return out[-1], hidden


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
    # move tensor to GPU
    states = states.cuda()

    seq_len = len(states)
    batch_size = seq_len

    if TRAIN_ON_MULTI_GPUS:
        hidden = model.module.init_hidden(batch_size)
    # initialize hidden state
    hidden = model.init_hidden(batch_size)

    # start sampling
    preds = []
    for i in range(size):

        pred, hidden = predict(model, states, hidden)
        preds.append(pred)
        save_to(pred, dest, False, i)  # false means it's not real future
        pred = pred.reshape((1, -1))   # reshape
        states = torch.cat((states[1:], pred))
    return preds


def load(prime_dir: str, size: int, seq_len: int):
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
    # tensor paths
    paths = glob(prime_dir + '/*.pkl')
    primes = []
    truths = []

    # select a random place to start draw the prime
    start = np.random.randint(0, len(paths)-seq_len)

    print(f'Prime states id: {start} -> {start+seq_len}')

    # load seq_len paths as the initial states, i.e. prime
    for sp in paths[start: start+seq_len]:
        prime_tensor = torch.load(sp).numpy()
        primes.append(prime_tensor)

    print(f'Predicting states id: {start+seq_len} -> {start+seq_len+size}')

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

    # print(f'tensor.shape -> {tensor.shape}')
    tensor = tensor.reshape((69, 69, 3))
    image_dest = dest + '/viz_images'
    tensor_dest = dest + '/tensors'

    # ensure the directories exist
    create_dir(image_dest)
    create_dir(tensor_dest)
    image_dest += f'/{"r" if real else "p"}-{id}.png'
    tensor_dest += f'/{"r" if real else "p"}-{id}.pkl'
    # save tensor
    torch.save(tensor, image_dest)
    # tensor to image
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()

    # simple normalize
    tensor *= (255 // tensor.max())
    tensor = tensor.astype('uint8')
    image = Image.fromarray(tensor)
    image = image.resize((345, 345))
    image.save(image_dest)


def retrieve_hyps(path: str):
    '''
    Retrieve hyperparameters that needed for reconstructing saved models from
    model name string.
    Args:
        path: path to serialize model state_dict file
    Returns:
        hyps: dictionary containing all necessary hyperparameters
    '''
    pass


def run(model_path, data_path, dest_path, size):
    '''
    Main function of predict module.
    Args:
        model_path: path to the trained model
        data_path: path to where init states at
        dest_path: destination to save prediction
        size: size of predicted future states
    '''
    print('Start predicting future states...')
    pattern = re.compile('(?<=sl)\d+(?=-)')
    seq_len = int(pattern.findall(model_path)[0])

    # load model
    device = torch.device('cuda')
    # the line below is temporal
    model = VanillaStateRNN(69*69*3, 69*69*3, 512,
                            n_layers=2, drop_prob=0.5)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    states, truths = load(data_path, size, seq_len)
    sample(model, states, size, dest_path)
    # save actual future
    for i, truth in enumerate(truths):
        save_to(truth, dest_path, True, i)
    print('Prediction finished!')


if __name__ == '__main__':

    model_path = 'trained_models/LSTM-sl25-bs512-lr0.001-nl2-dp0.5.pt'
    data_path = 'dataset/2018/15min/tensors'
    dest_path = 'future_states'
    run(model_path, data_path, dest_path, 14)
