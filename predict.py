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
import torch.nn.functional as F

from PIL import Image
from torch import nn, optim
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from glob import iglob, glob
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm

import models
from models import *
from datasets import *

# Environment global variable
TRAIN_ON_MULTI_GPUS = False  # (torch.cuda.device_count() >= 2)


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
    if type(hidden) == tuple:
        hidden = tuple([each.data for each in hidden])
    else:
        hidden = hidden.data
    out, hidden = model(states, hidden)
    # only want the last output
    return out[-1], hidden


def sample(model, states, size, dest, device='cuda:0'):
    '''
    Generate a size of future traffic state tensors (snapshots).

    Args:
        model: trained LSTM model, loaded from serialized file
        states: the initial states, shaped (1, 69*69)
        size: number of to be generated future states
        dest: saving destination
        device: hardware
    Returns:
        preds: generated future traffic state tensors
    '''
    # move tensor to GPU
    states = states.to(device)

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
    # np.random.seed(0)
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
    torch.save(tensor, tensor_dest)
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
    Retrieve hyperparameters that needed for reconstructing saved LSTM models
    from model name string.

    Args:
        path: path to serialize model state_dict file
    Returns:
        info: dictionary containing all necessary information
    '''
    path = path.split('/')[1]

    # hyperparameter extraction patterns
    model_name_pt = re.compile('(?<=mn)[A-Za-z]*(?=-)')
    input_size_pt = re.compile('(?<=is)\d*')
    output_size_pt = re.compile('(?<=os)\d*')
    seq_len_pt = re.compile('(?<=sl)\d*')
    batch_size_pt = re.compile('(?<=bs)\d*')
    n_layers_pt = re.compile('(?<=nl)\d')
    hidden_size_pt = re.compile('(?<=hd)\d*')
    lr_pt = re.compile('(?<=lr)0.\d*')
    drop_prob_pt = re.compile('(?<=dp)0.\d*')

    # create dictionary for information to reconstruct model
    hyps = {
        'mn': model_name_pt.findall(path)[0],
        'is': int(input_size_pt.findall(path)[0]),
        'os': int(output_size_pt.findall(path)[0]),
        'sl': int(seq_len_pt.findall(path)[0]),
        'bs': int(batch_size_pt.findall(path)[0]),
        'nl': int(n_layers_pt.findall(path)[0]),
        'hd': int(hidden_size_pt.findall(path)[0]),
        'lr': float(lr_pt.findall(path)[0]),
        'dp': float(drop_prob_pt.findall(path)[0])
    }

    return hyps


def reconstruct(model_path: str, hyps: dict, device='cuda:0'):
    '''
    Reconstruct the trained model from serialized .pt file.

    Args:
        model_path: actual place saving the model
        hyps: dictionary of infos needed to reconstruct model
    Returns:
        model: a reconstructed LSTM model
    '''
    # load model, decide device
    if torch.cuda.is_available():
        device = torch.device(device)
    else:
        device = torch.device('cpu')

    model = models.__dict__[hyps['mn']](
        hyps['is'],
        hyps['os'],
        hidden_dim=hyps['hd'],
        n_layers=hyps['nl'],
        drop_prob=hyps['dp'],
        train_on_gpu=True,
        device=device
    )

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def run(model_path, data_path, dest_path, size, device):
    '''
    Main function of predict module.

    Args:
        model_path: path to the trained model
        data_path: path to where init states at
        dest_path: destination to save prediction
        size: size of predicted future states
        device: gpu or cpu
    '''
    print('Start predicting future states...')
    # extract model information from model_path string
    hyps = retrieve_hyps(model_path)

    # recontruct model
    model = reconstruct(model_path, hyps, device=device)
    seq_len = hyps['sl']

    states, truths = load(data_path, size, seq_len)
    sample(model, states, size, dest_path, device)
    # save actual future
    for i, truth in enumerate(truths):
        save_to(truth, dest_path, True, i)
    print('Prediction finished!')


if __name__ == '__main__':

    model_path = 'trained_models/mnVanillaLSTM-os14283-is14283-hd256-nl2-dp0.5-sl4-bs64-lr0.01.pt'
    data_path = 'data/2018_15min/tensors'
    dest_path = 'future_states'
    run(model_path, data_path, dest_path, 14, 'cuda:1')
