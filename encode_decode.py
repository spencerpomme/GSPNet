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


def regen(model, truths, dest, mode, device='cuda:0'):
    '''
    Generate a size of decoded state tensors (snapshots).

    Args:
        model: trained autoencoder, loaded from serialized file
        truths: a list of tensors
        dest: saving destination
        mode: `od` or `pnf`
        device: hardware
    Returns:
        recs: regenerated traffic state tensors
    '''
    for i, tensor in enumerate(truths):
        tensor = tensor.type(torch.float)
        if 'conv1' in dir(model):
            tensor = torch.unsqueeze(tensor, 0)
            tensor = tensor.permute(0,3,2,1)
        else:
            tensor = tensor.reshape((1, -1))
        tensor = tensor.to(device)
        deco = model(tensor)
        save_to(deco, dest, False, i, mode)  # false means it's not real future


def load(path: str, size: int):
    '''
    Load seq_len adjacent tensors from a random place.

    Args:
        path: path to folder holding tensors
        size: number of predictions to be generated
    Returns:
        states: a list of tensor of shape (69,69,1)
        truths: a list of tensor of shape (69,69,1)
    '''
    # tensor paths
    paths = glob(path + '/*.pkl')
    truths = []

    # select a random place to start draw the prime
    # np.random.seed(0)
    start = np.random.randint(0, len(paths)-size)
    print(f'Testing states id: {start} -> {start+size}')
    # load actual truths states, for test prediction accuracy visually
    for fp in paths[start: start+size]:
        truths_tensor = torch.load(fp)
        truths.append(truths_tensor)

    return truths


def save_to(tensor: torch.Tensor, dest: str, real: bool, id: int, mode: str):
    '''
    Save a tensor and its visualization image to specified destination.

    Args:
        tensor: predicted traffic state tensor, of shape (1, 69*69)
        dest: destination path of saving
        real: boolean value, indicating real future or predicted
        id: id of generated tensor/image
        mode: `pnf` or `od`
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
    if mode == 'od':
        tensor = tensor.reshape((69, 69, 1))
    if mode == 'pnf':
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
    tensor = tensor.squeeze()
    if mode == 'od':
        image = Image.fromarray(tensor, mode='L')
    elif mode == 'pnf':
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
    batch_size_pt = re.compile('(?<=bs)\d*')
    hidden_size_pt = re.compile('(?<=hd)\d*')
    lr_pt = re.compile('(?<=lr)0.\d*')
    mode_pt = re.compile('(?<=md)[A-Za-z]*')

    # create dictionary for information to reconstruct model
    hyps = {
        'mn': model_name_pt.findall(path)[0],
        'is': int(input_size_pt.findall(path)[0]),
        'os': int(output_size_pt.findall(path)[0]),
        'bs': int(batch_size_pt.findall(path)[0]),
        'hd': int(hidden_size_pt.findall(path)[0]),
        'lr': float(lr_pt.findall(path)[0]),
        'md': mode_pt.findall(path)[0]
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

    if hyps['mn'] == 'ConvAutoEncoder' or hyps['mn'] == 'ConvAutoEncoderShallow':
        model = models.__dict__[hyps['mn']](
            hyps['is'],
            hyps['os'],
            train_on_gpu=True,
            device=device,
            mode=hyps['md']
        )
    else:
        model = models.__dict__[hyps['mn']](
            hyps['is'],
            hyps['os'],
            hidden_dim=hyps['hd'],
            train_on_gpu=True,
            device=device
        )

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return model


def run(model_path, data_path, dest_path, size, mode, device):
    '''
    Main function of predict module.

    Args:
        model_path: path to the trained model
        data_path: path to where init states at
        dest_path: destination to save prediction
        size: size of predicted future states
        device: gpu or cpu
    '''
    print('Start testing autoencoder...')
    # extract model information from model_path string
    hyps = retrieve_hyps(model_path)

    # recontruct model
    model = reconstruct(model_path, hyps, device=device)

    truths = load(data_path, size)
    regen(model, truths, dest_path, mode, device=device)
    # save actual future
    for i, truth in enumerate(truths):
        save_to(truth, dest_path, True, i, mode)
    print('Finished!')


if __name__ == '__main__':

    model_path = 'trained_models/mnConvAutoEncoderShallow-is14283-os14283-bs128-lr0.001-hd32-mdpnf.pt'
    data_path = 'data/2018_15min/tensors'
    dest_path = 'autoencoder_test'
    run(model_path, data_path, dest_path, 8, 'pnf', 'cuda:0')
