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
from torchvision import datasets, transforms

# import models, loss functions and datasets
import models
from models import *
from losses import *
from datasets import *
from train import *


def shrink_large_img(img_path):
    bad_file_list = []
    with Image.open(img_path) as img:
        if min(img.size) > 512:
            try:
                print(f'File {img_path} shrinked!')
                img.thumbnail((275, 275))
                img.save(img_path, 'JPEG')
            except OSError as e:
                print(f'Bad image: {img_path}')
                bad_file_list.append(img_path)
    for file in bad_file_list:
        os.remove(file)
        print(f'{file} is bad and removed!')


if __name__ == '__main__':
    # Hyperparameters:
    chan = 3
    batch_size = 256
    num_workers = 0
    learning_rate = 0.1
    epochs = 100
    mode = 'pnf'
    save_dir = 'trained_models'
    dest = 'autoencoder_test/sanitycheck'

    hyps = {
        'is': 69*69*chan,
        'os': 69*69*chan,
        'mn': 'sanitycheck',
        'hd': 121,
        'bs': batch_size,
        'lr': learning_rate,
        'md': mode
    }

    # training data transforms
    t = transforms.Compose([transforms.Resize(81), transforms.CenterCrop(69),
                            transforms.ToTensor()
                            # transforms.Normalize([0.485, 0.456, 0.406],
                            #                      [0.229, 0.224, 0.225])
                            ])

    # data = datasets.MNIST(
    #     'data/sanity_check/MNIST', train=True, transform=t, download=True)

    data = datasets.CIFAR10(
        'data/sanity_check/cifar10', train=True, transform=t, download=True)
    # split dataset for training and validation
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(0.8 * num_train))  # hard coded to 0.8

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SequentialSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)

    train_loader = DataLoader(data, sampler=train_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    valid_loader = DataLoader(data, sampler=valid_sampler,
                              batch_size=batch_size, num_workers=0, drop_last=True)

    model = ConvAutoEncoder(hyps['os'], mode=mode)
    model = model.to('cuda:0')

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # trained_model = train_encoder(model, optimizer, criterion, epochs,
    #                               train_loader, hyps, device='cuda:0')

    trained_model = ConvAutoEncoder(hyps['os'], mode=mode)
    trained_model.load_state_dict(torch.load(save_dir + f'/mnsanitycheck-bs{hyps["bs"]}-lr{hyps["lr"]}.pt'))
    trained_model.to('cuda:0')

    to_pil_image = transforms.ToPILImage()

    # test autoencoder:
    gen_num = 10
    counter = 0
    seed = np.random.randint(0, 255)
    for tensor, _ in valid_loader:
        if counter >= gen_num:
            break

        tensor = tensor.to('cuda:0')
        reconst_tensor = trained_model(tensor)
        # ensure the directories exist
        create_dir(dest)
        image_dest = dest + f'/r-{counter}.png'
        recon_dest = dest + f'/p-{counter}.png'

        tensor = tensor.cpu()[seed]
        reconst_tensor = reconst_tensor.cpu()[seed]

        real_img = to_pil_image(tensor)
        pred_img = to_pil_image(reconst_tensor)

        real_img = real_img.resize((690, 690))
        pred_img = pred_img.resize((690, 690))

        real_img.save(image_dest)
        pred_img.save(recon_dest)

        counter += 1
