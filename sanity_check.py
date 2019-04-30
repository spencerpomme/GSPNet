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
    batch_size = 256
    num_workers = 0

    # data root
    data_dir = 'data/dogcheck'

    # preprocess images: shrink too big images:
    dog_files = np.array(glob(data_dir + '/*.jpg'))

    for path in dog_files:
        shrink_large_img(path)

    # training data transforms
    t = transforms.Compose([transforms.Resize(100), transforms.CenterCrop(69),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                            ])

    # Pass transforms in here, then run the next cell to see how the
    #  transforms look
    data = datasets.ImageFolder(data_dir, transform=t)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=True)
