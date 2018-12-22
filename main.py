'''
Copyright (c) <2018> <Pingcheng Zhang>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Main module of GSPNet project.
'''

import pandas as pd
import numpy as np
import seaborn as sns
import sqlalchemy
import dask
import time
import psycopg2
import warnings
import re
from PIL import Image
from IPython.display import display
from scipy import stats
from matplotlib import pyplot as plt

# read data
table = pd.read_csv('dataset/nytaxi_yellow_2017_jan.csv')
# Load zone lookup table
zones = pd.read_csv('dataset/taxi_zone_lookup.csv')

# create time interval
intervals = pd.date_range('2017-01-01 00:00:00', '2017-02-01 00:00:00', freq='10min')

# set image size
y_zone = zones.loc[zones['Borough'] == 'Manhattan']
img_size = y_zone.shape[0]

# Before generating image, we have to map the zone id to a range from 0 to 54:
real_id = list(map(str, list(y_zone.loc[:, 'LocationID'])))
conv_id = [i for i in range(img_size)]
assert len(real_id) == len(conv_id)
mp = dict(zip(real_id, conv_id))


# TODO: finish this! 2018/12/17
def gen_snap_layers(table, intervals, bounds):
    '''
    Generate Past layer, Now layer and Future layer for one snapshot.
    Params:
        table:
        intervals:
        bounds:
    Return:
        PNF layers, a list.
    '''
    left = 72
    right = 73

    # convert dtype for entire table here:
    table['tpep_pickup_datetime'] = pd.to_datetime(table['tpep_pickup_datetime'])
    table['tpep_dropoff_datetime'] = pd.to_datetime(table['tpep_dropoff_datetime'])

    sorted_table = table.loc[:, ['tripid',
                                 'tpep_pickup_datetime',
                                 'tpep_dropoff_datetime',
                                 'pulocationid',
                                 'dolocationid']].sort_values(by=['tpep_pickup_datetime',
                                                                  'tpep_dropoff_datetime'])

    # The condition of making snapshot should be:
    # at least one temporal end of a trip should be within the bounds:
    snap = sorted_table.loc[
        ((sorted_table['tpep_pickup_datetime'] >= intervals[left]) &
         (sorted_table['tpep_pickup_datetime'] < intervals[right])) |
        ((sorted_table['tpep_dropoff_datetime'] >= intervals[left]) &
         (sorted_table['tpep_dropoff_datetime'] < intervals[right]))]

    # print(f'sorted_table.shape -> {sorted_table.shape}')
    # print(f'snap.shape -> {snap.shape}')

    # temp table to generate F,P,N layers
    # keep snap intact
    temp_snap = snap.loc[:, ['tripid',
                             'tpep_pickup_datetime',
                             'tpep_dropoff_datetime',
                             'pulocationid',
                             'dolocationid']]

    # Use the interval to 'catch' corresponding trips.
    # future layer
    f_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] < intervals[right]) &
                             (temp_snap['tpep_pickup_datetime'] >= intervals[left]) &
                             (temp_snap['tpep_dropoff_datetime'] >= intervals[right])]
    # past layer
    p_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] < intervals[left]) &
                             (temp_snap['tpep_dropoff_datetime'] >= intervals[left]) &
                             (temp_snap['tpep_dropoff_datetime'] < intervals[right])]
    # now layer
    n_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] >= intervals[left]) &
                             (temp_snap['tpep_dropoff_datetime'] < intervals[right])]

    # Their count should add up to total trips caught
    assert temp_snap.shape[0] == f_layer.shape[0] + p_layer.shape[0] + n_layer.shape[0]


def gen_image_set(p_layer, n_layer, f_layer):
    '''
    Generate an image using given matrices, both the image and the seperate layer images.
    '''
    # create a snapshot:
    snapshot = np.zeros([img_size, img_size, 3], dtype='float64')
    print(snapshot.shape)
    print(snapshot[1, 2, 1])

    left_zones = set()

    # future-Red: 0
    for _, row in f_layer.iterrows():
        try:
            snapshot[mp[str(row['pulocationid'])], mp[str(row['dolocationid'])], 0] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # past-Green: 1
    for _, row in p_layer.iterrows():
        try:
            snapshot[mp[str(row['pulocationid'])], mp[str(row['dolocationid'])], 1] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # now-Blue: 2
    for _, row in n_layer.iterrows():
        try:
            snapshot[mp[str(row['pulocationid'])], mp[str(row['dolocationid'])], 2] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    print(f'left_zones: {left_zones}')
    print(f'left_zones length: {len(left_zones)}')

    print(f'O max -> {snapshot[:,:,0].max()}')
    print(f'I max -> {snapshot[:,:,1].max()}')
    print(f'D max -> {snapshot[:,:,2].max()}')

    snapshot *= 255 // snapshot.max() # normalize
