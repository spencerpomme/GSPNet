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

    return p_layer, n_layer, f_layer


def gen_image(p_layer, n_layer, f_layer):
    '''
    Generate an image using given matrices.
    Params:
        p_layer: matrix of past layer
        n_layer: matrix of now layer
        f_layer: matrix of future layer
    Return:
        A PIL image.
    '''
    # create a snapshot
    snapshot = np.zeros([img_size, img_size, 3], dtype='float64')

    # unexpected zones
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

    # normalize
    snapshot *= 255 // snapshot.max()
    snapshot = snapshot.astype('uint8')
    image = Image.fromarray(snapshot)
    return image


def timesplit(stp: str, etp: str, freq='10min'):
    '''
    Create a DatetimeIndx interval.
    
    Params:
        stp: string, starting time point, first left bound
        etp: string, ending time point, last right bound
        freq: frequency, time interval unit of the splice operation
    The stp and etp must of pattern "yyyy-mm-dd hh:mm:ss", otherwise exception will be raised.
    
    Return:
        A list of time intervals tuples,each item is a tuple of two
        interval(i.e., pandas.core.indexes.datetimes.DatetimeIndex object)
        For example, a possible return could be [(2017-01-01 00:00:00, 2017-01-01 00:10:00),
                                                 (2017-01-01 00:10:00, 2017-01-01 00:20:00)]
    '''
    # Regex to match datetime string
    pattern = re.compile(
        '^([0-9]{4})-([0-1][0-9])-([0-3][0-9])\s([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$'
    )

    if pattern.match(stp) and pattern.match(etp):
        time_bounds = pd.date_range(stp, etp, freq=freq)
        sub_intervals = list(zip(time_bounds[:-1], time_bounds[1:]))
        
        return sub_intervals
    else:
        raise Exception('Provided time bound is of invalid format.')
