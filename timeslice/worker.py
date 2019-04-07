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

##============================================================================##

'''
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import time
import os
import copy

from PIL import Image
from tqdm import tqdm
from numba import jit
from numba import types
from numba.typed import Dict

# column name: index mapping
colmap = {
    'tripid': 0,
    'tpep_pickup_datetime': 1,
    'tpep_dropoff_datetime': 2,
    'pulocationid': 3,
    'dolocationid': 4
}

# helper functions
# Function gen_snap_layers is hard to optimize for the slicing part depends heavily on
# timestamp based selection. This functionality would be rather cumbersome if to
# be done in numpy.


def gen_snap_layers(table, bound):
    '''
    Generate Past layer, Now layer and Future layer for one snapshot.
    Params:
        table: pandas tabular data
        bounds: time bound tuple, for example: (left_timestring, right_timestring)
    Return:
        PNF layers, a list.
    '''
    # left bound and right bound of time interval
    assert type(bound) == tuple
    left = bound[0]
    right = bound[1]

    # The .loc operation below is moved to be executed in CSVSource initialization.
    # table = table.loc[:, ['tripid',
    #                                 'tpep_pickup_datetime',
    #                                 'tpep_dropoff_datetime',
    #                                 'pulocationid',
    #                                 'dolocationid']]

    # The condition of making snapshot should be:
    # AT LEAST ONE temporal end of a trip should be within the bounds:
    snap = table.loc[
        ((table['tpep_pickup_datetime'] >= left) &
         (table['tpep_pickup_datetime'] < right)) |
        ((table['tpep_dropoff_datetime'] >= left) &
         (table['tpep_dropoff_datetime'] < right))]

    # temp table to generate F,P,N layers
    # keep snap intact
    temp_snap = snap.copy()

    # Use the interval to 'catch' corresponding trips.
    # future layer
    f_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] < right) &
                            (temp_snap['tpep_pickup_datetime'] >= left) &
                            (temp_snap['tpep_dropoff_datetime'] >= right)]
    # past layer
    p_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] < left) &
                            (temp_snap['tpep_dropoff_datetime'] >= left) &
                            (temp_snap['tpep_dropoff_datetime'] < right)]
    # now layer
    n_layer = temp_snap.loc[(temp_snap['tpep_pickup_datetime'] >= left) &
                            (temp_snap['tpep_dropoff_datetime'] < right)]

    # Their count should add up to total trips caught
    assert temp_snap.shape[0] == f_layer.shape[0] + \
        p_layer.shape[0] + n_layer.shape[0]

    return p_layer, n_layer, f_layer


def gen_image(p_layer, n_layer, f_layer):
    '''
    Generate an image using given matrices.
    Params:
        p_layer: matrix of past layer, pandas dataframe
        n_layer: matrix of now layer, pandas dataframe
        f_layer: matrix of future layer, pandas dataframe
    Return:
        A PIL image.
    '''
    # create a snapshot
    snapshot = np.zeros(
        [Worker.image_size, Worker.image_size, 3], dtype='int32')

    # unexpected zones
    left_zones = set()

    # future-Red: 0
    for _, row in f_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 0] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # past-Green: 1
    for _, row in p_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 1] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # now-Blue: 2
    for _, row in n_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 2] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # simple normalize
    snapshot *= (255 // snapshot.max())
    snapshot = snapshot.astype('uint8')
    image = Image.fromarray(snapshot)
    return image


def gen_tensor(p_layer, n_layer, f_layer):
    '''
    Generate a tensor using given matrices.
    Params:
        p_layer: matrix of past layer, pandas dataframe
        n_layer: matrix of now layer, pandas dataframe
        f_layer: matrix of future layer, pandas dataframe
        
    Return:
        A torch tensor.
    '''
    # create a snapshot
    snapshot = np.zeros(
        [Worker.image_size, Worker.image_size, 3], dtype='float64')

    # unexpected zones
    left_zones = set()

    # future-Red: 0
    for _, row in f_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 0] += 1.0
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # past-Green: 1
    for _, row in p_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 1] += 1.0
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # now-Blue: 2
    for _, row in n_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])],
                     Worker.mp[str(row['dolocationid'])], 2] += 1.0
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # set all empty

    # normalize
    sm = snapshot.max()
    # print(sm)
    snapshot *= (255 // sm)
    snapshot = torch.from_numpy(snapshot)

    return snapshot


# adjacency matrix creater function
@jit(parallel=True)
def create_adjacency_matrix(arr, am, ly: int):
    '''
    Fill in values into the provided am(adjacency matrxi) with the connection info
    numpy array. Copy a list here for performance
    (dict is not supported even just plainly put within jit decorated function)

    Args:
        arr: OD information, 2d numpy array
        am: adjacency matrix, zero 2d numpy array
        ly: layer number of 3-layer tensor

    Returns:
        am: a filled adjacency matrix 2d numpy array
    '''
    mapping = {
        4: 0,   12: 1,   13: 2,   24: 3,   41: 4,   42: 5,
        43: 6,   45: 7,   48: 8,   50: 9,   68: 10,  74: 11,
        75: 12,  79: 13,  87: 14,  88: 15,  90: 16, 100: 17,
        103: 18, 104: 19, 105: 20, 107: 21, 113: 22, 114: 23,
        116: 24, 120: 25, 125: 26, 127: 27, 128: 28, 137: 29,
        140: 30, 141: 31, 142: 32, 143: 33, 144: 34, 148: 35,
        151: 36, 152: 37, 153: 38, 158: 39, 161: 40, 162: 41,
        163: 42, 164: 43, 166: 44, 170: 45, 186: 46, 194: 47,
        202: 48, 209: 49, 211: 50, 224: 51, 229: 52, 230: 53,
        231: 54, 232: 55, 233: 56, 234: 57, 236: 58, 237: 59,
        238: 60, 239: 61, 243: 62, 244: 63, 246: 64, 249: 65,
        261: 66, 262: 67, 263: 68
    }

    for i in range(arr.shape[0]):
        # this twisted roundabout is due to not supported feature for iterating 2d arrays:
        am[mapping[arr[i, :][3]],
           mapping[arr[i, :][4]],
           ly] += 1

    return am


# adjacency matrix creater function
# @jit(nopython=True, parallel=True)
# def create_adjacency_matrix(arr, am):
#     '''
#     Fill in values into the provided am(adjacency matrxi) with the connection info
#     numpy array.

#     Args:
#         arr: OD information, 2d numpy array
#         am: adjacency matrix, zero 2d numpy array

#     Returns:
#         am: a filled adjacency matrix 2d numpy array
#     '''
#     for i in range(arr.shape[0]):
#         # this twisted roundabout is due to not supported feature for iterating 2d arrays:
#         am[arr[i, :][3], arr[i, :][4]] += 1

#     return am


# numba enhanced version
def gen_image_fast(p_layer, n_layer, f_layer):
    '''
    Generate an image using given matrices.
    Params:
        p_layer: matrix of past layer, pandas dataframe
        n_layer: matrix of now layer, pandas dataframe
        f_layer: matrix of future layer, pandas dataframe
    Return:
        A PIL image.
    '''
    # convert pandas dataframe to numpy array, only get OD columns
    p_layer = p_layer.to_numpy()[1:, :]
    n_layer = n_layer.to_numpy()[1:, :]
    f_layer = f_layer.to_numpy()[1:, :]

    # create a snapshot
    snapshot = np.zeros(
        [Worker.image_size, Worker.image_size, 3], dtype='int16')

    # future-Red: 0
    snapshot = create_adjacency_matrix(p_layer, snapshot, 0)

    # past-Green: 1
    snapshot = create_adjacency_matrix(n_layer, snapshot, 1)

    # now-Blue: 2
    snapshot = create_adjacency_matrix(f_layer, snapshot, 2)

    # simple normalize
    snapshot *= (255 // snapshot.max())
    snapshot = snapshot.astype('uint8')
    image = Image.fromarray(snapshot)

    return image


# numba enhanced version
def gen_tensor_fast(p_layer, n_layer, f_layer):
    '''
    Generate a tensor using given matrices.
    Params:
        p_layer: matrix of past layer, pandas dataframe
        n_layer: matrix of now layer, pandas dataframe
        f_layer: matrix of future layer, pandas dataframe

    Return:
        A torch tensor.
    '''
    # convert pandas dataframe to numpy array, only get OD columns
    p_layer = p_layer.to_numpy()[1:, :]
    n_layer = n_layer.to_numpy()[1:, :]
    f_layer = f_layer.to_numpy()[1:, :]

    # create a snapshot
    snapshot = np.zeros(
        [Worker.image_size, Worker.image_size, 3], dtype='int64')

    # future-Red: 0
    snapshot = create_adjacency_matrix(p_layer, snapshot, 0)

    # past-Green: 1
    snapshot = create_adjacency_matrix(n_layer, snapshot, 1)

    # now-Blue: 2
    snapshot = create_adjacency_matrix(f_layer, snapshot, 2)

    # # normalize
    # sm = snapshot.max()
    # # print(sm)
    # snapshot *= (255 // sm)
    snapshot = torch.from_numpy(snapshot)

    return snapshot


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


class Worker:
    '''
    Worker class, generate tensors.
    '''
    # data used for the class
    zones = pd.read_csv('rawcsv/taxi_zone_lookup.csv')

    # very important globals:
    yellow_zone = zones.loc[zones['Borough'] == 'Manhattan']

    image_size = yellow_zone.shape[0]

    # create a mapping {'district_id_in_data_source': index from 0 to 68}
    real_id = list(map(str, list(yellow_zone.loc[:, 'LocationID'])))
    conv_id = [i for i in range(image_size)]

    # create matrix id mapping
    assert len(real_id) == len(conv_id)
    mp = dict(zip(real_id, conv_id))

    def __init__(self, pid: int, table, rule, destin: str, viz: bool):
        '''
        Init method for Worker.

        Args:
            pid:    process id
            destin: String, indicating where to store the generated tensors and
                    visualization images of the tensors (if any)
            table:  a dataframe
            rule:   Rule object that determines how to operate on source data
            subts:  sub time slice interval
            destin: directory to save generated tensor and image
            viz:    Boolean value, decide create visualization image of tonsors
                    or not
        '''

        self.pid = pid
        self.table = self.clean_rows(table)
        self.rule = rule

        # check if the destination dir exists.
        if not os.path.exists(destin):
            create_dir(destin)

        self.tensor_dir = destin + '/tensors'
        create_dir(self.tensor_dir)
        self.viz = viz
        if viz:
            self.visual_dir = destin + '/viz_images'
            create_dir(self.visual_dir)

    def __repr__(self):
        '''
        Representation method for Worker object.
        '''
        return f'Worker object: \n\
                 destination: {self.tensor_dir}\n\
                 visualize: {self.viz}\n\
                 viz_dir: {self.visual_dir if self.visual_dir else None}'

    def clean_rows(self, table):
        '''
        Remove rows in the table(pandas dataframe) if either of its location ID
        is not in Worker.mp

        Args:
            table: pandas DataFrame
        
        Returns:
            table: pandas DataFrame
        '''
        # print(f'before cleansing table size: {table.shape}')
        mapping = {
            '4': 0,   '12': 1,   '13': 2,   '24': 3,   '41': 4,   '42': 5,
            '43': 6,   '45': 7,   '48': 8,   '50': 9,   '68': 10,  '74': 11,
            '75': 12,  '79': 13,  '87': 14,  '88': 15,  '90': 16, '100': 17,
            '103': 18, '104': 19, '105': 20, '107': 21, '113': 22, '114': 23,
            '116': 24, '120': 25, '125': 26, '127': 27, '128': 28, '137': 29,
            '140': 30, '141': 31, '142': 32, '143': 33, '144': 34, '148': 35,
            '151': 36, '152': 37, '153': 38, '158': 39, '161': 40, '162': 41,
            '163': 42, '164': 43, '166': 44, '170': 45, '186': 46, '194': 47,
            '202': 48, '209': 49, '211': 50, '224': 51, '229': 52, '230': 53,
            '231': 54, '232': 55, '233': 56, '234': 57, '236': 58, '237': 59,
            '238': 60, '239': 61, '243': 62, '244': 63, '246': 64, '249': 65,
            '261': 66, '262': 67, '263': 68
        }

        table = table.loc[(table['pulocationid'].isin(mapping.keys())) &
                          (table['dolocationid'].isin(mapping.keys()))]
        # print(f'after cleansing table size: {table.shape}')

        return table

    def generate(self):
        '''
        Generate tensors given the data source and processing rules.

        Args:
            table: pandas.DataFrame object, one sub table from the
                   source.table_pool dictionary.
            pid: a number indicating the generation process id


        ***************************| benchmark |****************************
        | A full run of entire 2017 data (yellow regions) is approximately |
        | 100 minutes, which is extremely slow.                            |
        ********************************************************************
        '''
        for i, bound in enumerate(self.rule.fragments):

            # print(f'Generating tensor No.{i} : {bound}')
            # generate three layers
            p_layer, n_layer, f_layer = gen_snap_layers(self.table, bound)
            # print(table.head())

            # combine three layers to one tensor(image)
            tensor = gen_tensor_fast(p_layer, n_layer, f_layer)

            # start and end bound for entire sub interval
            stp = self.rule.stp
            etp = self.rule.etp

            # left bound and right bound of ONE time slice of the time interval
            lbd = bound[0]
            rbd = bound[1]

            # tensor save path
            tensor_path = os.path.abspath(
                self.tensor_dir +
                f'/{lbd}-{rbd}--{stp}-{etp}--p{self.pid}-{i}.pkl'.replace(
                    ' ', '_').replace(':', ';')
            )

            # image save path
            image_path = os.path.abspath(
                self.visual_dir +
                f'/{lbd}-{rbd}--{stp}-{etp}--p{self.pid}-{i}.jpg'.replace(
                    ' ', '_').replace(':', ';')
            )

            # save method 1 => time for 1 day is: 7m 47s
            torch.save(tensor, tensor_path)

            # if viz is true, then save images to separate folder
            if self.viz:

                image = gen_image_fast(p_layer, n_layer, f_layer)

                # resize to x50
                # vimage = image.resize((345,345))
                image.save(image_path)
