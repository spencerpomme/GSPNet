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
from multiprocessing import Process, Queue, cpu_count


# helper functions
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
    
    #print(left, right)
    # no need to sort table indeed?
    projected_table = table.loc[:, ['tripid',
                                    'tpep_pickup_datetime',
                                    'tpep_dropoff_datetime',
                                    'pulocationid',
                                    'dolocationid']]

    projected_table.head(5)

    # The condition of making snapshot should be:
    # AT LEAST ONE temporal end of a trip should be within the bounds:
    snap = projected_table.loc[
        ((projected_table['tpep_pickup_datetime'] >= left) &
         (projected_table['tpep_pickup_datetime'] < right)) |
        ((projected_table['tpep_dropoff_datetime'] >= left) &
         (projected_table['tpep_dropoff_datetime'] < right))]

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
    snapshot = np.zeros([Worker.image_size, Worker.image_size, 3], dtype='int32')

    # unexpected zones
    left_zones = set()

    # future-Red: 0
    for _, row in f_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 0] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # past-Green: 1
    for _, row in p_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 1] += 1
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # now-Blue: 2
    for _, row in n_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 2] += 1
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
        p_layer: matrix of past layer
        n_layer: matrix of now layer
        f_layer: matrix of future layer
        
    Return:
        A torch tensor.
    '''
    # create a snapshot
    img_size = Worker.image_size
    snapshot = np.zeros([Worker.image_size, Worker.image_size, 3], dtype='float64')

    # unexpected zones
    left_zones = set()

    # future-Red: 0
    for _, row in f_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 0] += 1.0
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # past-Green: 1
    for _, row in p_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 1] += 1.0
        except Exception as e:
            left_zones.add(str(row['pulocationid']))
            left_zones.add(str(row['dolocationid']))

    # now-Blue: 2
    for _, row in n_layer.iterrows():
        try:
            snapshot[Worker.mp[str(row['pulocationid'])], Worker.mp[str(row['dolocationid'])], 2] += 1.0
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
        print('Error: Creating directory. ' +  directory)
        raise OSError


def f(worker):
    worker.generate()



# This method is currently not working, need to modify
def parallel_gen(source, rule, destin='.', viz=True):
    '''
    Generate tonsors in parrele using multiprocessing.

    The tables passed from Source instance should be applied generation
    function in parallel.

    Optimized process number is hard coded.

    '''
    source.load()
    tables = source.table_pool # <- dictionary

    # parallel object holders
    process_pool = Queue()
    process_buffer = []

    # important variables
    tb_size = len(tables)

    # efficient process number
    p_unit = cpu_count()

    # initialize process bar
    progress = tqdm(total=tb_size, ascii=True)

    print(f'Tensor generation started at {time.ctime()}')
    start = time.time()

    # create process and put into a queue
    for pid, table in tables.items():

        # instantiate a new Worker object
        gen_worker = Worker(pid, table, rule, destin, viz)

        # table, rule, pid:int, tensor_dir, visual_dir, viz=True
        p = Process(target=f, args=(gen_worker,))
        process_pool.put(p)

    # do actual tensor creation and serialization
    while not process_pool.empty():
        for i in range(min(p_unit, tb_size)):
            p = process_pool.get()
            process_buffer.append(p)
        
        # number of tables remain not transformed to tensor
        done_processes = min(p_unit, tb_size)
        tb_size -= done_processes
        progress.update(done_processes)

        # start processes
        for p in process_buffer:
            p.start()

        # join processes
        for p in process_buffer:
            p.join()

        # clear finished processes from process buffer
        process_buffer = []

    end = time.time()
    print(f'Ended at {time.ctime()}, total time {end - start:.2f} seconds.')



class Worker:
    '''
    Worker class, generate tensors.
    '''
    # data used for the class
    zones = pd.read_csv('dataset/taxi_zone_lookup.csv')

    # very important globals:
    yellow_zone = zones.loc[zones['Borough'] == 'Manhattan']
    
    image_size = yellow_zone.shape[0]

    # create a mapping {'district_id_in_data_source': index from 0 to 68}
    real_id = list(map(str, list(yellow_zone.loc[:,'LocationID'])))
    conv_id = [i for i in range(image_size)]

    # create matrix id mapping
    assert len(real_id) == len(conv_id)
    mp = dict(zip(real_id, conv_id))

    def __init__(self, pid:int, table, rule, destin:str, viz:bool):
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
        self.table = table
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
            tensor = gen_tensor(p_layer, n_layer, f_layer)
            # print(tensor)

            # start and end bound for entire sub interval
            stp = self.rule.stp
            etp = self.rule.etp

            # left bound and right bound of ONE time slice of the time interval
            lbd = bound[0]
            rbd = bound[1]

            # tensor save path
            tensor_path = os.path.abspath(
                self.tensor_dir + f'/{lbd}-{rbd}-{stp}-{etp}-p{self.pid}-{i}.pkl'.replace(' ', '_').replace(':',';')
            )

            # image save path
            image_path = os.path.abspath(
                self.visual_dir + f'/{lbd}-{rbd}-{stp}-{etp}-p{self.pid}-{i}.jpg'.replace(' ', '_').replace(':',';')
            )

            # save method 1 => time for 1 day is: 7m 47s
            torch.save(tensor, tensor_path)

            # if viz is true, then save images to separate folder
            if self.viz:
                
                image = gen_image(p_layer, n_layer, f_layer)

                # resize to x50
                vimage = image.resize((345,345))
                vimage.save(image_path)




        