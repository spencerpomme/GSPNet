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

Module of dataset(raw) generation.
A part of GSPNet project.

'''

import timeslice
import timeslice.source as source
import timeslice.rule as rule
import timeslice.worker as worker
import timeslice.viz as viz  # to be done

import torch
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# multiprocessing function
def f(worker):
    '''
    Start a worker's generation.
    '''
    print(f'Generating worker {worker.pid}..')
    worker.generate()


# main entry point of this module
def gen(source: str, stp: str, etp: str, destdir: str, freqs: list):
    '''
    Main function of this data generation module.
    Args:
        source: database table name string
        stp: start time point string
        etp: end time point string
        destdir: folder name string, under /tensor_data folder
        freqs: list of time interval strings
    Example:
        gen('cleaned_small_yellow_2018_full',
            '2018-06-01 00:00:00', '2018-07-01 00:00:00',
            'test', ['10min', '15min'])
    '''
    DIRNAME = destdir

    # taxi = source.DatabaseSource('cleaned_small_yellow_2017_full', ('2017-01-01 00:00:00', '2017-02-01 00:00:00'))
    taxi = source.DatabaseSource(source,(stp, etp))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges
    tb_size = len(tables)

    total_start = time.time()
    # seems that only interval between 10min and 15min are usable.
    for freq in freqs:
        # multi processing generate data

        workers = []
        for k in tables.keys():

            # wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='10min'), 'full_year_10min', True)
            wp = worker.Worker(k, tables[k], rule.TimeSlice(
                *list(map(str, sub_ranges[k])), freq=freq), f'tensor_dataset/{DIRNAME}_{freq}', True)
            workers.append(wp)

        print(f'Start generating tensors with freq {freq} at {time.ctime()}\n')
        start = time.time()

        # create a process pool
        pn = cpu_count()
        print(f'Creating pool with {pn} processes.')

        with Pool(pn) as pool:
            pool.map(f, workers)

            pool.close()
            pool.join()

        end = time.time()
        print(
            f'Generation of tensors with freq {freq} finished at {time.ctime()} in {end-start :2f} seconds.\n\n')
    total_end = time.time()

    print(f'Entire process of generating tensor datas ended in {total_end-total_start :2f} seconds.')


if __name__ == '__main__':

    gen('cleaned_small_yellow_2018_full', '2018-06-01 00:00:00',
        '2018-07-01 00:00:00', 'test', ['15min'])
