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

Module of dataset(raw) generation.
A part of GSPNet project.

'''

import source as source
import rule as rule
import worker as worker
import viz as viz  # to be done

import os
import torch
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count


# multiprocessing function
def f_od(worker):
    '''
    Start a od worker's generation.

    Args:
        worker: a process object
    '''
    print(f'OD Generating worker {worker.pid}..')
    # For now manually change generate function.
    # worker.generate_pnf()
    worker.generate_od()


def f_pnf(worker):
    '''
    Start a pnf worker's generation.

    Args:
        worker: a process object
    '''
    print(f'PNF Generating worker {worker.pid}..')
    # For now manually change generate function.
    # worker.generate_pnf()
    worker.generate_pnf()


# main entry point of this module
def run(src: str, stp: str, etp: str, destdir: str, freqs: list, mode: str):
    '''
    Main function of this data generation module.

    Args:
        src: database table name string
        stp: start time point string
        etp: end time point string
        destdir: folder name string, under /tensor_data folder
        freqs: list of time interval strings
        mode: either `pnf` or `od`
    Example:
        >>> gen('cleaned_small_yellow_2018_full',
            '2018-06-01 00:00:00', '2018-07-01 00:00:00',
            'test', ['10min', '15min'], 'pnf')
    '''
    # Example:
    # src => 'cleaned_small_yellow_2017_full'
    # (stp, etp) => ('2017-01-01 00:00:00', '2017-02-01 00:00:00'))
    path = Path(os.path.dirname(os.path.realpath(__file__))
                ).resolve().parents[1].joinpath('data/processed/')
    taxi = source.DatabaseSource(src, (stp, etp))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges
    tb_size = len(tables)

    if mode == 'pnf':
        f = f_pnf
        path = path.joinpath('pnf/')
    elif mode == 'od':
        f = f_od
        path = path.joinpath('od/')
    else:
        raise ValueError(f'arg[5] mode expect `od` or `pnf`, but {mode} is provided.')

    total_start = time.time()
    # seems that only interval between 10min and 15min are usable.
    for freq in freqs:
        # multiprocessing generate data
        path = path.joinpath(f'{destdir}/' + f'{freq}/')
        print(path)
        workers = []
        for k in tables.keys():

            wp = worker.Worker(k, tables[k],
                               rule.TimeSlice(
                               *list(map(str, sub_ranges[k])), freq=freq),
                               str(path), True
                               )

            workers.append(wp)

        print(f'Start generating tensors with freq {freq} at {time.ctime()}')
        start = time.time()

        # create a process pool
        pn = cpu_count()
        print(f'Creating process pool with {pn} processes.')

        with Pool(pn) as pool:
            pool.map(f, workers)

            pool.close()
            pool.join()

        end = time.time()
        print(f'Tensors with freq {freq} finished at {time.ctime()}\
                in {end-start :.2f} seconds.\n\n')

    total_end = time.time()
    print(f'All generation ended in {total_end-total_start :.2f} seconds.')


if __name__ == '__main__':

    run('cleaned_small_yellow_2017_full', '2017-01-01 00:00:00',
        '2018-01-01 00:00:00', '2017', ['10min'], 'od')

    run('cleaned_small_yellow_2018_full', '2018-01-01 00:00:00',
        '2019-01-01 00:00:00', '2018', ['10min'], 'od')
