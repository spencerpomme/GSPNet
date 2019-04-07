import timeslice
import timeslice.source as source
import timeslice.rule as rule
import timeslice.worker as worker
import timeslice.viz as viz  # to be done

import torch
import time
from tqdm import tqdm

from multiprocessing import Process, Pool, cpu_count


def f(worker):
    '''
    Start a worker's generation.
    '''
    print(f'Generating worker {worker.pid}..')
    worker.generate()


if __name__ == '__main__':

    # taxi = source.DatabaseSource('cleaned_small_yellow_2017_full', ('2017-01-01 00:00:00', '2017-02-01 00:00:00'))
    taxi = source.DatabaseSource('cleaned_small_yellow_2017_08',
                                 ('2017-08-01 00:00:00', '2017-09-01 00:00:00'))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges
    tb_size = len(tables)

    total_start = time.time()
    # seems that only interval between 10min and 15min are usable.
    for freq in ['15min']:
        # multi processing generate data

        workers = []
        for k in tables.keys():

            # wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='10min'), 'full_year_10min', True)
            wp = worker.Worker(k, tables[k], rule.TimeSlice(
                *list(map(str, sub_ranges[k])), freq=freq), f'tensor_dataset/nn_test_{freq}_val', True)
            workers.append(wp)

        print(f'Start generating tensors with freq {freq}at {time.ctime()}\n')
        start = time.time()

        ############################### code start here ##############################

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
    print(
        f'Entire process of generating tensor datas ended in {total_end-total_start :2f} seconds.')
