import timeslice                    
import timeslice.source as source   
import timeslice.rule  as rule      
import timeslice.worker as worker
import timeslice.viz as viz # to be done

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
    taxi = source.DatabaseSource('cleaned_small_yellow_2017_full', ('2017-04-01 00:00:00', '2018-01-01 00:00:00'))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges
    tb_size = len(tables)
    workers = []


    # multi processing generate data
    for k in tables.keys():

        # wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='10min'), 'full_year_10min', True)
        wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='10min'), 'full_year_10min/rest', True)
        workers.append(wp)

    print(f'Start generating tensors at {time.ctime()}')
    start = time.time()

    ############################### code start here ##############################
    
    # create a process pool
    pn = cpu_count() * 2
    print(f'Creating pool with {pn} processes.')
    
    with Pool(pn) as pool:
        pool.map(f, workers)

        pool.close()
        pool.join()

    # for worker in workers:
    #     worker.generate()
    
    end = time.time()
    print(f'Generation finished at {time.ctime()} in {end-start :2f} seconds.')