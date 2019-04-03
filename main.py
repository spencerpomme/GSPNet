import timeslice                    
import timeslice.source as source   
import timeslice.rule  as rule      
import timeslice.worker as worker
import timeslice.viz as viz # to be done
import torch
import time
from tqdm import tqdm
from multiprocessing import Queue, Process, cpu_count


def f(worker):
    '''
    Start a worker process.
    '''
    worker.start()



if __name__ == '__main__':

    # connect to database source
    p_unit = cpu_count()

    taxi = source.DatabaseSource('cleaned_small_yellow_2017_full', ('2017-01-01 00:00:00', '2018-01-01 00:00:00'))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges

    process_pool = Queue()
    process_buffer = []
    tb_size = len(tables)


    # multi processing generate data
    print(f'Start generating tensors at {time.ctime()}')

    workers = []
    for k in tables.keys():

        wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='10min'), 'full_year_10min', True)
        workers.append(wp)


    start = time.time()



    for worker in workers:

        # worker.generate()
        p = Process(target=f, args=(worker,))
        process_pool.put(p)

    progress = tqdm(total=tb_size, ascii=True)

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


    print(f'Generation finished at {time.ctime()}')