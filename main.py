import timeslice                    
import timeslice.source as source   
import timeslice.rule  as rule      
import timeslice.worker as worker
import timeslice.viz as viz # to be done
import torch
import time
from tqdm import tqdm
from multiprocessing import Queue, Process, cpu_count



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



if __name__ == '__main__':

    # connect to database source
    taxi = source.DatabaseSource('cleaned_small_yellow_2017_full', ('2017-09-10 00:00:00', '2018-01-01 00:00:00'))
    taxi.load()

    tables = taxi.table_pool
    sub_ranges = taxi.sub_ranges


    print(f'Start generating tensors at {time.ctime()}')
    # worker_0 = worker.Worker(0, tables[0], rule.TimeSlice(*list(map(str, sub_ranges[0])), freq='15min'), 'new_test_dir', True)
    # worker_1 = worker.Worker(1, tables[1], rule.TimeSlice(*list(map(str, sub_ranges[1])), freq='15min'), 'new_test_dir', True)
    # worker_2 = worker.Worker(2, tables[2], rule.TimeSlice(*list(map(str, sub_ranges[2])), freq='15min'), 'new_test_dir', True)
    # worker_3 = worker.Worker(3, tables[3], rule.TimeSlice(*list(map(str, sub_ranges[3])), freq='15min'), 'new_test_dir', True)
    # worker_4 = worker.Worker(4, tables[4], rule.TimeSlice(*list(map(str, sub_ranges[4])), freq='15min'), 'new_test_dir', True)

    workers = []
    for k in tables.keys():

        wp = worker.Worker(k, tables[k], rule.TimeSlice(*list(map(str, sub_ranges[k])), freq='15min'), 'full_year_test', True)
        workers.append(wp)


    start = time.time()


    for worker in tqdm(workers, ascii=True):

        worker.generate()

    print(f'Generation finished at {time.ctime()}')
    tqdm.close()