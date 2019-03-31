import timeslice                    
import timeslice.source as source   
import timeslice.rule  as rule      
import timeslice.worker as worker
import timeslice.viz as viz # to be done
import torch
import time
from tqdm import tqdm


# connect to database source
taxi = source.DatabaseSource('cleaned_small_yellow_2017_02', ('2017-02-01 00:00:00', '2017-03-01 00:00:00'))
taxi.load()

tables = taxi.table_pool
sub_ranges = taxi.sub_ranges

# for i, table in tables.items():
#     print(f'table No.{i} -> table shape: {table.shape} table type: {type(table)}')

# for sub_range in sub_ranges:
#     print(f'time sub slice: {sub_range}')


r = rule.TimeSlice(*list(map(str, sub_ranges[0])), freq='15min')
print(r)

print(f'Start generating tensors at {time.ctime()}')
worker_0 = worker.Worker(0, tables[0], rule.TimeSlice(*list(map(str, sub_ranges[0])), freq='15min'), 'new_test_dir', True)
worker_1 = worker.Worker(1, tables[1], rule.TimeSlice(*list(map(str, sub_ranges[1])), freq='15min'), 'new_test_dir', True)
worker_2 = worker.Worker(2, tables[2], rule.TimeSlice(*list(map(str, sub_ranges[2])), freq='15min'), 'new_test_dir', True)
worker_3 = worker.Worker(3, tables[3], rule.TimeSlice(*list(map(str, sub_ranges[3])), freq='15min'), 'new_test_dir', True)
worker_4 = worker.Worker(4, tables[4], rule.TimeSlice(*list(map(str, sub_ranges[4])), freq='15min'), 'new_test_dir', True)


workers = [worker_0, worker_1, worker_2, worker_3, worker_4]

start = time.time()


for worker in workers:

    worker.generate()

print(f'Generation finished at {time.ctime()}')