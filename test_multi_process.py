import multiprocessing
from threading import Thread
from multiprocessing import Process
import numpy as np
import time

import timeslice                    # to be created library
import timeslice.source as source   # sub module in rnnpipe lib, define source raw data
import timeslice.rule  as rule      # sub module in rnnpipe lib, define transformation rule

import timeslice.viz as viz         # visualize dataset if apply
import torch


def f(worker, worker_id):
    print(f'Worker {id} started!')
    worker.generate()
    

def double(ls):
    for i in range(len(ls)):
        ls[i] *= 2
    print(ls)

if __name__ == '__main__':

    import timeslice                    # to be created library
    import timeslice.source as source   # sub module in rnnpipe lib, define source raw data
    import timeslice.rule  as rule      # sub module in rnnpipe lib, define transformation rule
    import timeslice.worker as worker   # generate dataset

    import timeslice.viz as viz         # visualize dataset if apply
    import torch

    # connect to database
    # taxi_dbs = source.DatabaseSource(db='database connect info')

    # or use a csv file as source
    # taxi_csv_05 = source.CSVSource(file='dataset/nytaxi_yellow_2017_05.csv')
    # taxi_csv_06 = source.CSVSource(file='dataset/nytaxi_yellow_2017_06.csv')
    taxi_csv_07 = source.CSVSource(file='dataset/nytaxi_yellow_2017_07.csv')
    taxi_csv_08 = source.CSVSource(file='dataset/nytaxi_yellow_2017_08.csv')
    # taxi_csv_09 = source.CSVSource(file='dataset/nytaxi_yellow_2017_09.csv')
    # taxi_csv_10 = source.CSVSource(file='dataset/nytaxi_yellow_2017_10.csv')
    taxi_csv_11 = source.CSVSource(file='dataset/nytaxi_yellow_2017_11.csv')
    taxi_csv_12 = source.CSVSource(file='dataset/nytaxi_yellow_2017_12.csv')

    taxi_csv_07.load()
    taxi_csv_08.load()
    taxi_csv_11.load()
    taxi_csv_12.load()

    # set time split rule to generate different dataset
    # time_rule_05 = rule.TimeSlice(stp='2017-05-01 00:00:00', etp='2017-06-01 00:00:00', freq='15min')
    # time_rule_06 = rule.TimeSlice(stp='2017-06-01 00:00:00', etp='2017-07-02 00:00:00', freq='15min')
    time_rule_07 = rule.TimeSlice(stp='2017-07-01 00:00:00', etp='2017-08-01 00:00:00', freq='60min')
    time_rule_08 = rule.TimeSlice(stp='2017-08-01 00:00:00', etp='2017-09-01 00:00:00', freq='60min')
    # time_rule_09 = rule.TimeSlice(stp='2017-09-01 00:00:00', etp='2017-10-01 00:00:00', freq='15min')
    # time_rule_10 = rule.TimeSlice(stp='2017-10-01 00:00:00', etp='2017-11-01 00:00:00', freq='15min')
    time_rule_11 = rule.TimeSlice(stp='2017-11-01 00:00:00', etp='2017-12-01 00:00:00', freq='60min')
    time_rule_12 = rule.TimeSlice(stp='2017-12-01 00:00:00', etp='2018-01-01 00:00:00', freq='60min')

    # instantiate data worker object
    # data_worker_05 = worker.Worker(source=taxi_csv_05, destin='yearly_continuous_test', rule=time_rule_05, viz=True)
    # data_worker_06 = worker.Worker(source=taxi_csv_06, destin='yearly_continuous_test', rule=time_rule_06, viz=True)
    data_worker_07 = worker.Worker(7, taxi_csv_07.table, time_rule_07, 'new_test', True)
    
    data_worker_08 = worker.Worker(8, taxi_csv_08.table, time_rule_08, 'new_test', True)
    # data_worker_09 = worker.Worker(source=taxi_csv_09, destin='yearly_continuous_test_mp', rule=time_rule_09, viz=True)
    # data_worker_10 = worker.Worker(source=taxi_csv_10, destin='yearly_continuous_test_mp', rule=time_rule_10, viz=True)
    data_worker_11 = worker.Worker(11, taxi_csv_11.table, time_rule_11, 'new_test', True)
    data_worker_12 = worker.Worker(12, taxi_csv_12.table, time_rule_12, 'new_test', True)
  

    print(f'Using multiprocessing...')
    print(f'Started at {time.ctime()}')
    stp = time.time()
    # p5 = Process(target=f, args=(data_worker_05, 5))
    # p6 = Process(target=f, args=(data_worker_06, 6))
    p7 = Process(target=f, args=(data_worker_07, 7))
    p8 = Process(target=f, args=(data_worker_08, 8))
    # p9 = Process(target=f, args=(data_worker_09, 9))
    # p10 = Process(target=f, args=(data_worker_10, 10))
    p11 = Process(target=f, args=(data_worker_11, 11))
    p12 = Process(target=f, args=(data_worker_12, 12))

    # p5.start()
    # p6.start()
    p7.start()
    p8.start()
    # p9.start()
    # p10.start()
    p11.start()
    p12.start()

    p8.join()
    p7.join()
    # p9.join()
    p11.join()
    p12.join()
    # p6.join()
    # p10.join()
    # p5.join()

    etp = time.time()
    print(f'Ended at {time.ctime()}')
    print(f'Total time using mp: {etp-stp} seconds.\n')

