'''
Copyright (c) <2018> <Pingcheng Zhang>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

A module to conduct IO missions in three different ways.
This module is used as a part of GSPNet project.
'''

import psycopg2
import glob
import pandas as pd
from threading import Thread
import time
import os
from multiprocessing import Queue, Process, cpu_count

class DirFileLoader:
    '''
    A util class to load csv files in a folder to a connected database.
    Paralleled IO can provide an average of 3X speed up.
    '''
    def __init__(self, pattern=os.getcwd()):
        '''
        Initialize DirFileLoader instance
        Params:
            pattern: path string pattern
        Attributes:
            dirs: a list of csv file directory strings
            dir_pool: a Queue object
            res_pool: a list used to collect results
        '''
        self.dirs = glob.glob(pattern)
        # print(self.dirs)
        self.dir_pool = Queue()
        self.res_pool = []
        self._init_pool()


    def _init_pool(self):
        '''
        Initialize thread/process pool for concurrent IO tasks.
        '''
        while not self.dir_pool.empty():
            self.dir_pool.get()
        for d in self.dirs:
            self.dir_pool.put(d)
        # print(f'Init complete:\n {list(self.dir_pool.queue)}')


    def sqstart(self):
        '''
        Sequential loading csv file in and convert to pd.DataFrame,
        serves as a base benchmark.
        Params:
            loc: path string, same thing as pattern param in __init__
        Return:
            fs: a list of pd.DataFrame objects
        '''
        print(f'========== Sequential Loading Start ==========')
        self._init_pool()
        s = time.time()
        self.task_io(0)
        t = time.time() - s
        print(f'Read time: {round(t//60)}min {round(t%60, 8)}sec.')


    def mpstart(self):
        '''
        Mulpti-processing IO task starter. 
        Delegate io functions wrapped by self.task_io wrapper to multiple processes.
        '''
        print(f'\n\n========== Parallel Loading Start ==========')
        self._init_pool()
        s = time.time()
        process_list = [Process(target=self.task_io, args=(i,))
             for i in range(cpu_count())
        ]
        for p in process_list:
            p.start()

        for p in process_list:
            if p.is_alive():
                p.join()
        print(f'========== Task end in {round(time.time() - s, 4)} sec ==========\n')


    def mcstart(self):
        '''
        Mulpti-thread IO task starter. 
        Delegate io functions wrapped by self.task_io wrapper to multiple threads.
        '''
        print(f'\n\n========== Multi-thread Loading Start ==========')
        self._init_pool()
        s = time.time()
        thread_list = [Thread(target=self.task_io, args=(i,))
            for i in range(len(self.dirs))
        ]
        for t in thread_list:
            t.start()

        for t in thread_list:
            if t.is_alive():
                t.join()
        print(f'========== Task end in {round(time.time() - s, 4)} sec ==========\n')


    def task_io(self, id: int):
        '''
        IO task wrapper.
        The task conducted is one of the __operations.
        Params:
            id: task number
        '''
        print(f'IO task[{id}] start')
        while not self.dir_pool.empty():
            try:
                csvfile = self.dir_pool.get(block=True, timeout=1)
                # io task:
                # tb = self.__readcsv(csvfile)
                # self.res_pool.append(tb)
                self.__csv2db(csvfile)
            except Exception as e:
                print(f'IO task[{id}] error: {e}')
        print(f'IO task[{id}] ended.')


    # Utility functions for one kind of IO opereation
    def __readcsv(self, file: str) -> pd.DataFrame:
        '''
        IO task: read in csv.
        Params:
            file: file directory get from queue
        Return:
            tb: a pandas DataFrame
        '''
        print(f'Reading {file}...')
        tb = pd.read_csv(file)
        print(f'{file} loaded.')
        return tb


    def __csv2db(self, file: str):
        '''
        Copy csv file into a database.
        Params:
            file:  file directory get from queue
        '''
        conn = psycopg2.connect(f'host=localhost dbname=taxi user=postgres')
        cur = conn.cursor()
        with open(file, 'r') as f:
            next(f)  # Skip the header row
            try:
                cur.copy_from(f, 'test_table', sep=',')
            except Exception as e:
                print(f'{e}')
        conn.commit()





if __name__ == '__main__':

    print(os.getcwd())
    loader = DirFileLoader('F:\\NY_taxi\\test\\*.csv')
    loader.mcstart()

    # res = seq_load('F:\\NY_taxi\\test\\*.csv')
    # print(map(type, res))
