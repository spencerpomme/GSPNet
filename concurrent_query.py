
import re
import numpy as np
import pandas as pd
import pprint
import psycopg2
from threading import Thread
from multiprocessing import Process, cpu_count
from queue import Queue
import time
from tqdm import tqdm


def _construct_sql(stp:str, etp:str):
        '''
        A private helper function to construct sql query, called another
        helper function _construct_split.

        Args:
            stp: datetime string, starting time point of a concurrent unit
            etp: datatime string, end time point of a concurrent unit

        Returns:
            sql: a constructed query string
        '''
        pattern = re.compile(
            '^([0-9]{4})-([0-1][0-9])-([0-3][0-9])\s([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$'
        )
        assert pattern.match(stp) and pattern.match(etp)

        return (f"select tripid,tpep_pickup_datetime,tpep_dropoff_datetime,pulocationid,dolocationid from cleaned_small_yellow_2017_full "
                f"where tpep_pickup_datetime >= '{stp}' and tpep_dropoff_datetime < '{etp}';")


def safe_weekly_divide(stp:str, etp:str):
    '''
    Return a safely rounded split of timeslices decided by "1W-MON".

    Args:
        stp: datetime string, starting time point of a concurrent unit
        etp: datatime string, end time point of a concurrent unit

    Returns:
        subs: a list of (Timestamp, Timestamp) pairs
    '''    
    bounds = pd.date_range(stp, etp, freq='1W-MON')
    print(bounds[0], bounds[-1])
    head_round = tail_round = None
    if bounds[0] != stp:
        head_round = (pd.Timestamp(stp), pd.Timestamp(bounds[0]))
    if bounds[-1] != etp:
        tail_round = (pd.Timestamp(bounds[-1]), pd.Timestamp(etp))
    print('\n\n\n')
    subs = [head_round] + list(zip(bounds[:-1], bounds[1:])) + [tail_round]
    
    return subs


def concurrent_read(id:int, df_pool:dict, query:str):
    
    host = 'localhost'
    dbname = 'taxi'
    user = 'postgres'
    conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
    # cursor = conn.cursor()
    # cursor.execute(query)
    df_pool[id] = pd.read_sql_query(query, conn)
    

# sub intervals:
subs = safe_weekly_divide('2017-01-01 00:00:00', '2018-01-01 00:00:00')

# containers to hold generated queries, new thread, started thread and generated dataframes:
queries = {}
thread_pool = Queue()
started_threads = []
dataframes = {} # global variables...

# create sub interval queries
for i, sub in enumerate(subs):
    stp, etp = list(map(str, sub))
    queries[i] = _construct_sql(stp, etp)


################################ START #################################
# number of queries in total
q_size = len(queries)
print(f'Total query number to be executed: {q_size}')

start = time.time()
print(f'Started at {time.ctime()}')



for i, query in queries.items():
    t = Thread(target=concurrent_read, args=(i, dataframes, query))
    thread_pool.put(t)

# progress bar util
bar = tqdm(total=q_size, ascii=True)

while not thread_pool.empty():
    for i in range(min(cpu_count() * 2 + 1, q_size)):
        t = thread_pool.get()
        started_threads.append(t)
    
    # number of queries left not executed:
    q_size -= (cpu_count() * 2 + 1)
    bar.update(cpu_count() * 2 + 1)
    # print(f'q_size: {q_size} started_threads: {len(started_threads)}\n')
    
    for t in started_threads:
        # print(f'Starting thread {t}')
        t.start()

    for t in started_threads:
        t.join()
        # print(f'Ended {t}')

    # print(f'Buffer started_threads flushed!')
    started_threads = []

end = time.time()
print(f'Ended at {time.ctime()}, total time {end-start} seconds.')

print(dataframes[0].shape, dataframes[3].shape)