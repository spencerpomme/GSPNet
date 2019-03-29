
import re
import numpy as np
import pandas as pd
import pprint
import psycopg2
from threading import Thread
from multiprocessing import Process
from queue import Queue
import time

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
    


def concurrent_read(id:int, df_pool:dict, query:str):
    print(f'id {id} thread executing...')
    host = 'localhost'
    dbname = 'taxi'
    user = 'postgres'

    conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
    cursor = conn.cursor()
    cursor.execute(query)
    
    
bounds = pd.date_range('2017-05-01 00:00:00', '2017-10-01 00:00:00', freq='1D')
subs = list(zip(bounds[:-1], bounds[1:]))

queries = {}
thread_pool = Queue()
df_pool = Queue()

# create sub interval queries
for i, sub in enumerate(subs):
    stp, etp = list(map(str, sub))
    queries[i] = _construct_sql(stp, etp)

start = time.time()
print(f'Started at {time.ctime()}')
for i, query in queries.items():
    t = Thread(target=concurrent_read, args=(i, df_pool, query))
    thread_pool.put(t)
    print(f'starting {t}')
    t.start()


while not thread_pool.empty():
    p = thread_pool.get()
    p.join()
    print(f'{p} is finished!')

end = time.time()
print(f'Ended at {time.ctime()}, total time {end-start} seconds.')