import re
import numpy as np
import pandas as pd
import pprint
import psycopg2
import time
from tqdm import tqdm


def _construct_sql(stp: str, etp: str):
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


def safe_weekly_divide(stp: str, etp: str):
    '''
    Return a safely rounded split of timeslices decided by "1D".

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


subs = safe_weekly_divide('2017-01-01 00:00:00', '2018-01-01 00:00:00')
queries = {}

host = 'localhost'
dbname = 'taxi'
user = 'postgres'

conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
cursor = conn.cursor()

# create sub interval queries
for i, sub in enumerate(subs):
    stp, etp = list(map(str, sub))
    queries[i] = _construct_sql(stp, etp)

dataframes = {}


start = time.time()
print(f'Started at {time.ctime()}')
for i, query in tqdm(queries.items(), ascii=True):
    dataframes[i] = pd.read_sql_query(query, conn)

end = time.time()
print(f'Ended at {time.ctime()}, total time {end-start} seconds.')
print(dataframes.keys())
