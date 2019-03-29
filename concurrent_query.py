import re
import pandas as pd
import pprint
import psycopg2
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
    


bounds = pd.date_range('2017-05-01 00:00:00', '2017-10-01 00:00:00', freq='1W-MON')
subs = list(zip(bounds[:-1], bounds[1:]))
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
for i, query in queries.items():
    # dataframes[i] = pd.read_sql_query(query, conn)
    cursor.execute(query)

end = time.time()
print(f'Ended at {time.ctime()}, total time {end-start} seconds.')
