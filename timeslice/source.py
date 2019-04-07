'''
Copyright <2019> <COPYRIGHT Pingcheng Zhang>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

##============================================================================##

'''
import numpy as np
import pandas as pd
import sqlalchemy
import psycopg2
import time
import re

from threading import Thread
from queue import Queue
from multiprocessing import cpu_count
from tqdm import tqdm


# 3 classes in this module
class Source:
    '''
    Base class of data source.

    '''

    def __init__(self, table):
        '''
        Base init method, data should be directely convertible to tensors.

        Args:
            table: a pandas dataframe object
        '''
        self.table = table

    def __repr__(self):
        '''
        Base representation method.
        '''
        return f'Source: {self.__class__} of type {type(data)}'

    def describe(self):
        '''
        Wrapper of table describe method.
        '''
        self.table.describe()

    def format_time_cols(self):
        '''
        Transform time type colums in table to datatime type from string type.
        '''
        self.table['tpep_pickup_datetime'] = pd.to_datetime(
            self.table['tpep_pickup_datetime'])
        self.table['tpep_dropoff_datetime'] = pd.to_datetime(
            self.table['tpep_dropoff_datetime'])


#========================#
# Data source: .csv file #
#========================#
class CSVSource(Source):
    '''
    Data source class when the data source is a .csv file.

    The initialization of this instances of this class is postponed
    to there user, the Worker class. This somehow twisted design
    is for the good of utilizing parallel comptuing in tensor generation.
    '''

    def __init__(self, file: list):
        '''
        Init method for csv source.

        Args:
            file: Directory string of given .csv file
        '''
        assert type(file) == str
        assert file[-4:] == '.csv'
        self.file = file
        self.table = None

        self.shape = None
        self.dtypes = None

    def __repr__(self):
        '''
         Representation method for csv source.
        '''
        if isinstance(self.table, str):
            return self.table

        shape = self.table.shape

        return f'CSV file source: {self.file} | Shape:  {shape[0], shape[1]}'

    def __len__(self):
        '''
        Returns the total number of rows ot self.table.
        '''
        try:
            total_rows = self.table.shape[0]
        except Exception as e:
            print(e)
            print('The actual data is not loaded yet. It is suggested to call "load" method at\
                   worker side.')
            total_rows = 0

        return total_rows

    def load(self):
        '''
        Actually read in the .csv file. It better not to do the IO at
        initialization time to get more flexibility.
        '''
        try:
             # read csv file and format time related colums
            temp = pd.read_csv(self.file)
            self.table = temp.loc[:, ['tripid',
                                      'tpep_pickup_datetime',
                                      'tpep_dropoff_datetime',
                                      'pulocationid',
                                      'dolocationid']]

            self.shape = self.table.shape
            self.format_time_cols()
            self.dtypes = self.table.dtypes

        except Exception as e:
            print(e)
            print(f'Provided file {file} is not a valid file source.')

    def head(self, first_n_rows: int):
        '''
        Wrapper of table head method.

        Args:
            first_n_rows: first n rows of the table
        
        Returns:
            head object
        '''
        return self.table.head(first_n_rows)


#==================================#
# Data source: relational database #
#==================================#
class DatabaseSource(Source):
    '''
    Data source class when the data source is a relational database.
    
    It is effectively using a relational database as source while still using
    pandas or dask to do the heavy lifting, without creating wheels to manually
    do the works.
    '''

    def __init__(self, tbname, big_bound=('2017-01-01 00:00:00', '2018-01-01 00:00:00'),
                 host: str = 'localhost', dbname: str = 'taxi',
                 user: str = 'postgres', concurrent: bool = True):
        '''
        Init method for database source.

        Args:
            tbname: name of the table
            host: database address
            dbname: database name
            user: (admin) user of the database
            concurrent: decide whether to divide a big table into small one and
                        load concurrently or not.
        
        Example:
            host="localhost" dbname="taxi" user="postgres"
        '''
        self.tbname = tbname
        self.host = host
        self.dbname = dbname
        self.user = user
        self.concurrent = concurrent

        # connect to the database
        self.conn = psycopg2.connect(
            f'host={self.host} dbname={self.dbname} user={self.user}')
        self.cur = self.conn.cursor()

        # time rule attributes for concurrent load strategy:
        self.big_bound = big_bound

        # datetime format checking pattern
        self.pattern = re.compile(
            '^([0-9]{4})-([0-1][0-9])-([0-3][0-9])\s([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$'
        )

        # unified container: table_pool
        self.table_pool = {}
        self.queries = self._concurrent_split()

        # sub ranges of time interval
        # for example, an entire year can be divided into 4 quaters.
        self.sub_ranges = None

        # data source info
        # self.total_rows = self._sumrows()

    def __repr__(self, verbose=False):
        '''
        Representation method for database source.

        Args:
            verbose: if true, then more information will be displayed

        Returns:
            Strings about the Source instance
        '''
        info = f'DatabaseSource\n\
                  host: {self.host} | databse: {self.dbname} | table: {self.table}\n'

        if verbose:
            detail = f'Total rows: {self._sumrows()} | Columns: {self._get_columns()}'
            info = info + detail

        return info

    def __len__(self):
        '''
        Returns the total number of rows ot self.table.
        '''
        # sql = "select count(*) from cleaned_small_yellow_2017;"
        # temp_df = pd.read_sql_query(sql, self.conn)

        return self.total_rows

    def load(self):
        '''
        Actually read in the .csv file. It better not to do the IO at
        initialization time to get more flexibility.
        '''
        if self.concurrent:

            # decide threads that can run concurrently at a time
            cpus = cpu_count()

            # maximum efficient concurrent thread number
            t_unit = cpus * 2 + 1

            # containers to hold generated queries, new thread, started thread and generated dataframes:
            queries = self._concurrent_split()
            thread_pool = Queue()
            thread_buffer = []

            # an important variable
            q_size = len(queries)

            print(f'Database data load started at {time.ctime()}')
            start = time.time()

            for i, query in self.queries.items():
                t = Thread(target=DatabaseSource._concurrent_read,
                           args=(i, self.table_pool, query))
                thread_pool.put(t)

            progress = tqdm(total=q_size, ascii=True)

            while not thread_pool.empty():
                for i in range(min(t_unit, q_size)):
                    t = thread_pool.get()
                    thread_buffer.append(t)

                # number of queries left not executed, update progress bar
                done_threads = min(t_unit, q_size)
                q_size -= done_threads
                progress.update(done_threads)

                # start threads
                for t in thread_buffer:
                    t.start()

                # join finished threads
                for t in thread_buffer:
                    t.join()

                # clear finished threads from thread buffer
                thread_buffer = []

            # close progress bar
            progress.close()
            end = time.time()
            print(
                f'Ended at {time.ctime()}, total time {end - start:.2f} seconds.')

        else:
            sql = f"select tripid, tpep_pickup_datetime, tpep_dropoff_datetime, pulocationid, dolocationid \
                    from {self.tbname} \
                    where tpep_pickup_datetime >= '{self.big_bound[0]}' and \
                          tpep_dropoff_datetime < '{self.big_bound[1]}';"

            print(f'Database data load started at {time.ctime()}')
            bare_start = time.time()

            self.table = pd.read_sql_query(sql, self.conn)
            self.total_rows = self.table.shape[0]
            self.table_pool[0] = self.table

            bare_end = time.time()
            print(
                f'Ended at {time.ctime()}, total time {bare_end - bare_start:.2f} seconds.')

    # helper functions

    @staticmethod
    def _concurrent_read(id: int, df_pool: dict, query: str):
        '''
        Create a new connector to database, each for one thread.

        Args:
            id: unique marker for a connector
            df_pool: dataframe pool, storing loaded dataframes
            query: sql string
        '''
        host = 'localhost'
        dbname = 'taxi'
        user = 'postgres'

        conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
        # cursor = conn.cursor()
        # cursor.execute(query)
        df_pool[id] = pd.read_sql_query(query, conn)

    def _concurrent_split(self, granularity: str = '1W-MON'):
        '''
        Saperate a query that returns a potentially large table into several
        sub queries and then do them concurrently. 
        
        For example, when set granularity to 1 week, then if the big_bound is
        an period of 1 month, then the query will be divided into 30 sub queries,
        each returns a data of 1 day.

        Args:
            granularity: minimum time unit
            This granularity is TESTED TO BE MOST EFFICIENT with current hardware.
            Don't change unless necessary!

        Returns:
            queries: a list containing sql string initialized with sub time bounds
        '''
        subs = self._process_granularity(
            self.big_bound[0], self.big_bound[1], freq=granularity)
        self.sub_ranges = subs
        queries = {}

        # create sub interval queries
        for i, sub in enumerate(subs):
            stp, etp = list(map(str, sub))
            queries[i] = self._construct_sub_sql(stp, etp)

        return queries

    def _construct_sub_sql(self, stp: str, etp: str):
        '''
        A private helper function to construct sql query, called another
        helper function _construct_split.

        Args:
            stp: datetime string, starting time point of a concurrent unit
            etp: datatime string, end time point of a concurrent unit

        Returns:
            sql: a constructed query string
        '''
        pattern = self.pattern
        assert pattern.match(stp) and pattern.match(etp)

        return (f"select tripid,tpep_pickup_datetime,tpep_dropoff_datetime,pulocationid,dolocationid from {self.tbname} "
                f"where tpep_pickup_datetime >= '{stp}' and tpep_dropoff_datetime < '{etp}';")

    def _process_granularity(self, stp: str, etp: str, freq: str):
        '''
        Function to divide a table according freq (the concurrent time unit).

        The main problem solved by this function is alignment and round up of
        weeks when dividing a month or year into weekly-sub-tables. The arguments
        of this function is kept the same as pandas.date_range function.

        Args:
            stp: datetime string, starting time point of a concurrent unit
            etp: datatime string, end time point of a concurrent unit
            freq: frequency, time interval unit of the splice operation
                  The supported frequency units are:
                  
                    Alias	    Description
                    B	        business day frequency
                    C	        custom business day frequency
                    D	        calendar day frequency
                    W	        weekly frequency
                    M	        month end frequency
                    H	        hourly frequency
                    
        The stp and etp must of pattern "yyyy-mm-dd hh:mm:ss", otherwise
        exception will be raised.

        Returns:
            subs: DataTimeIndex object, i.e. return type of pandas.date_range

        Raises:
            AssertionError
        '''
        assert freq in ['B', 'C', 'D', 'W', '1W-MON',
                        'M'], 'Only supported frequencies allowed.'
        bounds = pd.date_range(stp, etp, freq=freq)
        # print(bounds[0], bounds[-1])

        head_round = tail_round = None

        if bounds[0] != stp:
            head_round = (pd.Timestamp(stp), pd.Timestamp(bounds[0]))
        if bounds[-1] != etp:
            tail_round = (pd.Timestamp(bounds[-1]), pd.Timestamp(etp))

        # rounded time interval
        subs = [head_round] + list(zip(bounds[:-1], bounds[1:])) + [tail_round]

        # return rounded time intervals
        return subs

    def subset(self, stp: str, etp: str):
        '''
        Make a subset from the table according to time.
        This function is a part of the parallel IO of tensor generation.

        Args:
            stp: starting time point of subset
            etp: ending time point of subset
            both these two strings are of format "YYYY-MM-DD hh:mm:ss"

        Return:
            sub_table: A list of time intervals tuples,each item is a tuple of
            two interval(i.e., pandas.core.indexes.datetimes.DatetimeIndex object)
            For example, a possible return could be:

            [(2017-01-01 00:00:00, 2017-01-01 00:10:00),
                               ......
             (2017-01-01 01:10:00, 2017-01-01 01:20:00)]
        '''
        # regular expression that guarantee stp and etp are of right format
        pattern = self.pattern

        if pattern.match(self.stp) and pattern.match(self.etp):
            sql = f"""
                    select tripid, tpep_pickup_datetime, tpep_dropoff_datetime, pulocationid, dolocationid from 
                    {self.tbname} where
                    tpep_dropoff_datetime > '{stp}' or
                    tpep_pickup_datetime <= '{etp}';
                    """
        else:
            raise Exception('Provided time bound is of invalid format.')

        sub_table = pd.read_sql_query(sql, self.conn)

        return sub_table

    def _weekly_parallel(self):
        '''
        A convenient wrapper of concurrent load using a week as deviding unit.

        Args:
            table: A table connector

        Returns:
            table_pool: A pool of sub tables
        '''
        return self._concurrent_split('1W-MON')

    def _monthly_parallel(self):
        '''
        A convenient wrapper of concurrent load using a day as deviding unit.

        Args:
            table: A table connector

        Returns:
            table_pool: A pool of sub tables
        '''
        return self._concurrent_split('1M')

    def _sumrows(self):
        '''
        Sum up sub table rows in self.table_pool.
        '''
        raise NotImplementedError

    def _get_columns(self):
        '''
        Return the column names and types of self.table.

        A sure (but maybe slow) way would be directly construct
        a query to get this info

        Returns:
            column_info: a dictionary
        '''
        # construct a query to retrieve column meta data
        sql = f"""
            select column_name as name, data_type as dtype
            from information_schema.columns
            where table_schema = 'public' and 
            table_name = '{self.tbname}';"""

        info = pd.read_sql_query(sql, self.conn)

        # create dictionary {'column_name': 'data_type'}
        pair = list(zip(info['name'].to_list(), info['dtype'].to_list()))
        column_info = {name: dtype for (name, dtype) in pair}

        return column_info
