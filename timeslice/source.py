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
import re

from dask import dataframe as dd


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


    def head(self, first_n_rows: int):
        '''
        Wrapper of table head method.

        Args:
            first_n_rows: first n rows of the table
        
        Returns:
            head object
        '''
        return self.table.head(first_n_rows)


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
            self.table = pd.read_csv(self.file)
            self.shape = self.table.shape
            self.format_time_cols()
            self.dtypes = self.table.dtypes

        except Exception as e:
            print(e)
            print(f'Provided file {file} is not a valid file source.')


    
class DatabaseSource(Source):
    '''
    Data source class when the data source is a relational database.
    
    It is effectively using a relational database as source while still using
    pandas or dask to do the heavy lifting, without creating wheels to manually
    do the works.
    '''

    def __init__(self, tbname, host:str='localhost', dbname:str='taxi',
                 user:str='postgres', concurrent:bool=True):
        '''
        Init method for database source.

        Args:
            tbname: name of the table
            host: database address
            dbname: database name
            user: (admin) user of the database
            concurrent: decide whether to divide a big table into small one and
                        load concurrently or not.
        '''
        # host=localhost dbname=taxi user=postgres

        self.host = host
        self.dbname = dbname
        self.user = user
        self.tbname = tbname
        self.concurrent = concurrent

        # connect to the database
        self.conn = psycopg2.connect(f'host={self.host} dbname={self.dbname} user={self.user}')
        self.cur = self.conn.cursor()

        # unified container: table_pool
        self.table_pool = []
        
        self.total_rows = self.__sumrows(self.table_pool)


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
            detail = f'Total rows: {self.count_rows()} | Columns: {self.get_columns()}'
            info = info + detail

        return info


    def __len__(self):
        '''
        Returns the total number of rows ot self.table.
        '''
        # sql = "select count(*) from cleaned_small_yellow_2017;"
        # temp_df = pd.read_sql_query(sql, self.conn)
        
        return self.total_rows

    
    def __sumrows(self):
        '''
        Sum up sub table rows in self.table_pool.
        '''
        raise NotImplementedError


    def load(self):
        '''
        Actually read in the .csv file. It better not to do the IO at
        initialization time to get more flexibility.
        '''
        sql = f"select tripid, tpep_pickup_datetime, tpep_dropoff_datetime, pulocationid, dolocationid \
                from cleaned_small_yellow_2017_full \
                where tpep_pickup_datetime >= {} and \
                      tpep_dropoff_datetime < {};"
        if self.concurrent:
            self.tbname
        self.table = pd.read_sql_query(sql, self.conn)
        self.total_rows = self.table.shape[0]


    def get_columns(self):
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


    def subset(self, stp:str, etp:str):
        '''
        Make a subset from the table according to time.
        This function is a part of the parallel IO of tensor generation.

        Args:
            stp: starting time point of subset
            etp: ending time point of subset
            both these two strings are of format "YYYY-MM-DD hh:mm:ss"

        Return:
            sub_slice: A list of time intervals tuples,each item is a tuple of
            two interval(i.e., pandas.core.indexes.datetimes.DatetimeIndex object)
            For example, a possible return could be:

            [(2017-01-01 00:00:00, 2017-01-01 00:10:00),
                               ......
             (2017-01-01 01:10:00, 2017-01-01 01:20:00)]
        '''
        # regular expression that guarantee stp and etp are of right format
        
        pattern = re.compile('^([0-9]{4})-([0-1][0-9])-([0-3][0-9])\s([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$')
        
        if pattern.match(self.stp) and pattern.match(self.etp):
            sql = f"""
                select * from 
                cleaned_small_yellow_2017 where
                tpep_dropoff_datetime > {stp} or
                tpep_pickup_datetime <= {etp};"""
        else:
            raise Exception('Provided time bound is of invalid format.')

        sub_table = pd.read_sql_query(sql, self.conn)

        return sub_table


    def weekly_parallel(self, table):
        '''
        A convenient wrapper of concurrent load using a week as deviding unit.

        Args:
            table: A table connector

        Returns:
            table_pool: A pool of sub tables
        '''
        raise NotImplementedError


    def daily_parallel(self, table):
        '''
        A convenient wrapper of concurrent load using a day as deviding unit.

        Args:
            table: A table connector

        Returns:
            table_pool: A pool of sub tables
        '''
        raise NotImplementedError
        

