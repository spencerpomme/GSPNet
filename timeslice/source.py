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
        self.table = 'Please call instance.load() to initialize'

        self.shape = ''
        self.dtypes = ''


    def __repr__(self):
        '''
         Representation method for csv source.
        '''
        if isinstance(self.table, str):
            return self.table

        shape = self.table.shape

        return f'CSV file source: {self.file} | Shape:  {shape[0], shape[1]}'


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

    def __init__(self, host:str, dbname:str, user:str, tbname:str):
        '''
        Init method for database source.

        Args:
            host: database address
            dbname: database name
            user: (admin) user of the database
            tbname: name of the table
        '''
        # self.conn = psycopg2.connect(f'host=localhost dbname=taxi user=postgres')

        self.host = host
        self.dbname = dbname
        self.user = user
        self.tbname = tbname

        # connect to the database
        self.conn = psycopg2.connect(f'host={self.host} dbname={self.dbname} user={self.user}')
        self.cur = self.conn.cursor()

        # initialize table
        sql = f'select * from {self.tbname}'
        self.table = pd.read_sql_query(sql, self.conn)


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
        sql = "select count(*) from cleaned_small_yellow_2017;"
        temp_df = pd.read_sql_query(sql, self.conn)
        
        return temp_df.iloc[0][0]


    def get_columns(self):
        '''
        Return the column names and types of self.table.

        Returns:
            column_info: a dictionary
        '''
        raise NotImplementedError


    def subset(self, start:str, end:str):
        '''
        Make a subset from the table according to time.
        This function is a part of the parallel IO of tensor generation.

        Args:
            start: starting time point of subset
            end: ending time point of subset

        Return:
            sub_slice:
            A list of time intervals tuples,each item is a tuple of two
            interval(i.e., pandas.core.indexes.datetimes.DatetimeIndex object)
            For example, a possible return could be:

            [(2017-01-01 00:00:00, 2017-01-01 00:10:00),
                               ... ...
             (2017-01-01 01:10:00, 2017-01-01 01:20:00)]
        '''
        sql = "select * from cleaned_small_yellow_2017 where ;"
        sub_table = pd.read_sql_query(sql, self.conn)
        

