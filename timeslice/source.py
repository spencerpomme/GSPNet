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
        try:
            # read csv file and format time related colums
            self.table = pd.read_csv(self.file)
            self.format_time_cols()

        except Exception as e:
            print(e)
            print(f'Provided file {file} is not a valid file source.')

        self.shape = self.table.shape
        self.dtypes = self.table.dtypes


    def __repr__(self):
        '''
         Representation method for csv source.
        '''
        shape = self.table.shape

        return f'CSV file source: {self.file} | Shape:  {shape[0], shape[1]}'


    
class DatabaseSource(Source):
    '''
    Data source class when the data source is a relational database.
    
    It is effectively using a relational database as source while still using
    pandas or dask to do the heavy lifting, without creating wheels to manually
    do the works.
    '''

    def __init__(self, host:str, dbname:str, user:str):
        '''
        Init method for database source.

        Every instance of this class should contain detailed information about
        the connected database source, including:
            (1) shape: rows and columns
            (2) 

        Args:
            host: database address
            dbname: database name
            user: (admin) user of the database
        '''
        # self.conn = psycopg2.connect(f'host=localhost dbname=taxi user=postgres')
        self.conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
        self.cur = conn.cursor()
        self.table = pd.read_sql()


    def __repr__(self, verbose=False):
        '''
        Representation method for database source.
        '''
        raise NotImplementedError

