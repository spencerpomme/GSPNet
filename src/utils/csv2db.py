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
import numpy as np
from threading import Thread
import time
import os
from multiprocessing import cpu_count


# Numpy dtype -> PostgreSQL type convertion dictionary
def project(file: str, cols: list):
    '''
    Read a csv file to a numpy array
    Args:
        file: .csv file path string
        cols: needed columns
    '''
    raise NotImplementedError

# Currently this module is under development.


def analysis(file: str):
    '''
    Read a .csv file and proceed some basic analysis.
    Args:
        file: a string file path
    Returns:
        Information about the .csv file
    '''
    tb = pd.read_csv(file)


if __name__ == '__main__':

    '''
    CSV raw data columns:
    {
        VendorID,
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        RatecodeID,
        store_and_fwd_flag,
        PULocationID,
        DOLocationID,
        payment_type,
        fare_amount,
        extra,
        mta_tax,
        tip_amount,
        tolls_amount,
        improvement_surcharge,
        total_amount
    }

    database table columns: (in SQL)
    {
        tripid bigint,
        tpep_pickup_datetime timestamp without time zone,
        tpep_dropoff_datetime timestamp without time zone,
        pulocationid integer,
        dolocationid integer,
        trip_distance double precision,
        passenger_count integer,
        total_amount double precision,
        trip_time interval,
        trip_avg_speed double precision,
        trip_time_sec double precision
    }
    '''

    pattern = 'F:\\NY_taxi\\2018\\*.csv'
    dirs = glob.glob(pattern)
    print(dirs)
