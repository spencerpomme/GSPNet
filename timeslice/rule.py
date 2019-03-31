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
import re
import numpy as np
import pandas as pd

class TimeSlice:
    '''
    Create an array of time intervals.

    '''
    pattern = re.compile(
            '^([0-9]{4})-([0-1][0-9])-([0-3][0-9])\s([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$'
        )
    def __init__(self, stp:str, etp:str, freq='10min'):
        '''
        Init method for TimeSlice object

        Args:
            stp: string, starting time point, first left bound
            etp: string, ending time point, last right bound
            freq: frequency, time interval unit of the splice operation
                  The supported frequency units are:
                  
                    Alias	    Description
                    B	        business day frequency
                    C	        custom business day frequency
                    D	        calendar day frequency
                    W	        weekly frequency
                    M	        month end frequency
                    SM	        semi-month end frequency (15th and end of month)
                    BM	        business month end frequency
                    CBM	        custom business month end frequency
                    MS	        month start frequency
                    SMS	        semi-month start frequency (1st and 15th)
                    BMS	        business month start frequency
                    CBMS	    custom business month start frequency
                    Q	        quarter end frequency
                    BQ	        business quarter end frequency
                    QS	        quarter start frequency
                    BQS	        business quarter start frequency
                    A, Y	    year end frequency
                    BA, BY	    business year end frequency
                    AS, YS	    year start frequency
                    BAS, BYS	business year start frequency
                    BH	        business hour frequency
                    H	        hourly frequency
                    T, min	    minutely frequency
                    S	        secondly frequency
                    L, ms	    milliseconds
                    U, us	    microseconds
                    N	        nanoseconds
        The stp and etp must of pattern "yyyy-mm-dd hh:mm:ss", otherwise
         exception will be raised.
        '''
        self.stp = stp
        self.etp = etp
        self.freq = freq
        self.fragments = None

        # initialize rule
        self.timesplit()


    def __repr__(self):
        '''
        Representation method for TimeSlice object
        '''
        return f'TimeSlice {self.stp} -- {self.etp} Interval: {self.freq}'
        

    def timesplit(self):
        '''
        Create a DatetimeIndx interval.
        
        Return:
            A list of time intervals tuples,each item is a tuple of two
            interval(i.e., pandas.core.indexes.datetimes.DatetimeIndex object)
            For example, a possible return could be:

            [(2017-01-01 00:00:00, 2017-01-01 00:10:00),
             (2017-01-01 00:10:00, 2017-01-01 00:20:00)]
        '''
        pattern = TimeSlice.pattern

        if pattern.match(self.stp) and pattern.match(self.etp):

            time_bounds = pd.date_range(self.stp, self.etp, freq=self.freq)
            self.fragments = list(zip(time_bounds[:-1], time_bounds[1:]))

        else:
            raise Exception('Provided time bound is of invalid format.')


    def get_slices(self):
        '''
        A convenient wrapper to return time slices.
        '''
        return self.fragments