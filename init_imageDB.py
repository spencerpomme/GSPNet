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


A module that maintains the image database.
This module is used as a part of GSPNet project.
'''

import psycopg2 as pg
import pandas as pd

conn = pg.connect(f'host=localhost dbname=taxi user=postgres')
curr = conn.cursor()
curr.execute('''
                CREATE TABLE IF NOT EXISTS image_dirs (
                    id varchar(32),
                    time_interval  varchar(64),
                    store_loc varchar(128),
                    save_time timestamp,
                    batch_seq varchar(16)
                )
                WITH (
                OIDS = FALSE
                )
                TABLESPACE pg_default;
                ALTER TABLE public.image_dirs
                    OWNER to postgres;
            ''')
conn.commit()


