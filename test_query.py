import psycopg2
import time
from threading import Thread

host = 'localhost'
dbname = 'taxi'
user = 'postgres'


def f(sql, i):
    print(f'Thread {i} started!')
    conn = psycopg2.connect(f'host={host} dbname={dbname} user={user}')
    cursor = conn.cursor()
    cursor.execute(sql)

sql = 'select * from cleaned_small_yellow_2017_05;'

t1 = Thread(target=f, args=(sql, 1))
t2 = Thread(target=f, args=(sql, 2))
t3 = Thread(target=f, args=(sql, 3))
t4 = Thread(target=f, args=(sql, 4))



start = time.time()
print(f'Started at {time.ctime()}')
t1.start()
t1.join()
t2.start()
t2.join()
t3.start()
t3.join()
t4.start()
t4.join()
# t1.join()
# t2.join()
# t3.join()
# t4.join()
end = time.time()
print(f'Ended at {time.ctime()}, total time {end-start} seconds.')