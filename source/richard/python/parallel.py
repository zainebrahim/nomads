import numpy as np
import time
import timeit
import multiprocessing as mp

def finalize(p, parallel_time):
    total_time = parallel_time / (1 - p)
    non_parallel = total_time*p
    time.sleep(non_parallel)

    def task_nonparallel(data, sleep_time):
    for i in range(len(data)):
        time.sleep(sleep_time)
        data[i] += 42
    return

def split(data, n):
    if n > len(data):
        n = len(data)
    avg = len(data) / float(n)
    out = []
    last = 0.0

    while last < len(data):
        out.append(data[int(last):int(last + avg)])
        last += avg
    return out

def task_parallel(data, sleep_time, num_processors):
    num_workers = mp.cpu_count() - 1
    if (num_workers > num_processors):
        num_workers = num_processors
    jobs = []
    data = split(data, num_workers)
    for worker in range(len(data)):
        sub_data = data[worker]
        p = mp.Process(target = task_nonparallel, args = (sub_data, sleep_time))
        try:
            p.start()
        except:
            p.terminate()
            raise
        jobs.append(p)
    # make sure all jobs end
    for job in jobs:
        job.join()
        job.terminate()
    return
