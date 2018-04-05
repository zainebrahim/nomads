import numpy as np
from timeit import default_timer as timer

def test(mag):
    start = timer()
    print("##### Test #####")
    iterations = 10**mag
    for i in range(iterations):
        X = np.arange(1000)
    print("Created NumPy Array " + str(iterations) + " times")
    end = timer()
    print("Time = " + str(end - start) + " seconds")
    return 0

def run_python():
    print("Running from Python:")
    for i in range(3, 8):
        test(i)

#run_python()
