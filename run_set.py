import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime

st = datetime.now()
for itraj in range(10000):
    print(itraj,  (datetime.now() - st).seconds)
    os.system("python3 numerics/integration/integrate.py --itraj {} --pdt 1 --dt 1e-4 --total_time 8.001".format(itraj))
    print(itraj,  (datetime.now() - st).seconds, "flip")
    os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1  --pdt 1 --dt 1e-4 --total_time 8.001".format(itraj))
