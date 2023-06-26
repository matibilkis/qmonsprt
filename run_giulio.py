import os
import multiprocessing as mp
from numerics.utilities.misc import *
from datetime import datetime

global step
cores = step =  19
def simu(itraj):
    st = datetime.now()
    for k in range(step):
        os.system("python3 numerics/integration/integrate.py --itraj {} --pdt 1 --dt 1e-4 --total_time 8.".format(itraj+k))
        os.system("python3 numerics/integration/integrate.py --itraj {} --flip_params 1  --pdt 1 --dt 1e-4 --total_time 8.".format(itraj+k))
        print(itraj, cores, (datetime.now() - st).seconds)

itrajs = np.array(list(range(0,20000,step)))
with mp.Pool(cores) as p:
    p.map(simu, itrajs)

np.array(list(range(0,20000,step)))


global step
cores = step =  19
def simu(itraj):
    os.makedirs("exp/{}/".format(itraj),exist_ok=True)
    W = np.random.randn(100000,2)
    np.save("exp/{}/".format(itraj)+"w.npy",W)

itrajs = np.array(list(range(19)))
with mp.Pool(cores) as p:
    p.map(simu, itrajs)


noises = np.stack([np.load("exp/{}/w.npy".format(itraj)) for itraj in range(19)])
[noises[0] - noises[k] for k in range(19)]
