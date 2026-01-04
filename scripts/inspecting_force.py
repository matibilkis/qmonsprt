import os
import multiprocessing as mp
from numerics.utilities.misc_force import *
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

#cores =  mp.cpu_count()
cores = 8

def simu(itraj):
    st = datetime.now()
    os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} ".format(itraj+k))
    os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} --flip_params 1".format(itraj+k))
    print(itraj, "holi - 8 cores //", (datetime.now() - st).seconds, datetime.now())

with mp.Pool(cores) as p:
    p.map(simu, list(range(8)))

itraj = 10
os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} ".format(itraj))
os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} --flip_params 1".format(itraj))



h0 = gamma0, omega0, n0, eta0, kappa0, b0 = 500.0, 100.0, 1.0, 1, 10.0, 0.0
h1 = gamma1, omega1, n1, eta1, kappa1, b1 = 500, 100.0, 1.0, 1, 10.0, 40
params = [h1,h0]
omega_pro = (omega0 + omega1)/2
period = (2*np.pi/omega_pro)
dt = period/100
total_time = 1000*period
times = np.arange(0,total_time+dt,dt)
indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
timind = [times[ind] for ind in indis]

pp = get_path_config(str(params)+"/", total_time = total_time, dt = dt, itraj=itraj)
pp0 = get_path_config(str([h0,h1])+"/", total_time = total_time, dt = dt, itraj=itraj)

liks = np.load(pp+"logliks.npy")
liks0 = np.load(pp0+"logliks.npy")
plt.plot(timind,liks[:,1]-liks[:,0])
plt.plot(timind,liks0[:,0]-liks0[:,1])
