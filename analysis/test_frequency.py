import os
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numerics.utilities.misc import *
import pickle
import numpy as np
from tqdm import tqdm
import argparse

Ntraj=int(20000)

h0 = gamma0, omega0, n0, eta0, kappa0 = 500.0, 10000.0, 1.0, 1, 1000.0
h1 = gamma1, omega1, n1, eta1, kappa1 = 500, 10200.0, 1.0, 1, 1000.0

omega_pro = (omega0 + omega1)/2
period = (2*np.pi/omega_pro)
dt = period/100
total_time = 10000*period
times = np.arange(0, dtotal_time + dt, dt )
if len(times)>int(1e4):
    indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    timind = [times[k] for k in indis]
else:
    timind=times
indis_range = list(range(len(indis)))


def load_freq(itraj, what="logliks.npy", flip_params=0,dtotal_time=8.):
    h0 = gamma0, omega0, n0, eta0, kappa0 = 500.0, 10000.0, 1.0, 1, 1000.0
    h1 = gamma1, omega1, n1, eta1, kappa1 = 500, 10200.0, 1.0, 1, 1000.0
    omega_pro = (omega0 + omega1)/2
    period = (2*np.pi/omega_pro)
    dt = period/100
    total_time = 10000*period
    if flip_params != 0:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(list(params))+"/"
    l =load_data(exp_path=exp_path, itraj=itraj, total_time=total_time, dt=dt, what=what)
    return l

save_path_data = "/media/giq/Nuevo vol/quantera/paper/frequency/"
os.makedirs(save_path_data, exist_ok=True)

B = 10
dB = .2
boundsB= np.arange(-B,B+dB,dB)

bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]



deter, stop = {}, {}
stop["_0"] = {i:[] for i in range(1,Ntraj)}
stop["_1"] = {i:[] for i in range(1,Ntraj)}
deter["h0/h1"] ={indb:[0]*len(indis) for indb in range(len(boundsB))}
deter["h1/h0"] = {indb:[0]*len(indis) for indb in range(len(boundsB))}

n=1
ers = []
for itraj in tqdm(range(20000)):
    try:

        [l0_1,l1_1], [l1_0,l0_0] = load_freq(itraj=itraj,what="logliks.npy", flip_params=0).T, load_freq( itraj=itraj,what="logliks.npy", flip_params=1).T
        log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0

        for indb,b in enumerate(boundsB):
            deter["h0/h1"][indb] += ((log_lik_ratio[indis_range] < b).astype(int)  - deter["h0/h1"][indb])/n
            deter["h1/h0"][indb] += ((log_lik_ratio_swap[indis_range] > b).astype(int)  - deter["h1/h0"][indb])/n
            if b>=0:
                stop["_1"][itraj].append(get_stop_time(log_lik_ratio, b, timind))
                stop["_0"][itraj].append(get_stop_time(log_lik_ratio_swap, b,timind))
        n+=1
    except Exception:
        ers.append(itraj)
        print("error {}".format(itraj))


path_data = save_path_data+"B{}_db{}_{}/".format(B,dB,n)
os.makedirs(path_data,exist_ok=True)

with open(path_data+"stop.pickle","wb") as g:
    pickle.dump(stop, g, protocol=pickle.HIGHEST_PROTOCOL)

with open(path_data+"deter.pickle","wb") as f:
    pickle.dump(deter, f, protocol=pickle.HIGHEST_PROTOCOL)
