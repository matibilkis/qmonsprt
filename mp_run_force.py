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

cores =  18

def simu(itraj):
    for k in range(cores):
        st = datetime.now()
        os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} ".format(itraj+k))
        os.system("python3 numerics/integration/integrate_freq_force.py --itraj {} --flip_params 1".format(itraj+k))
        print(itraj+k)

with mp.Pool(cores) as p:
    p.map(simu, list(range(int(1e4),int(2e4),cores)))
