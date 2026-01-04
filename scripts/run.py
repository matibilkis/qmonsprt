import os
import numpy as np
import sys
from datetime import datetime
import argparse

# Add parent directory to path to import numerics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from numerics.utilities.misc import *

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--mode", type=str, default="damping")
parser.add_argument("--dt", type=float, default=1e-6)
parser.add_argument("--ppp", type=float, default=1000)
parser.add_argument("--total_time", type=float, default=50)

args = parser.parse_args()

seed = args.seed
mode = args.mode
dt = args.dt
total_time = args.total_time
ppp=args.ppp

# Get root directory for running integration scripts
root_dir = os.path.join(os.path.dirname(__file__), '..')
integrate_script = os.path.join(root_dir, 'numerics', 'integration', 'integrate.py')

s = datetime.now()
for k in range(10):
    os.system("python3 {} --itraj {} --mode {} --dt {} --total_time {} --ppp {}".format(
        integrate_script, seed+k, mode, dt, total_time, ppp))
    print(datetime.now()-s)
    os.system("python3 {} --itraj {} --flip_params 1 --mode {} --dt {} --total_time {} --ppp {}".format(
        integrate_script, seed+k, mode, dt, total_time, ppp))
    print(datetime.now()-s)
