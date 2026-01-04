import os
import sys
import multiprocessing as mp
from datetime import datetime
import argparse

# Add parent directory to path to import numerics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from numerics.utilities.misc import *

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

#cores =  mp.cpu_count()
cores = 8#32
gammas = np.linspace(110., 10000, 32) #---> all batch, before 6/11
gammas = np.linspace(gammas[0] + 10, gammas[1]-10, 32) #---> all batch, before 6/11


# Get root directory for running integration scripts
root_dir = os.path.join(os.path.dirname(__file__), '..')
integrate_script = os.path.join(root_dir, 'numerics', 'integration', 'integrate.py')

def simu(gamma):
    st = datetime.now()
    os.system("python3 {} --itraj {} --gamma {} --pdt 1 --dt 1e-4".format(integrate_script, itraj, gamma))
    os.system("python3 {} --itraj {} --flip_params 1 --gamma {} --pdt 1 --dt 1e-4".format(integrate_script, itraj, gamma))
    print(itraj, cores, gamma, (datetime.now() - st).seconds)

with mp.Pool(cores) as p:
    p.map(simu, gammas)
