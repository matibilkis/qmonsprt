import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc_force import *
from numerics.integration.steps import Ikpw, Robler_step
import numpy as np
from tqdm import tqdm
import argparse
import ast
from numba import jit
from scipy.linalg import solve_continuous_are
#import pyarrow.parquet as pq
#import pyarrow as pa

def IntegrationLoop(y0_hidden, y0_exp, times, dt):
    """
    dy = f(y,t) *dt + G(y,t) *dW
    """

    yhidden = np.zeros((len(times)+1, 2))
    yexper = np.zeros((len(times)+1, len(y0_exp)))

    yhidden[0] = y0_hidden
    yexper[0] = y0_exp
    dys = []

    for ind, t in enumerate(times):
        yhidden[ind+1] = yhidden[ind] + Fhidden(yhidden[ind], t, dt)*dt + np.dot(XiCov1,dW[ind,:]) + f1*dt
        x1 = yhidden[ind][:2]
        dy =np.dot(C1,x1)*dt + np.dot(proj_C, dW[ind,:])
        dys.append(dy)
        yexper[ind+1] = EulerUpdate_x0_logliks(x1, dy, yexper[ind], dt)
    return yhidden, yexper, dys

def EulerUpdate_x0_logliks(x1,dy,s, dt):
    """
    this function updates the value of {x0,cov0} (wrong hypothesis) by using the dy
    also updates the log likelihoods l1 and l0
    """
    ### x1 is the hidden state i use to simulate the data
    x0 = s[:2]
    XiCov0C0 = np.dot(XiCov0,C0)

    dx0 = np.dot(A0 - XiCov0C0, x0)*dt + np.dot(XiCov0, dy) + f0*dt

    l0, l1 = s[2:]
    u0 = np.dot(C0,x0)
    u1 = np.dot(C1,x1)
    dl0 = -dt*np.dot(u0,u0)/2 + np.dot(u0, dy)
    dl1 = -dt*np.dot(u1,u1)/2 + np.dot(u1, dy)
    return [(x0 + dx0)[0], (x0 + dx0)[1], l0 + dl0, l1+dl1 ]


@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x1 = s[:2]
    x1_dot = np.dot(A1,x1)
    return np.array([x1_dot[0], x1_dot[1]])

@jit(nopython=True)
def Ghidden():
    return XiCov1

def integrate(params, total_time=1, dt=1e-1, itraj=1, exp_path="",**kwargs):
    """
    h1 is the hypothesis i use to get the data. (with al the coefficients gamma1...)
    """
    global proj_C, A0, A1, XiCov0, XiCov1, C0, C1, dW, model, f0, f1
    model = give_model()
    pdt = kwargs.get("pdt",1)
    dt *=pdt
    times = np.arange(0,total_time+dt,dt)
    # np.random.seed(itraj) FUCKING BIAS!!!!!!!
    dW = np.sqrt(dt)*np.random.randn(len(times),2)

    [gamma1, omega1, n1, eta1, kappa1, f1],[gamma0, omega0, n0, eta0, kappa0, f0] = params
    ### XiCov = S C.T + G.T
    #### dx  = (A - XiCov.C )x dt + (XiCov dy) = A x dt + XiCov dW
    #### dy = C x dt + dW
    #### dCov = AS + SA.T + D - xiCov xiCov.T

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        C = np.sqrt(4*eta*kappa)*np.eye(2)#
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        return A, C, D,G

    A1, C1, D1, G1 = give_matrices(gamma1, omega1, n1, eta1, kappa1)
    A0, C0, D0, G0 = give_matrices(gamma0, omega0, n0, eta0, kappa0)

    proj_C = np.linalg.pinv(C1/C1[0,0])
    x1in ,p1in, x0in, p0in, dyxin, dypin, lin0, lin1 = np.zeros(8)

    sst1 = solve_continuous_are( (A1-np.dot(G1.T,C1)).T, C1.T, D1 - np.dot(G1.T, G1), np.eye(2)) #### A.T because the way it's implemented!
    sst0 = solve_continuous_are( (A0-np.dot(G0.T,C0)).T, C0.T, D0 - np.dot(G0.T, G0), np.eye(2)) #### A.T because the way it's implemented!

    XiCov1  = np.dot(sst1, C1.T) + G1.T
    XiCov0  = np.dot(sst0, C0.T) + G0.T

    s0_hidden = np.array([x1in, p1in])
    s0_exper = np.array([x0in, p0in, lin0, lin1])

    hidden_state, exper_state, signals = IntegrationLoop(s0_hidden, s0_exper,  times, dt)
    states1 = hidden_state[:,0:2]
    states0 = exper_state[:,:2]
    liks = exper_state[:,2:]

    path = get_path_config(total_time=total_time, dt=dt, itraj=itraj, exp_path=exp_path)
    os.makedirs(path, exist_ok=True)

    if len(times)>1e4:
        indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    else:
        indis = np.arange(0,len(times))

    timind = [times[ind] for ind in indis]

    logliks_short =  np.array([liks[ii] for ii in indis])
    states1_short =  np.array([states1[ii] for ii in indis])
    states0_short =  np.array([states0[ii] for ii in indis])


    np.save(path+"logliks",logliks_short)
    np.save(path+"states1",states1_short)
    np.save(path+"states0",states0_short)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--flip_params", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=11000.)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--total_time", type=float, default=8.)
    parser.add_argument("--pdt", type=int, default=1)

    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    flip_params = args.flip_params
    gamma = args.gamma
    dt = args.dt
    pdt = args.pdt
    total_time = args.total_time


    # h0 = gamma0, omega0, n0, eta0, kappa0 = 100., 0., 1., 1, 9.
    # h1 = gamma1, omega1, n1, eta1, kappa1 = 429, 0., 1., 1, 9.
    h0 = gamma0, omega0, n0, eta0, kappa0, b0 = 500.0, 100.0, 1.0, 1, 10.0, 0.0
    h1 = gamma1, omega1, n1, eta1, kappa1, b1 = 500, 100.0, 1.0, 1, 10.0, 40

    omega_pro = (omega0 + omega1)/2
    period = (2*np.pi/omega_pro)
    dt = period/100
    total_time = 1000*period

    if flip_params == 1:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(params)+"/"


    integrate(params=params,
              total_time = total_time,
              dt = dt,
              itraj=itraj,
              exp_path = exp_path,
              pdt = pdt)


###
