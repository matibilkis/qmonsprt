# %% codecell
import os
os.chdir('/home/giq/qmonsprt')
import sys
sys.path.insert(0, os.getcwd())

import numpy as np
from numerics.utilities.misc import *
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.special import erf
from numerics.utilities.misc import *

Ntraj=int(20000)
gamma = 429
dtotal_time = 8.
dt = 1e-5
times = np.arange(0, dtotal_time + dt, dt )
if len(times)>int(1e4):
    indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    timind = [times[k] for k in indis]
else:
    timind=times
indis_range = list(range(len(indis)))


def load_gamma(itraj, what="logliks.npy", flip_params=0,dtotal_time=8.):
    h0 = gamma0, omega0, n0, eta0, kappa0 = 100., 0., 1., 1, 9.
    h1 = gamma1, omega1, n1, eta1, kappa1 = 429, 0., 1., 1, 9.
    if flip_params != 0:
        params = [h0, h1]
    else:
        params = [h1,h0]
    exp_path = str(params)+"/"
    l =load_data(exp_path=exp_path, itraj=itraj, total_time=dtotal_time, dt=1e-5, what=what)
    return l



liks_1 = {}
liks_0 = {}
for itraj in tqdm(range(20000)):
    try:
        liks_1[itraj] = load_gamma(itraj)
        liks_0[itraj] = load_gamma(itraj, flip_params =1)
    except Exception:
        pass


save_path_data = "/media/giq/Nuevo vol/quantera/paper/damping/"
os.makedirs(save_path_data,exist_ok=True)

liks_data_0 = np.stack(list(liks_0.values()))
liks_data_1 = np.stack(list(liks_1.values()))
ll1 = liks_data_1[:,:,1] - liks_data_1[:,:,0]
ll0 = liks_data_0[:,:,0] - liks_data_0[:,:,1]

np.save(save_path_data+"ll_0",ll0)
np.save(save_path_data+"ll_1",ll1)


plt.figure(figsize=(20,20))
ax=plt.subplot()
timindt = timind[::100]
ax.plot(timind,np.mean(ll1,axis=0),color="red", alpha=.7, linewidth=5,label=r'$\langle\ell\rangle_{|1}$')
ax.plot(timind,np.mean(ll0,axis=0),color="blue", alpha=.7, linewidth=5,label=r'$\langle\ell\rangle_{|0}$')

for itraj in range(1,10):
    [l0_1,l1_1], [l1_0,l0_0] = load_gamma(itraj=itraj,what="logliks.npy", flip_params=0).T, load_gamma(itraj=itraj,what="logliks.npy", flip_params=1).T
    log_lik_ratio, log_lik_ratio_swap = l1_1-l0_1, l1_0-l0_0
    ax.plot(timind,log_lik_ratio,alpha=.4, color="red")
    ax.plot(timind,log_lik_ratio_swap, alpha=.4, color="blue")
ax.legend(prop={"size":30})
ax.set_xlabel("time",size=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.savefig(save_path_data+"avg_ell_time.pdf")





def fit_2moments(timind,l1_mean, l1_std):
    ini = 10
    fini = -1
    sqrtimind = np.array(np.sqrt(timind))
    timind = np.array(timind)
    mu, oomu = np.polyfit(timind[ini:fini], np.abs(l1_mean)[ini:fini],1)
    sigma, oosig = np.polyfit(np.array(np.sqrt(timind))[ini:fini], l1_std[ini:fini],1)
    return [mu, oomu], [sigma, oosig]

def give_me_gauss_x(b, mu,sigma,xx):
    gauss = lambda x,m,g: np.exp(-((x-m)**2)/(2*g**2))/np.sqrt(2*np.pi*g**2)
    return xx, np.array([gauss(x, mu, sigma) for x in xx])

times = np.linspace(0,8+1e-5, int(1e4))[::100]
timstats = [times[k] for k in np.linspace(0,len(times)-1,ll0.shape[1]).astype(int)]

tp = 30
plt.figure(figsize=(40,40))
for i,k in enumerate([1000,-1]):

    ax=plt.subplot2grid((3,1),(i,0))
    indstats = k
    t = timind[indstats]
    lW=6
    c1, b1 = np.histogram(ll1[:,indstats], bins=50, normed=True)
    c0, b0 = np.histogram(ll0[:,indstats], bins=50, normed=True)

    [mu1, oomu1], [sigma1, oosig1] = fit_2moments(timind,np.mean(ll1, axis=0),np.std(ll1,axis=0))
    [mu0, oomu0], [sigma0, oosig0] = fit_2moments(timind,np.mean(ll0, axis=0),np.std(ll0,axis=0))

    bb1,gg1 = give_me_gauss_x(b1, mu1*t,sigma1*np.sqrt(t), b1[:-1])
    bb0,gg0 = give_me_gauss_x(b0, -mu0*t,sigma0*np.sqrt(t), b0[:-1])

    ax.plot(bb0,gg0,'--',color="blue",linewidth=lW,label=r'$\mathcal{N}(\mu_0 t, \sigma_0 \sqrt{t})$')
    ax.plot(bb1,gg1,'--',color="red",linewidth=lW,label=r'$\mathcal{N}(\mu_1 t, \sigma_1 \sqrt{t})$')
    ax.bar(b0[1:],c0,color="blue",alpha=0.7,label=r'$\hat{P}_{|0}(\ell)$',edgecolor="black", width=b0[1]-b0[0])
    ax.bar(b1[1:],c1,color="red",alpha=0.7,label=r'$\hat{P}_{|1}(\ell)$',edgecolor="black", width=b1[1]-b1[0])
    #ax.set_title(r'$t = $'+str(np.round(t,2)), size=30)
    ax.legend(prop={"size":45})
    ax.set_xlim(-70,70)
    ax.set_xlabel(r'$\ell$',size=int(2*tp))
    ax.xaxis.set_tick_params(labelsize=tp)
    ax.yaxis.set_tick_params(labelsize=tp)
plt.subplots_adjust(hspace=0.25)
plt.savefig(save_path_data+"histograms_ell.pdf")








def give_mu(gamma, flip_params=0):
    h0 = gamma0, omega0, n0, eta0, kappa0 = 100., 0., 1., 1., 9
    h1 = gamma1, omega1, n1, eta1, kappa1 = gamma, 0., 1., 1., 9

    if flip_params ==1:
        params = [h0,h1]
    else:
        params = [h1, h0]

    [gamma1, omega1, n1, eta1, kappa1],[gamma0, omega0, n0, eta0, kappa0] = params

    Su1 = n1 + 0.5 + (kappa1 / gamma1)
    Su0 = n0 + 0.5 + (kappa0 / gamma0)
    S1 = (np.sqrt(1 + (16.0*eta1*kappa1*Su1/gamma1)) - 1)*(gamma1/(8.0*eta1*kappa1))
    S0 = (np.sqrt(1 + (16.0*eta0*kappa0*Su0/gamma0)) - 1)*( gamma0/(8.0*eta0*kappa0))

    lam = gamma0 + (8*eta0*kappa0*S0)

    aa = (4*eta1*kappa1*(S1**2))/gamma1
    bb =(4*eta0*kappa0*S0**2)*(1+((16.0*eta1*kappa1*S1)/ (gamma1 + lam)) + (64.0*(eta1 * kappa1 * S1)**(2)/(gamma1 * (gamma1 + lam))))/ lam
    c =8 *(S0*S1*(eta0*kappa0 *eta1*kappa1)**(0.5)) * (gamma1+ (4.0*eta1*kappa1*S1) ) / ((gamma1 + lam)*gamma1)

    mu = 4*(eta1*kappa1*aa + (eta0*kappa0*bb) - 2*np.sqrt(eta1*kappa1*eta0*kappa0)*c)
    return mu


def give_var(gamma, flip_params=0):
    h0 = gamma0, omega0, n0, eta0, kappa0 = 100., 0., 1., 1., 9
    h1 = gamma1, omega1, n1, eta1, kappa1 = gamma, 0., 1., 1., 9

    if flip_params ==0:
        params = [h0,h1]
    else:
        params = [h1, h0]

    [gamma1, omega1, n1, eta1, kappa1],[gamma0, omega0, n0, eta0, kappa0] = params

    def give_matrices(gamma, omega, n, eta, kappa):
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        mm = np.eye(2)#homodyne but in Rotating Frame
        C = np.sqrt(4*eta*kappa)*mm#
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        return A, C, D,G

    A1, C1, D1, G1 = give_matrices(gamma1, omega1, n1, eta1, kappa1)
    A0, C0, D0, G0 = give_matrices(gamma0, omega0, n0, eta0, kappa0)

    proj_C = np.linalg.pinv(C1/C1[0,0])

    sst1 = solve_continuous_are( (A1-np.dot(G1.T,C1)).T, C1.T, D1 - np.dot(G1.T, G1), np.eye(2)) #### A.T because the way it's implemented!
    sst0 = solve_continuous_are( (A0-np.dot(G0.T,C0)).T, C0.T, D0 - np.dot(G0.T, G0), np.eye(2)) #### A.T because the way it's implemented!

    sigma0 = sst0[0,0]
    sigma1 = sst1[1,1]
    eta = eta1
    kappa = kappa1
    n = n1
    var1 = (32*(eta**2*gamma0**5*gamma1**3*kappa**2*sigma0**2 + 3*eta**2*gamma0**4*gamma1**4*kappa**2*sigma0**2 + 3*eta**2*gamma0**3*gamma1**5*kappa**2*sigma0**2 + eta**2*gamma0**2*gamma1**6*kappa**2*sigma0**2 + 8*eta**3*gamma0**4*gamma1**3*kappa**3*sigma0**3 + 24*eta**3*gamma0**3*gamma1**4*kappa**3*sigma0**3 + 24*eta**3*gamma0**2*gamma1**5*kappa**3*sigma0**3 + 8*eta**3*gamma0*gamma1**6*kappa**3*sigma0**3 + 16*eta**4*gamma0**3*gamma1**3*kappa**4*sigma0**4 + 48*eta**4*gamma0**2*gamma1**4*kappa**4*sigma0**4 + 48*eta**4*gamma0*gamma1**5*kappa**4*sigma0**4 + 16*eta**4*gamma1**6*kappa**4*sigma0**4 - 4*eta**2*gamma0**5*gamma1**3*kappa**2*sigma0*sigma1 - 8*eta**2*gamma0**4*gamma1**4*kappa**2*sigma0*sigma1 - 4*eta**2*gamma0**3*gamma1**5*kappa**2*sigma0*sigma1 + 24*eta**3*gamma0**5*gamma1**2*kappa**3*sigma0**2*sigma1 + 16*eta**3*gamma0**4*gamma1**3*kappa**3*sigma0**2*sigma1 - 8*eta**3*gamma0**3*gamma1**4*kappa**3*sigma0**2*sigma1 + 192*eta**4*gamma0**4*gamma1**2*kappa**4*sigma0**3*sigma1 + 256*eta**4*gamma0**3*gamma1**3*kappa**4*sigma0**3*sigma1 + 192*eta**4*gamma0**2*gamma1**4*kappa**4*sigma0**3*sigma1 + 128*eta**4*gamma0*gamma1**5*kappa**4*sigma0**3*sigma1 + 384*eta**5*gamma0**3*gamma1**2*kappa**5*sigma0**4*sigma1 + 512*eta**5*gamma0**2*gamma1**3*kappa**5*sigma0**4*sigma1 + 384*eta**5*gamma0*gamma1**4*kappa**5*sigma0**4*sigma1 + 256*eta**5*gamma1**5*kappa**5*sigma0**4*sigma1 + eta**2*gamma0**6*gamma1**2*kappa**2*sigma1**2 + 3*eta**2*gamma0**5*gamma1**3*kappa**2*sigma1**2 + 3*eta**2*gamma0**4*gamma1**4*kappa**2*sigma1**2 + eta**2*gamma0**3*gamma1**5*kappa**2*sigma1**2 - 48*eta**3*gamma0**5*gamma1**2*kappa**3*sigma0*sigma1**2 - 128*eta**3*gamma0**4*gamma1**3*kappa**3*sigma0*sigma1**2 - 80*eta**3*gamma0**3*gamma1**4*kappa**3*sigma0*sigma1**2 + 192*eta**4*gamma0**5*gamma1*kappa**4*sigma0**2*sigma1**2 - 64*eta**4*gamma0**4*gamma1**2*kappa**4*sigma0**2*sigma1**2 - 512*eta**4*gamma0**3*gamma1**3*kappa**4*sigma0**2*sigma1**2 - 256*eta**4*gamma0**2*gamma1**4*kappa**4*sigma0**2*sigma1**2 + 1536*eta**5*gamma0**4*gamma1*kappa**5*sigma0**3*sigma1**2 + 1024*eta**5*gamma0**3*gamma1**2*kappa**5*sigma0**3*sigma1**2 + 512*eta**5*gamma0*gamma1**4*kappa**5*sigma0**3*sigma1**2 + 3072*eta**6*gamma0**3*gamma1*kappa**6*sigma0**4*sigma1**2 + 2048*eta**6*gamma0**2*gamma1**2*kappa**6*sigma0**4*sigma1**2 + 1024*eta**6*gamma1**4*kappa**6*sigma0**4*sigma1**2 + 8*eta**3*gamma0**6*gamma1*kappa**3*sigma1**3 + 48*eta**3*gamma0**5*gamma1**2*kappa**3*sigma1**3 + 72*eta**3*gamma0**4*gamma1**3*kappa**3*sigma1**3 + 32*eta**3*gamma0**3*gamma1**4*kappa**3*sigma1**3 - 256*eta**4*gamma0**5*gamma1*kappa**4*sigma0*sigma1**3 - 768*eta**4*gamma0**4*gamma1**2*kappa**4*sigma0*sigma1**3 - 512*eta**4*gamma0**3*gamma1**3*kappa**4*sigma0*sigma1**3 + 512*eta**5*gamma0**5*kappa**5*sigma0**2*sigma1**3 - 1024*eta**5*gamma0**4*gamma1*kappa**5*sigma0**2*sigma1**3 - 3072*eta**5*gamma0**3*gamma1**2*kappa**5*sigma0**2*sigma1**3 - 2048*eta**5*gamma0**2*gamma1**3*kappa**5*sigma0**2*sigma1**3 + 4096*eta**6*gamma0**4*kappa**6*sigma0**3*sigma1**3 + 8192*eta**7*gamma0**3*kappa**7*sigma0**4*sigma1**3 + 16*eta**4*gamma0**6*kappa**4*sigma1**4 + 240*eta**4*gamma0**5*gamma1*kappa**4*sigma1**4 + 624*eta**4*gamma0**4*gamma1**2*kappa**4*sigma1**4 + 400*eta**4*gamma0**3*gamma1**3*kappa**4*sigma1**4 - 512*eta**5*gamma0**5*kappa**5*sigma0*sigma1**4 - 2560*eta**5*gamma0**4*gamma1*kappa**5*sigma0*sigma1**4 - 1024*eta**5*gamma0**3*gamma1**2*kappa**5*sigma0*sigma1**4 - 2048*eta**6*gamma0**4*kappa**6*sigma0**2*sigma1**4 - 10240*eta**6*gamma0**3*gamma1*kappa**6*sigma0**2*sigma1**4 - 4096*eta**6*gamma0**2*gamma1**2*kappa**6*sigma0**2*sigma1**4 + 384*eta**5*gamma0**5*kappa**5*sigma1**5 + 2304*eta**5*gamma0**4*gamma1*kappa**5*sigma1**5 + 2432*eta**5*gamma0**3*gamma1**2*kappa**5*sigma1**5 - 4096*eta**6*gamma0**4*kappa**6*sigma0*sigma1**5 - 16384*eta**7*gamma0**3*kappa**7*sigma0**2*sigma1**5 + 3072*eta**6*gamma0**4*kappa**6*sigma1**6 + 7168*eta**6*gamma0**3*gamma1*kappa**6*sigma1**6 + 8192*eta**7*gamma0**3*kappa**7*sigma1**7))/(gamma0**3*(gamma1 + 8*eta*kappa*sigma1)**3*(gamma0 + gamma1 + 8*eta*kappa*sigma1)**3)
    return var1


mu1, mu0 = give_mu(429),give_mu(429,flip_params=1)
s1, s0 = give_var(429),give_var(429,flip_params=1)


lw = 5
aa = 0.7
ini = 100
fini = -1
ls = 50
tp = 40
fig =plt.figure(figsize=(40,40))
ax=plt.subplot2grid((2,2),(0,0))
ax.plot(timind[ini:fini],(np.abs(np.mean(ll1,axis=0)/timind)/mu1)[ini:fini], linewidth=lw, alpha=aa,label=r'$\frac{\langle \ell \rangle_{|1}}{\mu_1 t}$', color="red")
ax.axhline(1, linestyle="--",linewidth=lw, alpha=aa, color="black")
ax.set_xticks([])

ax=plt.subplot2grid((2,2),(1,0))
ax.plot(timind[ini:fini],(np.abs(np.mean(ll0,axis=0)/timind)/mu0)[ini:fini], linewidth=lw, alpha=aa,label=r'$\frac{\langle \ell \rangle_{|0}}{\mu_0 t}$',color="blue")
ax.axhline(1, linestyle="--",linewidth=lw, alpha=aa, color="black")

ax.set_xlabel("time",size=ls)
ax=plt.subplot2grid((2,2),(0,1))
ax.plot(timind[ini:fini],(np.abs(np.std(ll1,axis=0)**2/timind)/s1)[ini:fini], linewidth=lw, alpha=aa,label=r'$\frac{Var[\ell]_{|1}}{\sigma_1^2 t}$', color="red")
ax.axhline(1, linestyle="--",linewidth=lw, alpha=aa, color="black")
ax.set_xticks([])

ax=plt.subplot2grid((2,2),(1,1))
ax.plot(timind[ini:fini],(np.abs(np.std(ll0,axis=0)**2/timind)/s0)[ini:fini], linewidth=lw, alpha=aa,label=r'$\frac{Var[\ell]_{|0}}{\sigma_0^2 t}$',color="blue")
ax.axhline(1, linestyle="--",linewidth=lw, alpha=aa, color="black")
ax.set_xlabel("time",size=ls)

for i, ax in enumerate(fig.axes):
    ax.xaxis.set_tick_params(labelsize=tp)
    ax.yaxis.set_tick_params(labelsize=tp)
    ax.legend(prop={"size":80})
plt.subplots_adjust(hspace=0.05)

plt.savefig(save_path_data+"moments.pdf")













B = 6.0
dB = 0.2
n = 20000
boundsB= np.arange(-B,B+dB,dB)
bpos = boundsB[boundsB>=0]
bneg = boundsB[boundsB<0]
path_data = save_path_data+"B{}_db{}_{}/".format(B,dB,n)


with open(path_data+"deter.pickle","rb") as f:
    deter = pickle.load(f)

alphas = list(deter["h1/h0"].values())
betas = list(deter["h0/h1"].values())

alphas = np.stack(alphas)
betas = np.stack(betas)
epsilon = lambda o: (1-np.exp(-abs(o)))/(np.exp(abs(o)) - np.exp(-abs(o)))
indb0 = np.argmin(np.abs(boundsB))

def dete_alpha(t, b, mu,sigma):
    inside = (b + mu*t)/(np.sqrt(2*t*sigma**2))
    return (1 -  erf(inside))/2

def dete_beta(t, b, mu,sigma):
    inside = (b - mu*t)/(np.sqrt(2*t*sigma**2))
    return (1 +  erf(inside))/2



NP = 1000
AA = 50
titi = np.linspace(1e-14, timind[AA:][-1], NP)
indb = np.argmin(np.abs(boundsB))
alphagau = [.5] + [dete_alpha(t, boundsB[indb],give_mu(gamma,flip_params=1), np.sqrt(give_var(gamma,flip_params=1))) for t in titi]
betagau = [.5] + [dete_beta(t, boundsB[indb],give_mu(gamma,flip_params=0), np.sqrt(give_var(gamma,flip_params=0))) for t in titi]
titi = np.array([0] + list(titi))
sim = .5*(np.array(alphagau) + np.array(betagau))

plt.figure(figsize=(22,12))
ax=plt.subplot2grid((2,2),(0,0))
S=40
ss=30
LW = 5
step = 80
asca=.7
x0,y0 = .3,.2
CALPHA = "magenta"
CALPHAG = "green"
BETAC = "red"
BETAG = "purple"
SIMC = "purple"
SIML = "orange"
ax.scatter(timind[AA:][::step],alphas[indb0+1,:][AA:][::step], s=S, alpha=asca, edgecolor="face", color=CALPHA, label=r'$\hat{\alpha}$')
ax.plot(titi,alphagau, linewidth=LW,alpha=.7, color=CALPHAG, label=r'$\alpha_G$')
axins = ax.inset_axes([x0,y0,.6,.5])

axins.plot(titi,alphagau,linewidth=LW,alpha=.7, color=CALPHAG)
axins.scatter(timind[AA:][::step],betas[indb0,:][AA:][::step],  s=S, alpha=asca, edgecolor="face", color=CALPHA)

axins.set_yscale("log")

ax.set_xlabel(r'$time$',size=int(.7*ss))
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
legend = ax.legend(prop={"size":20})
ax=plt.subplot2grid((2,2),(0,1))

ax.plot(titi,betagau, linewidth=LW,alpha=.7, color=BETAG, label=r'$\beta_G$')

ax.scatter(timind[AA:][::step],betas[indb0+1,:][AA:][::step],  alpha=asca, edgecolor="face", color=BETAC,  label=r'$\hat{\beta}$')
axins = ax.inset_axes([x0,y0,.6,.5])
axins.plot(titi,betagau, linewidth=LW,alpha=.7, color=BETAG)
axins.scatter(timind[AA:][::step],betas[indb0,:][AA:][::step], s=S, alpha=asca, edgecolor="face", color=BETAC)

axins.set_yscale("log")

ax.set_xlabel(r'$time$',size=int(.7*ss))
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
legend = ax.legend(prop={"size":20})

ax=plt.subplot2grid((2,2),(1,0), colspan=2)
ax.plot(titi,sim, linewidth=LW,alpha=.7, color=SIML, label=r'$P^{(G)}_e$')
ax.scatter(timind[AA:][::step],.5*(alphas[indb0+1,:][AA:] + betas[indb0+1,:][AA:])[::step], s=S, alpha=asca, edgecolor="face", color=SIMC, label=r'$\hat{P}_e$')

axins = ax.inset_axes([.3,.4,.6,.5])
axins.plot(titi,sim,  linewidth=LW,alpha=.7, color=SIML)
axins.scatter(timind[AA:][::step],.5*(alphas[indb0+1,:][AA:] + betas[indb0+1,:][AA:])[::step], s=S, alpha=asca, edgecolor="face", color=SIMC)

axins.set_yscale("log")
legend = ax.legend(prop={"size":20})

ax.set_xlabel(r'$time$',size=ss)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.savefig(save_path_data+"deterministic_errors_time.pdf")










with open(path_data+"stop.pickle","rb") as f:
    stop = pickle.load(f)

stops0 = [[] for k in range(len(bpos))]
stops1 = [[] for k in range(len(bpos))]

values1 = list(stop["_1"].values())
values0 = list(stop["_0"].values())

stops0 = np.array([k[:len(bpos)] for k in values0])
stops1 = np.array([k[:len(bpos)] for k in values1])

avg_times1 = np.nanmean(stops1,axis=0)
avg_times0 = np.nanmean(stops0,axis=0)

alpha_seq = lambda b: (1-np.exp(-b))/(np.exp(b) - np.exp(-b))
beta_seq = lambda b: (1-np.exp(-b))/(np.exp(b) - np.exp(-b))
epsilon = lambda b: np.exp(-abs(b))/(1 + np.exp(-abs(b)))
bb = lambda ep: np.log((1-ep)/ep)


LW = 4
LAP = 0.7
tp = 40
ss = 50
LS=40
SS = 200
fig=plt.figure(figsize=(20,20))
ax = plt.subplot2grid((2,2),(0,0))
ax.scatter(-np.log(epsilon(bpos)), avg_times1, color="red", alpha=0.7, edgecolor="black",s=SS, label=r'$\langle \tau \rangle_{|1}$')
ax.plot(-np.log(epsilon(bpos)), bpos*(1-2*np.array([beta_seq(b) for b in bpos]))/mu1, color="green", linewidth=LW, alpha=LAP, label=r'$\frac{\langle \ell_{\tau} \rangle_{|1}}{\mu_1}$')

ax = plt.subplot2grid((2,2),(0,1))
ax.scatter(-np.log(epsilon(bpos)), avg_times0, color="blue", alpha=0.7, edgecolor="black",s=SS,label=r'$\langle \tau \rangle_{|0}$')
ax.plot(-np.log(epsilon(bpos)), bpos*(-2*np.array([alpha_seq(abs(b)) for b in bpos]) +1.)/mu0, color="magenta", linewidth=LW, alpha=LAP, label=r'$\frac{\langle \ell_{\tau} \rangle_{|0}}{\mu_0}$')
legend = ax.legend(prop={"size":20})

ax = plt.subplot2grid((2,2),(1,0),colspan=2)
ax.scatter(-np.log(epsilon(bpos)), .5*(avg_times1+avg_times0), color="red", alpha=0.7, s=SS,edgecolor="black", label=r'$\langle \bar{\tau} \rangle$')
ax.plot(-np.log(epsilon(bpos)), .5*(bpos*(-2*np.array([alpha_seq(abs(b)) for b in bpos]) +1.)/mu0 + bpos*(1-2*np.array([beta_seq(b) for b in bpos]))/mu1), color="green", linewidth=LW, alpha=LAP, label=r'$\frac{1}{2}(\frac{\langle \ell_{\tau} \rangle_{|0}}{\mu_0} + \frac{\langle \ell_{\tau} \rangle_{|1}}{\mu_1})$')

for i, ax in enumerate(fig.axes):
    ax.xaxis.set_tick_params(labelsize=tp)
    ax.yaxis.set_tick_params(labelsize=tp)
    ax.legend(prop={"size":LS})
    ax.set_xlabel(r'$-\log{\epsilon}$',size=ss)

legend = ax.legend(prop={"size":LS})
plt.savefig(save_path_data+"wald_stop_time.pdf")



LW = 4
LAP = 0.7
tp = 40
ss = 50
LS=40
SS = 200
fig=plt.figure(figsize=(20,20))
ax = plt.subplot()#((2,2),(1,0),colspan=2)
ax.scatter(-np.log(epsilon(bpos)), .5*(avg_times1+avg_times0), color="red", alpha=0.7, s=SS, edgecolor="black", label=r'$\langle \bar{\tau} \rangle$')
ax.plot(-np.log(epsilon(bpos)), .5*(bpos*(-2*np.array([alpha_seq(abs(b)) for b in bpos]) +1.)/mu0 + bpos*(1-2*np.array([beta_seq(b) for b in bpos]))/mu1), color="green", linewidth=LW, alpha=LAP, label=r'$\frac{1}{2}(\frac{\langle \ell_{\tau} \rangle_{|0}}{\mu_0} + \frac{\langle \ell_{\tau} \rangle_{|1}}{\mu_1})$')

for i, ax in enumerate(fig.axes):
    ax.xaxis.set_tick_params(labelsize=tp)
    ax.yaxis.set_tick_params(labelsize=tp)
    ax.legend(prop={"size":LS})
    ax.set_xlabel(r'$-\log{\epsilon}$',size=ss)

legend = ax.legend(prop={"size":LS})
plt.savefig(save_path_data+"wald_stop_time_symm.pdf")




indb = -1
counts1, bins1 = np.histogram(stops1[:,indb][~np.isnan(stops1[:,indb])], 51, normed=True)

def prob_craft(t, b, mu, S):
    div = (np.sqrt(2*np.pi)*S*(t**(3/2)))
    return  abs(b)*np.exp(-((b-mu*t)**2)/(2*t*(S**2)))/div


LW=5
ss=30
ts = 30
tgauss1=np.linspace(1e-8, bins1[-1], 100)
good1 = [prob_craft(ttt, bpos[indb] , mu1, sigma1) for ttt in tgauss1]

plt.figure(figsize=(20,10))
ax=plt.subplot2grid((1,1),(0,0))
ax.bar(.5*(bins1[:-1]+bins1[1:]), counts1, width=bins1[1]-bins1[0], color="red", alpha=0.75, edgecolor="black",label=r'$\hat{P}_{|1}(\tau)$')
ax.plot(tgauss1, good1, color="black", linewidth=LW, label=r'$P^{(G)}_{|1}(\tau)$')
ax.set_xlabel(r'$\tau$',size=50)
ax.xaxis.set_tick_params(labelsize=ts)
ax.yaxis.set_tick_params(labelsize=ts)
legend = ax.legend(prop={"size":40})
plt.savefig(save_path_data+"/wald_distro.pdf")



PD = 150
LS=30
avg = .5*(avg_times0 + avg_times1)
sime = .5*(alphas[indb0,:] + betas[indb0,:])
Pe_seq = .5*(alpha_seq(bpos) + beta_seq(bpos))
time_det = np.array([timind[np.argmin(np.abs(sime-err))] for err in Pe_seq])
lim = np.argmin(np.abs(timind - time_det[-1]))

xnumer_seq = xanal_seq = xanal_det = -np.log(Pe_seq)
ynumer_seq = .5*(avg_times1+avg_times0)
yanal_seq = .5*(bpos*(-2*np.array([alpha_seq(abs(b)) for b in bpos]) +1.)/give_mu(gamma, flip_params=1) + bpos*(1-2*np.array([beta_seq(b) for b in bpos]))/give_mu(gamma))
yanal_det = time_det
xnume_det = -np.log(sime)[:lim][::PD]
ynume_det = timind[:lim][::PD]

plt.figure()
LAP = .7
SIML = "red"
SIMC = "blue"
asca = .5

SS=400

plt.figure(figsize=(20,20))
ax = plt.subplot2grid((1,1),(0,0))#,colspan=2)
ax.scatter(xnumer_seq,ynumer_seq,color="red", alpha=0.7,s=SS,edgecolor="black", label=r'$\langle \bar{\tau} \rangle$')
ax.plot(xanal_seq, yanal_seq, color="green", linewidth=LW, alpha=LAP, label=r'$\langle \bar{\tau} \rangle_{Wald}$')

ax.plot(xanal_det, yanal_det, color="magenta", linewidth=LW, alpha=LAP,label=r'$T_{det}^G$')
ax.scatter(xnume_det, ynume_det, alpha=0.7, edgecolor="black", marker="s", s=SS,color="purple", label=r'$T_{det}^G$')


ax.set_xlabel(r'$-\log{P_e}$',size=40)
ax.xaxis.set_tick_params(labelsize=LS)
ax.yaxis.set_tick_params(labelsize=LS)
legend = ax.legend(prop={"size":40})
plt.savefig(save_path_data+"/comparison.pdf")

<# %% codecell

# %% codecell

# %% codecell
