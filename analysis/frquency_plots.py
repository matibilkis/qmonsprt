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

Ntraj=int(5000)

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


liks_1 = {}
liks_0 = {}
for itraj in tqdm(range(5000)):
    try:
        liks_1[itraj] = load_freq(itraj)
        liks_0[itraj] = load_freq(itraj, flip_params =1)
    except Exception:
        pass


save_path_data = "/media/giq/Nuevo vol/quantera/paper/frequency/"
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
    [l0_1,l1_1], [l1_0,l0_0] = load_freq(itraj=itraj,what="logliks.npy", flip_params=0).T, load_freq(itraj=itraj,what="logliks.npy", flip_params=1).T
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

times = timind[::100]
timstats = [times[k] for k in np.linspace(0,len(times)-1,ll0.shape[1]).astype(int)]


tp = 30
plt.figure(figsize=(40,40))
for i,k in enumerate([1000, -1]):

    ax=plt.subplot2grid((2,1),(i,0))
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
    #ax.set_xlim(-70,70)
    ax.set_xlabel(r'$\ell$',size=int(2*tp))
    ax.xaxis.set_tick_params(labelsize=tp)
    ax.yaxis.set_tick_params(labelsize=tp)
plt.subplots_adjust(hspace=0.25)
plt.savefig(save_path_data+"histograms_ell.pdf")



def fit_2momentss(timind,l1_mean, l1_std):
    ini = 10
    fini = -1
    sqrtimind = np.array(np.sqrt(timind))
    timind = np.array(timind)
    mu, oomu = np.polyfit(timind[ini:fini], np.abs(l1_mean)[ini:fini],1)
    sigma, oosig = np.polyfit(np.array(np.sqrt(timind))[ini:fini], l1_std[ini:fini],1)
    return mu,sigma

mu1, sigma1 = fit_2momentss(timind, np.mean(ll1,axis=0), np.std(ll1,axis=0))
mu0, sigma0 = fit_2momentss(timind, np.mean(ll0,axis=0), np.std(ll0,axis=0))





B = 400
dB = 15
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
alphagau = [.5] + [dete_alpha(t, boundsB[indb],mu0, sigma0) for t in titi]
betagau = [.5] + [dete_beta(t, boundsB[indb],mu1, sigma1) for t in titi]
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




indb = 10
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
yanal_seq = .5*(bpos*(-2*np.array([alpha_seq(abs(b)) for b in bpos]) +1.)/mu0 + bpos*(1-2*np.array([beta_seq(b) for b in bpos]))/mu1)
yanal_det = time_det
xnume_det = -np.log(sime)[:lim][::PD]
ynume_det = timind[:lim][::PD]

plt.figure()
LAP = .7
SIML = "red"
SIMC = "blue"
asca = .5

SS=400

xnume_det

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
