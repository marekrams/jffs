import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv

from run_evolution_fermions import folder_evol, folder_gs

from load_data import *

N, a = Nas[-1]

xs = np.linspace(-50, 50, N//2)
ts = Ln[m, N][:, 0]
T = len(ts)

x = np.zeros((T, N//2))
t = np.zeros((T, N//2))

x[:, :] = xs[np.newaxis, :]
t[:, :] = ts[:, np.newaxis]

lc = (x**2<t**2)    # forward light cone
tau = np.sqrt(t**2-x**2)[lc]

tausort = np.sort(tau)

for k in [0, 1, 2, 3]:

    m = ms[k]
    ee = Ln[m, N][:, 1:]
    ee = ee -1 # + dQ
    ee = (ee[:, 0::2] + ee[:, 1::2]) /2  #  here we calculate mean of sites 2*l and 2*l+1

    plt.figure()
    Ls = ee[lc]

    plt.scatter(tau, Ls, s=1, alpha=0.2, label='quantum')



    # if k==0:
    #     plt.plot(tausort, -(jv(0, mb*tausort)), color='r', label='classical')

    phi = np.loadtxt('tabphi_{}.dat'.format(k))
    plt.plot(phi[:,0], phi[:, 1], color='g', label='classical Berges Eq. (10)')

    mid = (ee[:, N//4] + ee[:, N//4-1])/2
    plt.plot(ts, mid, color='r', label='quantum x=0')
    print(xs[N//4-1], xs[N//4])

    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'Electric field($\tau$)')
    plt.title(f'boost invariance {m/g=:0.2f} {a=:0.2f}')

plt.show()
