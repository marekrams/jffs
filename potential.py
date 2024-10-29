import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv
from scipy.interpolate import CubicSpline

from run_evolution_fermions import folder_evol, folder_gs

from load_data import *

N, a = Nas[-1]

xs = np.linspace(-50, 50, N//2)
ts = Ln[m, N][:, 0]
T = len(ts)

t = np.linspace(0.6, 50, 500)

plt.figure()
plt.subplots_adjust(left=0.15)

for k in [0, 1, 2, 3]:

    m = ms[k]
    ee = Ln[m, N][:, 1:]
    ee = ee -1 # + dQ
    ee = (ee[:, 0::2] + ee[:, 1::2]) /2  #  here we calculate mean of sites 2*l and 2*l+1

    phimid = (ee[:, N//4] + ee[:, N//4-1])/2

    phispline = CubicSpline(ts, phimid)
    phi = phispline(t, nu=0)
    phit = phispline(t, nu=1)
    phitt = phispline(t, nu=2)

    kin = -( phitt + 1/t * phit)


    plt.plot(phi, kin, label=f'{m/g=:0.2f}')



plt.axhline(0, color='k', alpha=0.15, linestyle='dashed')
plt.axvline(0, color='k', alpha=0.15, linestyle='dashed')

plt.legend()
plt.ylabel(r"$dV_{eff}/d\phi \equiv -\phi''(\tau) - \frac{1}{\tau}\phi'(\tau)$")
plt.xlabel(r'$\phi$ $\equiv$ Electric field')
plt.title(r'Effective potential $dV_{eff}/d\phi$')



plt.show()
