import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run_evolution import folder_evol

g = 1
v, Q = 1, 1
tol, method = 1e-6, '12site'
D0, D = 256, 256
#
ms = [0, 0.1, 0.2, 0.318309886, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mg = [g * x for x in ms]
NaDdt =  [(512, 0.125, 1024, 1/16)]  # (512, 0.0625, 256, 1/16),

data = {}

for m in ms:
    for N, a, D, dt in NaDdt:
        D0 = D
        try:
            folder = folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method, mkdir=False)
            data[m, N, a, D, dt] = np.load(folder / f"results.npy", allow_pickle=True).item()
        except FileNotFoundError:
            pass


#

NUM_COLORS = 11
cm = plt.get_cmap('gist_rainbow')
colors = [cm(i / NUM_COLORS) for i in range(NUM_COLORS)]
lines = ['-', '--', ':']


# 

def get_tsm(signals, ev):
    tm = signals["time"]
    mask = tm > -1
    tm = tm[mask]
    ee = signals[ev][mask]
    ee = ee - ee[0, :]
    ee = (ee[:, 0::2] + ee[:, 1::2]) / 2  # average over 2*n and 2*n+1
    mid = (ee[:, N//4] +ee[:, N//4-1])/2
    return tm, ee, mid

# FIG 3 left

sel = [3, 4, 5, 6, 7]  #, 7]


def sqrtreg(arr):
    out = np.zeros_like(arr)
    out[arr>0] = np.sqrt(arr[arr>0])
    return out

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i not in sel:
            continue


        tm, T00, midE = get_tsm(data[m, N, a, D, dt], 'T00')
        tm, T11, midp = get_tsm(data[m, N, a, D, dt], 'T11')
        tm, T01, midp = get_tsm(data[m, N, a, D, dt], 'T01')

        # print(T00.shape, T11.shape, T01.shape)

        T00 = (T00[:, 1:] + T00[:, :-1])/2
        T11 = (T11[:, 1:] + T11[:, :-1])/2

        xmax = N * a / 2
        xs = np.linspace(-xmax, xmax, T00.shape[-1])
        
        x = np.zeros_like(T00)
        t = np.zeros_like(T00)
        x[:, :] = xs[np.newaxis, :]
        t[:, :] = tm[:, np.newaxis]

        tau2 = t**2 - x**2
        tau = np.zeros_like(tau2)
        tau[tau2>0] = np.sqrt(tau2[tau2>0])

        taumin = 2.5
        selfwd = (3*np.abs(x)<t)

        eps = 1/2*(T00 - T11 + np.sqrt((T00+T11)**2-4*T01**2))
        epsreg = 1/2*(T00 - T11 + sqrtreg((T00+T11)**2-4*T01**2))

        preg = T11 + epsreg - T00

        epsreg[tau<taumin] = 0
        preg[tau<taumin] = 0

        seltx = (tau>taumin) & selfwd

        lines = np.amax(epsreg)*np.exp(-2*(t-3*np.abs(x))**2)
        lines[tau<taumin] = 0

        # epsreg[~seltx] = 0
        # preg[~seltx] = 0


        plt.figure()
        plt.imshow(epsreg + lines, extent=(-xmax, xmax, 0, tm[-1]), origin='lower')
        #plt.imshow(preg, extent=(-xmax, xmax, 0, tm[-1]), origin='lower')
        #plt.scatter(preg[seltx], epsreg[seltx], s=1)

        # plt.colorbar()
        plt.title(r'$\varepsilon(\tau>2.5)$' + '     ' + f'{m/g=:.2f}')
        plt.xlabel('x')
        plt.ylabel('t')

        plt.tight_layout()

# FIG 4

plt.figure()

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i not in sel:
            continue


        tm, T00, midE = get_tsm(data[m, N, a, D, dt], 'T00')
        tm, T11, midp = get_tsm(data[m, N, a, D, dt], 'T11')
        tm, T01, midp = get_tsm(data[m, N, a, D, dt], 'T01')

        # print(T00.shape, T11.shape, T01.shape)

        T00 = (T00[:, 1:] + T00[:, :-1])/2
        T11 = (T11[:, 1:] + T11[:, :-1])/2

        xmax = N * a / 2
        xs = np.linspace(-xmax, xmax, T00.shape[-1])
        
        x = np.zeros_like(T00)
        t = np.zeros_like(T00)
        x[:, :] = xs[np.newaxis, :]
        t[:, :] = tm[:, np.newaxis]

        tau2 = t**2 - x**2
        tau = np.zeros_like(tau2)
        tau[tau2>0] = np.sqrt(tau2[tau2>0])

        taumin = 2.5
        selfwd = (3*np.abs(x)<t)

        eps = 1/2*(T00 - T11 + np.sqrt((T00+T11)**2-4*T01**2))
        epsreg = 1/2*(T00 - T11 + sqrtreg((T00+T11)**2-4*T01**2))

        preg = T11 + epsreg - T00

        epsreg[tau<taumin] = 0
        preg[tau<taumin] = 0


        seltx = (tau>taumin) & selfwd & (t<tm[-1])

        epsreg[~seltx] = 0
        preg[~seltx] = 0


        #plt.scatter(preg[seltx], epsreg[seltx], s=0.5, color=colors[i], label=f'{m/g=:.2f}', alpha=0.5)
        plt.scatter(preg[seltx], epsreg[seltx], s=0.5, label=f'{m/g=:.2f}', alpha=0.5)


plt.xlabel(r'$p$')
plt.ylabel(r'$\varepsilon$')

plt.ylim(0, 0.5)

plt.legend()
plt.tight_layout()



plt.show()
