import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run_evolution import folder_evol
from analytical_results import fT00, fT01, fT11, fj0, fj1, fnu, fLn


g = 1 / 2
v, Q = 1, 1
D0, D, tol, method = 64, 64, 1e-6, '12site'
#
mg = [0, 0.1, 0.2, 0.318309886, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ms = [g * x for x in mg]
Nas =  [(512, 0.25)]
#
data = {}
#
for m in ms:
    for N, a in Nas:
        dt = min(1 / 32, a / 4)
        folder = folder_evol(g, m, a, N, v, Q, D0, dt, D, tol, method)
        data[m, N, a] = np.load(folder / f"results.npy", allow_pickle=True).item()


def get_tsm(signals, ev):
    tm = data[m, N, a]["time"]
    mask = tm > -1
    tm = tm[mask]
    ee = data[m, N, a][ev][mask]
    ee = ee - ee[0, :]
    ee = (ee[:, 0::2] + ee[:, 1::2]) / 2  # average over 2*n and 2*n+1
    mid = (ee[:, N//4] +ee[:, N//4-1])/2
    return tm, ee, mid

N, a = Nas[0]

plt.figure(figsize=(10, 5))

for i, m in enumerate(ms):
    tm, ee, mid = get_tsm(data[m, N, a], 'Ln')

    plt.plot(tm[10:-2], mid[10:-2], label='m/g={:.2f}'.format(m/g))

plt.legend()
plt.xlabel('t')
plt.title('Ln')

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

for i, m in enumerate(ms):
    tm, ee, midE = get_tsm(data[m, N, a], 'T00')

    tm, ee, midp = get_tsm(data[m, N, a], 'T11')

    ax[0].plot(tm[10:-2], midE[10:-2], label='m/g={:.2f}'.format(m/g))
    ax[1].plot(tm[10:-2], midp[10:-2], label='m/g={:.2f}'.format(m/g))
    ax[2].plot(midp[10:-2], midE[10:-2], label='m/g={:.2f}'.format(m/g))

ax[0].set_title('T00')
ax[1].set_title('T11')
ax[2].set_title('T00 versus T11')

ax[0].set_xlabel('t')
ax[1].set_xlabel('t')
ax[2].set_xlabel('T11')


plt.legend()




plt.show()

