import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from run_evolution import folder_evol


plt.rcParams.update({'font.size':18})
plt.rcParams.update({'legend.fontsize':14})
plt.rcParams['figure.constrained_layout.use'] = True


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

#

sel = [3, 4, 5, 6, 7]

# modified FIG 6

plt.figure()

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):

        if i==0 or i>7:
            continue

        ee = data[m, N, a, D, dt]["entropy_1"]
        eemid = ee[:, N // 2] - ee[0, N // 2]
        eerel = eemid/eemid[-1]


        tm, ee, midE = get_tsm(data[m, N, a, D, dt], 'T00')
        if i<3:
            plt.plot(eerel[10:-2], midE[10:-2], '--', color=colors[i], label=f'{m/g=:.2f}', alpha=0.75)
        else:
            plt.plot(eerel[10:-2], midE[10:-2], '-', color=colors[i], label=f'{m/g=:.2f}')




plt.xlabel(r'$S/S_{plateau}$')
plt.ylabel(r'energy')

plt.legend()



plt.figure()

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):

        if i==0 or i>7:
            continue

        ee = data[m, N, a, D, dt]["entropy_1"]
        eemid = ee[:, N // 2] - ee[0, N // 2]
        eerel = eemid/eemid[-1]


        tm, ee, midp = get_tsm(data[m, N, a, D, dt], 'T11')
        if i<3:
            plt.plot(eerel[10:-2], midp[10:-2], '--', color=colors[i], label=f'{m/g=:.2f}', alpha=0.75)
        else:
            plt.plot(eerel[10:-2], midp[10:-2], '-', color=colors[i], label=f'{m/g=:.2f}')



plt.xlabel(r'$S/S_{plateau}$')
plt.ylabel(r'pressure')

plt.legend()



# max electric field


plt.figure()


maxLns = []

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        tm, ee, mid = get_tsm(data[m, N, a, D, dt], 'Ln')
        maxLns.append(np.amax(mid[:-2]))

plt.plot(ms[:-2], maxLns[:-2], 'o', alpha=0.75)

plt.xlabel('m/g')
plt.ylabel('max Ln')





plt.show()
