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

#

sel = [3, 4, 5, 6, 7]

plt.figure(figsize=(10, 5))

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i not in sel:
            continue
        tm, ee, mid = get_tsm(data[m, N, a, D, dt], 'Ln')
        line, = plt.plot(tm[10:-2], mid[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')
        if j == 2:
            line.set_label(f'{m/g=:.2f}')

plt.legend()
plt.xlabel('t')
plt.title('Ln')

fig, ax = plt.subplots(1, 3, figsize=(13, 6))

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i not in sel:
            continue

        tm, ee, midE = get_tsm(data[m, N, a, D, dt], 'T00')
        tm, ee, midp = get_tsm(data[m, N, a, D, dt], 'T11')
        ax[0].plot(tm[10:-2], midE[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')
        ax[1].plot(tm[10:-2], midp[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')
        ax[2].plot(midp[10:-2], midE[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')

ax[0].set_title('T00')
ax[1].set_title('T11')
ax[2].set_title('T00 versus T11')

ax[0].set_xlabel('t')
ax[1].set_xlabel('t')
ax[2].set_xlabel('T11')

plt.legend()

plt.tight_layout()

# FIG 6

fig, ax = plt.subplots(1, 2, figsize=(13, 6))

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i not in sel:
            continue

        ee = data[m, N, a, D, dt]["entropy_1"]
        eemid = ee[:, N // 2] - ee[0, N // 2]
        eerel = eemid/eemid[-1]


        tm, ee, midE = get_tsm(data[m, N, a, D, dt], 'T00')
        tm, ee, midp = get_tsm(data[m, N, a, D, dt], 'T11')
        ax[0].plot(eerel[10:-2], midE[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')
        ax[1].plot(eerel[10:-2], midp[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')

        # ax[0].plot(tm[10:-2], midE[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')
        # ax[1].plot(tm[10:-2], midp[10:-2], lines[j], color=colors[i], label=f'{m/g=:.2f}')


ax[0].set_title('T00')
ax[1].set_title('T11')

ax[0].set_xlabel(r'$S/S_{plateau}$')
ax[1].set_xlabel(r'$S/S_{plateau}$')

plt.legend()

plt.tight_layout()

# FIG 7 left

plt.figure()

i0 = -17

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        # if i not in sel:
        #     continue

        tm = data[m, N, a, D, dt]["time"]

        ee = data[m, N, a, D, dt]["entropy_1"]
        ee = ee - ee[0]

        xmax = N * a / 2
        xs = np.linspace(-xmax, xmax, ee.shape[-1])

        plt.plot(xs, ee[i0], color=colors[i], label=f'{m/g=:.2f}')




plt.legend()

plt.xlabel('x')
plt.ylabel('entanglement entropy cut at x')

plt.title('t={}'.format(tm[i0]))

# FIG 7 right

plt.figure()

t0 = 3.0

for j, (N, a, D, dt) in enumerate(NaDdt):
    for i, m in enumerate(ms):
        if i==0:
            continue
        # if i not in sel:
        #     continue

        tm = data[m, N, a, D, dt]["time"]

        ee = data[m, N, a, D, dt]["entropy_1"]
        ee = ee - ee[0]

        eemid = ee[:, N // 2] - ee[0, N // 2]

        eeratio = np.amax(ee, axis=1)/eemid

        plt.plot(tm[tm>t0], eeratio[tm>t0], color=colors[i], label=f'{m/g=:.2f}')




plt.legend()

plt.xlabel('t')
plt.ylabel('entanglement entropy max/mid')

plt.title('entanglement entropy concentration')


#

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
        tau = np.ones_like(tau2)
        tau[tau2>0] = np.sqrt(tau2[tau2>0])

        # ut = t/tau
        # ux = x/tau

        # eps = T00*ut*ut+T11*ux*ux-2*T01*ut*ux

        # eps[tau2<=4] = 0

        eps = 1/2*(T00 - T11 + np.sqrt((T00+T11)**2-4*T01**2))
        epsreg = 1/2*(T00 - T11 + sqrtreg((T00+T11)**2-4*T01**2))

        logepsreg = np.log(epsreg)

        eps[tau2<0] = 0
        epsreg[tau2<0] = 0
        logepsreg[tau2<0] = 0


        sqr = (T00+T11)**2-4*T01**2
        sqr[tau2<=0] = 0

        plt.figure()
        # plt.imshow(T00, extent=(-xmax, xmax, 0, tm[-1]), origin='lower')
        plt.imshow(epsreg, extent=(-xmax, xmax, 0, tm[-1]), origin='lower')
        # plt.colorbar()
        plt.title(f'{m/g=:.2f}')

        # plt.figure()
        # plt.plot(T00[i0])
        # plt.plot(T11[i0])
        # plt.plot(T01[i0])
        # plt.axhline(0)



# FIG 3 middle


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


        vt = 1
        vx = 2*T01/(T00 + T11 + sqrtreg((T00+T11)**2-4*T01**2))
        norm = np.sqrt(1 + vx**2)
        vt = vt/norm
        vx = vx/norm



        Xs = []
        Ys = []
        Us = []
        Vs = []

        Xsb = []
        Ysb = []
        Usb = []
        Vsb = []

        for i in range(0, len(tm), 16):
            for j in range(0, N//2, 8):
                t = tm[i]
                x = xs[j]
                if np.abs(t)>np.abs(x):
                    Xs.append(x)
                    Ys.append(t)
                    Us.append(vx[i, j])
                    Vs.append(vt[i, j])

                    Xsb.append(x)
                    Ysb.append(t)
                    Usb.append(x/np.sqrt(t**2 + x**2))
                    Vsb.append(t/np.sqrt(t**2 + x**2))

        plt.figure()
        plt.quiver(Xsb, Ysb, Usb, Vsb, color='r')
        plt.quiver(Xs, Ys, Us, Vs)


        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'{m/g=:.2f}')




# FIG. 3 right

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



        vt = 1
        vx = 2*T01/(T00 + T11 + sqrtreg((T00+T11)**2-4*T01**2))
        norm = np.sqrt(1 + vx**2)
        vt = vt/norm
        vx = vx/norm

        ratio = vx/vt
        ratio[tau2<=0] = 0
        ratio = np.clip(ratio, -1, 1)

        plt.plot(xs, ratio[i0], color=colors[i], label=f'{m/g=:.2f}')
        # plt.imshow(ratio, extent=(-xmax, xmax, 0, tm[-1]), origin='lower')
        # plt.colorbar

plt.plot([-30,30],[-1,1], color='r', alpha=1.0, label='boost invariant')

plt.legend()
plt.xlabel('x')
plt.ylabel('velocity')
plt.title('t={}'.format(tm[i0]))
plt.axvline(-10, color='k', alpha=0.25, linestyle='dashed')
plt.axvline(10, color='k', alpha=0.25, linestyle='dashed')

#

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



        vt = 1
        vx = 2*T01 / (T00 + T11 + sqrtreg((T00+T11)**2-4*T01**2))
        norm = np.sqrt(1 + vx**2)
        vt = vt/norm
        vx = vx/norm

        ratio = vx/vt
        ratio[tau2<=0] = 0
        ratio = np.clip(ratio, -1, 1)

        dn = 1
        dratio = (ratio[:, 127+dn] - ratio[:, 127-dn])/(xs[127+dn]-xs[127-dn])

        plt.plot(tm[10:], dratio[10:], color=colors[i], label=f'{m/g=:.2f}')


plt.plot(tm[10:], 1/tm[10:], color='r', label='boost invariant (1/t)')

plt.legend()
plt.xlabel('t')
plt.ylabel('velocity gradient at x=0')

plt.show()
