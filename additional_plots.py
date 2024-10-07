import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv

from run_evolution import folder_evol, folder_gs
from operators import dQn


# ADDITIONAL PLOTS:

# Energy profiles at t=45 for various lattice spacing (CHECK WITH ANALYTIC)
# energy profile detailed check (CHECK WITH ANALYTIC)
# energy value at x=0 as a function of time (CHECK WITH ANALYTIC)
# electric field value at x=0 as a function of time (CHECK WITH ANALYTIC)
# Electric field profile at t=45 (CHECK WITH ANALYTIC)
# J0 profile at t=45  (CHECK WITH ANALYTIC)
# J1 profile at t=45  (CHECK WITH ANALYTIC)
# charge current arrow plots
# J1 value at x=0 as a function of time  (CHECK WITH ANALYTIC)



g = 1 / 5
L0 = 0
D0 = 64
ms = [0 * g, 0.1 * g, 0.318 * g, 1 * g]
Nas =  [(100, 1.0), (200, 0.5), (400, 0.25)]

engs, ents = {}, {}
T00, T11, T01 = {}, {}, {}
j0, j1, Ln = {}, {}, {}


mb = g/np.sqrt(np.pi)

for m in ms:
    for N, a in Nas:
        folder = folder_evol(g, m, L0, a, N, 1, 1, D0)
        engs[m, N] = np.loadtxt(folder / "engs.txt", delimiter=";")
        ents[m, N] = np.loadtxt(folder / "ents.txt", delimiter=";")
        T00[m, N] = np.loadtxt(folder / "T00.txt", delimiter=";")
        T01[m, N] = np.loadtxt(folder / "T01.txt", delimiter=";")
        T11[m, N] = np.loadtxt(folder / "T11.txt", delimiter=";")
        j0[m, N] = np.loadtxt(folder / "j0.txt", delimiter=";")
        j1[m, N] = np.loadtxt(folder / "j1.txt", delimiter=";")
        Ln[m, N] = np.loadtxt(folder / "Ln.txt", delimiter=";")




# Energy profiles at t=45 for various lattice spacing


xticks = np.linspace(-50, 50, 11)

for k in [0, 2]:

    m = ms[k]
    k0 = 45*2

    plt.figure()

    for N, a in Nas:
        ee = engs[m, N][:, 1:] - engs[m, N][0, 1:]
        tm = engs[m, N][:, 0]
        ee1 = (ee[:, 0::2] + ee[:, 1::2]) /(2*a)  #  here we calculate mean of sites 2*l and 2*l+1
        ee2 = (ee[:, :-1] + ee[:, 1:])/(2*a)
        
        xs = np.linspace(-50, 50, N//2)
        xs2 = np.linspace(-50, 50, N-1)

        #print(tm[k0], ee1.shape, ee2.shape, T00[m, N][20, N//2]-T00[m, N][0, N//2], engs[m, N][20, N//2])

        plt.plot(xs, ee1[k0], label='a={}'.format(a))
        #plt.plot(xs2, ee2[k0], label='a={}'.format(a))


    t =tm[k0]
    plt.axvline(t, color='k', alpha=0.15)
    plt.axvline(-t, color='k', alpha=0.15)
    # plt.axhline(0, color='k')

    if k==0:
        subset = (np.abs(xs)<44)
        x = xs[subset]
        arg = np.sqrt(t**2 - x**2)

        enprofile = 0.5*mb**2*np.pi*( jv(0, mb*arg)**2 + (t**2 + x**2)/(t**2 - x**2) * jv(1, mb*arg)**2  )

        plt.plot(x, enprofile, color='k', linestyle='dashed', label='analytical')



    plt.legend()
    plt.title(f"Energy profile  {m/g=:0.2f} {t=:0.1f}")
    plt.xticks(xticks)

    plt.xlabel('x')
    plt.ylabel(f'Energy(x, {t=:0.1f})')

# energy profile detailed check

m = 0
k0 = 45*2
N, a = Nas[-1]

ee = engs[m, N][:, 1:] - engs[m, N][0, 1:]
tm = engs[m, N][:, 0]
ee1 = (ee[:, 0::2] + ee[:, 1::2]) /(2*a)  #  here we calculate mean of sites 2*l and 2*l+1

xs = np.linspace(-50, 50, N//2)
subset = (np.abs(xs)<35)

plt.figure()

plt.plot(xs[subset], ee1[k0][subset], label='m/g=0.00')

t = tm[k0]
x = xs[subset]
arg = np.sqrt(t**2 - x**2)

enprofile = 0.5*mb**2*np.pi*( jv(0, mb*arg)**2 + (t**2 + x**2)/(t**2 - x**2) * jv(1, mb*arg)**2  )

plt.plot(x, enprofile, color='k', linestyle='dashed', label='analytical')

plt.legend()
plt.title(f"Energy profile detailed check {m/g=:0.2f} {t=:0.1f}")
plt.xticks(xticks)

plt.xlabel('x')
plt.ylabel(f'Energy(x, {t=:0.1f})')



# energy value at x=0 as a function of time
# eliminate last time point

plt.figure()

for m in ms:
    N, a = Nas[-1]

    ee = engs[m, N][:, 1:] - engs[m, N][0, 1:]
    tm = engs[m, N][:, 0]
    ee1 = (ee[:, 0::2] + ee[:, 1::2]) /(2*a)  #  here we calculate mean of sites 2*l and 2*l+1

    NN = ee1.shape[1]
    enmid = (ee1[:, NN//2] + ee1[:, NN//2-1])/2

    plt.plot(tm[:-1], enmid[:-1], label=f'{m/g=:0.2f}')


ts = tm[:-1]
enmid = 0.5*mb**2 *np.pi * (jv(0, mb*ts)**2 + jv(1, mb*ts)**2)
plt.plot(ts, enmid, color='k', linestyle='dashed', label='analytical')


plt.legend()
plt.xlabel('t')
plt.ylabel('Energy(x=0, t)')
plt.title(f"Energy evolution x=0")



# electric field value at x=0 as a function of time
# eliminate last time point

plt.figure()

for m in ms:
    N, a = Nas[-1]


    ee = Ln[m, N][:, 1:]
    tm = Ln[m, N][:, 0]

    # dQ = np.zeros((len(tm), N), dtype=np.float64)
    # for iii, t in enumerate(tm):
    #     for n in range(N):
    #         dQ[iii, n] = dQn(n, t, (N - 1) / 2, L0, a, v=1, Q=1)
    ee = ee -1 # + dQ

    ee = (ee[:, 0::2] + ee[:, 1::2]) /2  #  here we calculate mean of sites 2*l and 2*l+1

    tm = engs[m, N][:, 0]
    ts = tm[:-1]

    NN = ee1.shape[1]
    enmid = (ee[:, NN//2] + ee[:, NN//2-1])/2

    plt.plot(ts, enmid[:-1], label=f'{m/g=:0.2f}')


plt.plot(ts, -(jv(0, mb*ts)), color='k', linestyle='dashed', label='bessel')

plt.axhline(0, color='k', alpha=0.15)

plt.legend()
plt.xlabel('t')
plt.ylabel('Electric field(x=0, t)')
plt.title('Electric field evolution x=0')


# Electric field profile

plt.figure()

for k in [0, 1, 2, 3]:

    m = ms[k]
    k0 = 45*2

    N, a = Nas[-1]

    ee = Ln[m, N][:, 1:]
    tm = Ln[m, N][:, 0]

    ee = ee -1 # + dQ

    ee = (ee[:, 0::2] + ee[:, 1::2]) /2  #  here we calculate mean of sites 2*l and 2*l+1

    xs = np.linspace(-50, 50, N//2)

    plt.plot(xs, ee[k0], label=f'{m/g=:0.2f}')

x = np.linspace(-45, 45, 100)
plt.plot(x, -(jv(0, mb*np.sqrt(t**2-x**2))), color='k', linestyle='dashed', label='bessel')

plt.axhline(0, color='k', alpha=0.15)


plt.legend()
plt.xlabel('t')
plt.ylabel(f'Electric field(x, {t=:0.1f})')
plt.title(f"Electric field profile {t=:0.1f}")


# charge density profile

plt.figure()

for k in [0, 1, 2, 3]:

    m = ms[k]
    k0 = 45*2

    N, a = Nas[-1]

    J0 = j0[m, N][:, 1:]
    J0 = (J0[:, 0::2] + J0[:, 1::2]) /2
    J0 = J0 - J0[0]

    xs = np.linspace(-50, 50, N//2)

    plt.plot(xs, J0[k0], label=f'{m/g=:0.2f}')

x = np.linspace(-45, 45, 100)
j0anal = mb*x/np.sqrt(t**2 - x**2) * jv(1,mb*np.sqrt(t**2 - x**2))*a

plt.plot(x, -4*j0anal, color='k', linestyle='dashed', label='-4*bessel')


plt.legend()
plt.xlabel('t')
plt.ylabel(f'J0(x, {t=:0.1f})')
plt.title(f"J0 profile {t=:0.1f}")

# charge current profile

plt.figure()

for k in [0, 1, 2, 3]:

    m = ms[k]
    k0 = 45*2

    N, a = Nas[-1]

    J1 = j1[m, N][:, 1:]
    J1 = J1 - J1[0]

    xs = np.linspace(-50, 50, N//2)

    plt.plot(xs, J1[k0], label=f'{m/g=:0.2f}')


x = np.linspace(-45, 45, 100)
j1anal = mb*t/np.sqrt(t**2 - x**2) * jv(1,mb*np.sqrt(t**2 - x**2))*a

plt.plot(x, 2*j1anal, color='k', linestyle='dashed', label='2*bessel')

plt.legend()
plt.xlabel('t')
plt.ylabel(f'J1(x, {t=:0.1f})')
plt.title(f"J1 profile {t=:0.1f}")


# arrow plot of charge current


for k in [0, 2]:

    m = ms[k]

    N, a = Nas[-1]

    J0 = j0[m, N][:, 1:]
    J1 = j1[m, N][:, 1:]

    J0 = (J0[:, 0::2] + J0[:, 1::2]) /2

    J0 = J0 - J0[0]
    J1 = J1 - J1[0]

    # print(J0.shape, J1.shape)


    xs = np.linspace(-50, 50, N//2)
    ts = j0[m, N][:, 0]


    Xs = []
    Ys = []
    Us = []
    Vs = []

    for i in range(0, len(ts), 4):
        for j in range(0, N//2, 4):
            t = ts[i]
            x = xs[j]
            if np.abs(t)>np.abs(x):
                Xs.append(x)
                Ys.append(t)
                Us.append(J1[i, j])
                Vs.append(J0[i, j])


    plt.figure()
    plt.quiver(Xs, Ys, Us, Vs)


    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'charge current {m/g=:0.2f}')



# J1 value at x=0 as a function of time
# eliminate last time point

plt.figure()

for m in ms:
    N, a = Nas[-1]

    J1 = j1[m, N][:, 1:]
    J1 = J1 - J1[0]

    NN = J1.shape[1]
    J1mid = J1[:, NN//2]

    tm = j1[m, N][:, 0]


    plt.plot(tm, J1mid, label=f'{m/g=:0.2f}')


j1anal = mb*jv(1,mb*tm)*a
plt.plot(tm, 2*j1anal, color='k', linestyle='dashed', label='2*bessel')

plt.legend()
plt.xlabel('t')
plt.ylabel('J1(x=0, t)')
plt.title(f"J1 evolution x=0")



plt.show()
