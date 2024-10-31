import numpy as np
import yastn
import yastn.tn.mps as mps


def HNN(N, a, m, ops=None):
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    terms =  [mps.Hterm(-1j / (2 * a), [n, n + 1], [cp, cm]) for n in range(N - 1)]
    terms += [mps.Hterm( 1j / (2 * a), [n + 1, n], [cp, cm]) for n in range(N - 1)]
    terms += [mps.Hterm(m * (-1) ** n, n, d) for n in range(N)]
    H = mps.generate_mpo(I, terms, N=N)
    return H


def Qp(n, t, x0, a, v, Q):
    return Q * np.maximum(1 - np.abs(x0 + v * t / a - n), 0)


def Qm(n, t, x0, a, v, Q):
    return Q * np.maximum(1 - np.abs(x0 - v * t / a - n), 0)


def cLn(n, t, x0, a, v, Q):
    ns = np.arange(n + 1, dtype=np.float64)
    return np.sum(Qp(ns, t, x0, a, v, Q) - Qm(ns, t, x0, a, v, Q)) - (n + 1) // 2


def cLns(N, t, x0, a, v, Q):
    ns = np.arange(N, dtype=np.float64)
    return np.cumsum(Qp(ns, t, x0, a, v, Q) - Qm(ns, t, x0, a, v, Q)) - (ns + 1) // 2


def set_x0(N, a):
    return (N - 1) / 2


def Ln(n, N, t, a, v, Q, ops=None):
    I, d = ops.I(), ops.n()
    terms = [mps.Hterm(1, k, d) for k in range(n + 1)]
    terms.append(mps.Hterm(cLn(n, t, set_x0(N, a), a, v, Q)))
    return mps.generate_mpo(I, terms, N=N)


def sumLn2(N, t, a, v, Q, ops=None):
    """ sum_{n=0}^{N-2} Ln^2 """
    x0 = set_x0(N, a)
    ecLns = cLns(N, t, x0, a, v, Q)
    # r_ecLns = sum(ecLns[n] for k in range(n, N - 1))
    r_ecLns = np.cumsum(ecLns)
    r_ecLns[1:] = r_ecLns[:-1]
    r_ecLns[0] = 0
    r_ecLns = r_ecLns[-1] - r_ecLns
    #
    I, d = ops.I(), ops.n()
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d = d.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H = mps.Mpo(N)
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    for n in range(N):
        An = d
        Bn = (2 * (N - 1 - n)) * d
        hn = N - 1 - n + 2 * r_ecLns[n]
        Cn = hn * d
        if n < N - 1:
            Cn = Cn + (ecLns[n] ** 2) * I
        if n == 0:
            H[n] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
        elif n == N - 1:
            H[n] = yastn.block({(0, 2): Cn,
                                (1, 2): Bn,
                                (2, 2): I}, common_legs=(1, 3))
        else:
            H[n] =  yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                            (1, 1): I,  (1, 2): Bn,
                                                        (2, 2): I}, common_legs=(1, 3))
    return H


def measure_local_observables_old(psi, t, a, g, m, v, Q, ops):
    N = psi.N
    cp, cm, d = ops.cp(), ops.c(), ops.n()

    ecpcm = mps.measure_2site(psi, cp, cm, psi, bonds='r1r-1r2r-2')
    ed = mps.measure_1site(psi, d, psi)
    edd = mps.measure_2site(psi, d, d, psi, bonds='<')
    #
    T00 = np.zeros(N, dtype=np.float64)
    T11 = np.zeros(N, dtype=np.float64)
    for n in range(N):
        tmp = (-1j / (4 * a * a)) * (ecpcm.get((n-1, n), 0) + ecpcm.get((n, n+1), 0))
        tmp += (1j / (4 * a * a)) * (ecpcm.get((n, n-1), 0) + ecpcm.get((n+1, n), 0))
        T00[n] += tmp.real
        T00[n] += (m / a) * ((-1) ** n) * ed[n].real
        T11[n] += tmp.real
    for n in range(N - 1):
        cst = cLn(n, t, set_x0(N, a), a, v, Q)
        tmp = cst * cst
        tmp += (2 * cst + 1) * sum(ed[k] for k in range(n + 1))
        tmp += 2 * sum(edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
        T00[n] += (g * g / 2) * tmp.real
        T11[n] -= (g * g / 2) * tmp.real

    T01 = np.zeros(N - 2, dtype=np.float64)
    for n in range(N - 2):
        T01[n] = (-1j * (ecpcm[n, n+2] - ecpcm[n+2, n]) / (2 * a * a)).real

    j0 = np.zeros(N // 2, dtype=np.float64)
    j1 = np.zeros(N // 2, dtype=np.float64)
    nu = np.zeros(N // 2, dtype=np.float64)
    for n in range(N // 2):
        j0[n] = (ed[2 * n] + ed[2 * n + 1]).real / a
        j1[n] = (ecpcm[2 * n, 2 * n + 1] + ecpcm[2 * n + 1, 2 * n]).real / a
        nu[n] = (ed[2 * n] - ed[2 * n + 1]).real / a

    Ln = np.zeros(N, dtype=np.float64)
    for n in range(N):
        Ln[n] += cLn(n, t, set_x0(N, a), a, v, Q)
        Ln[n] += sum(ed[k] for k in range(n + 1)).real

    return T00, T11, T01, j0, j1, nu, Ln


def measure_local_observables(psi, t, a, g, m, v, Q, ops):
    N = psi.N
    I, cp, cm, d = ops.I(), ops.cp(), ops.c(), ops.n()
    ecLns = cLns(N, t, set_x0(N, a), a, v, Q)

    ecpcm = mps.measure_2site(psi, cp, cm, psi, bonds='r1r-1r2r-2')
    ed = {k: v.real for k, v in mps.measure_1site(psi, d, psi).items()}

    sum_dd = np.zeros(N, dtype=np.float64)  # (sum_k^n d_k) ** 2
    L, C, R = dd_mpo_elements(I, d)
    Hdd = mps.product_mpo(I, N=N)
    env = mps.Env(psi, [Hdd, psi])
    env.setup_(to='first')
    sum_dd[0] = ed[0]
    Hdd[0] = L
    for n in range(1, N):
        env.update_env_(n-1, to='last')
        Hdd[n] = R
        env.update_env_(n, to='first')
        sum_dd[n] = env.measure(bd=(n-1, n)).real
        Hdd[n] = C
    #
    T00 = np.zeros(N, dtype=np.float64)
    T11 = np.zeros(N, dtype=np.float64)
    for n in range(N):
        tmp = (-1j / (4 * a * a)) * (ecpcm.get((n-1, n), 0) + ecpcm.get((n, n+1), 0))
        tmp += (1j / (4 * a * a)) * (ecpcm.get((n, n-1), 0) + ecpcm.get((n+1, n), 0))
        T00[n] += tmp.real
        T11[n] += tmp.real
        T00[n] += (m / a) * ((-1) ** n) * ed[n]

    cum_ed = 0
    for n in range(N - 1):
        cum_ed += ed[n]
        tmp = ecLns[n] ** 2 + 2 * ecLns[n] * cum_ed + sum_dd[n]
        T00[n] += (g * g / 2) * tmp
        T11[n] -= (g * g / 2) * tmp

    T01 = np.zeros(N - 2, dtype=np.float64)
    for n in range(N - 2):
        T01[n] = (-1j * (ecpcm[n, n+2] - ecpcm[n+2, n]) / (2 * a * a)).real

    j0 = np.zeros(N // 2, dtype=np.float64)
    j1 = np.zeros(N // 2, dtype=np.float64)
    nu = np.zeros(N // 2, dtype=np.float64)
    for n in range(N // 2):
        j0[n] = (ed[2 * n] + ed[2 * n + 1]) / a
        j1[n] = (ecpcm[2 * n, 2 * n + 1] + ecpcm[2 * n + 1, 2 * n]).real / a
        nu[n] = (ed[2 * n] - ed[2 * n + 1]) / a

    Ln = np.zeros(N, dtype=np.float64)
    cum_ed = 0
    for n in range(N):
        cum_ed += ed[n]
        Ln[n] = ecLns[n] + cum_ed

    return T00, T11, T01, j0, j1, nu, Ln


def dd_mpo_elements(I, d):
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d = d.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    #
    An = d
    Bn = 2 * d
    Cn = d
    L = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
    R = yastn.block({(0, 2): Cn,
                        (1, 2): Bn,
                        (2, 2): I}, common_legs=(1, 3))
    C = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                (1, 1): I,  (1, 2): Bn,
                                            (2, 2): I}, common_legs=(1, 3))
    return L, C, R