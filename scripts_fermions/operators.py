import numpy as np
import yastn
import yastn.tn.mps as mps


def HNN(N, a, m, ops=None):
    I, cp, cm, occ = ops.I(), ops.cp(), ops.c(), ops.n()
    terms =  [mps.Hterm(-1j / (2 * a), [n, n + 1], [cp, cm]) for n in range(N - 1)]
    terms += [mps.Hterm( 1j / (2 * a), [n + 1, n], [cp, cm]) for n in range(N - 1)]
    terms += [mps.Hterm(m * (-1) ** n, [n], [occ]) for n in range(N)]
    H = mps.generate_mpo(I, terms, N=N)
    return H


def Qp(n, t, x0, a, v, Q):
    return Q * np.maximum(1 - np.abs(x0 + v * t - n * a), 0)


def Qm(n, t, x0, a, v, Q):
    return Q * np.maximum(1 - np.abs(x0 - v * t - n * a), 0)


def cLn(n, t, x0, a, v, Q):
    ns = np.arange(n + 1, dtype=np.float64)
    return np.sum(Qp(ns, t, x0, a, v, Q) - Qm(ns, t, x0, a, v, Q)) - (n + 1) // 2


def set_x0(N, a):
    return a * (N - 1) / 2


def Ln(n, N, t, a, v, Q, ops=None):
    I, occ = ops.I(), ops.n()
    terms = [mps.Hterm(1, [m], [occ]) for m in range(n + 1)]
    terms.append(mps.Hterm(cLn(n, t, set_x0(N, a), a, v, Q), [0], [I]))
    return mps.generate_mpo(I, terms, N=N)


def sumLn2(N, t, a, v, Q, ops=None):
    """ sum_{n=0}^{N-2} Ln^2 """
    x0 = set_x0(N, a)
    I, d = ops.I(), ops.n()
    #
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d = d.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H = mps.Mpo(N)
    #
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    #
    for n in range(N):
        An = d
        Bn = (2 * (N - 1 - n)) * d
        hn = N - 1 - n + 2 * sum(cLn(k, t, x0, a, v, Q) for k in range(n, N - 1))
        Cn = hn * d
        if n < N - 1:
            Cn = Cn + (cLn(n, t, x0, a, v, Q) ** 2) * I
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


# def momentum_total(N, ops=None):
#     I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
#     terms =  [mps.Hterm(-1j, (n, n + 1, n + 2), (Sm, Z, Sp)) for n in range(N - 2)]
#     terms += [mps.Hterm(+1j, (n, n + 1, n + 2), (Sp, Z, Sm)) for n in range(N - 2)]
#     HI = mps.product_mpo(I, N)
#     return mps.generate_mpo(HI, terms)


# def momentum_n(n, N, ops=None):
#     I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
#     terms = [mps.Hterm(-1j, (n - 1, n, n + 1), (Sm, Z, Sp)),
#              mps.Hterm(+1j, (n - 1, n, n + 1), (Sp, Z, Sm))]
#     HI = mps.product_mpo(I, N)
#     return mps.generate_mpo(HI, terms)


# def measure_T_per_site(psi, a, g, m, ops):
#     e0 = a * g * g / 2
#     I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
#     d = {0: Sp @ Sm, 1: Sm @ Sp}
#     dd = {n: d[n % 2] for n in range(psi.N)}
#     eSpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
#     eSmSp = mps.measure_2site(psi, Sm, Sp, psi, bonds='r1')
#     ed = mps.measure_1site(psi, dd, psi)
#     edd = mps.measure_2site(psi, dd, dd, psi, bonds='<')
#     #
#     T00 = np.zeros(psi.N, dtype=np.float64)
#     T11 = np.zeros(psi.N, dtype=np.float64)
#     for n in range(psi.N):
#         T00[n] += (1 / (4 * a)) * (eSpSm.get((n-1, n), 0) + eSpSm.get((n, n+1), 0)).real
#         T00[n] += (1 / (4 * a)) * (eSmSp.get((n-1, n), 0) + eSmSp.get((n, n+1), 0)).real
#         T00[n] += m * ed[n].real
#         T11[n] += (1 / (4 * a)) * (eSpSm.get((n-1, n), 0) + eSpSm.get((n, n+1), 0)).real
#         T11[n] += (1 / (4 * a)) * (eSmSp.get((n-1, n), 0) + eSmSp.get((n, n+1), 0)).real
#     for n in range(psi.N - 1):
#         tmp = 0 * 0
#         tmp += 2 * 0 * sum(((-1) ** k) * ed[k] for k in range(n + 1))
#         tmp += sum(ed[k] for k in range(n + 1))
#         tmp += 2 * sum(((-1) ** (k + l)) * edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
#         T00[n] += e0 * tmp.real
#         T11[n] -= e0 * tmp.real

#     T01 = np.zeros(psi.N - 2, dtype=np.float64)
#     HI = mps.product_mpo(I, psi.N)
#     for n in range(1, psi.N - 2):
#         Pn = mps.generate_mpo(HI, [mps.Hterm(1, (n - 1, n, n + 1), (Sm, Z, Sp))])
#         T01[n - 1] = (1 / a) * mps.vdot(psi, Pn, psi).imag

#     return T00, T11, T01

# def measure_Ln(psi, ops):
#     Zn = mps.measure_1site(psi, ops.z(), psi)
#     Ln = np.zeros(psi.N, dtype=np.float64)
#     for n in range(psi.N):
#         Ln[n] = sum(Zn[k] + ((-1) ** k) for k in range(n + 1)).real / 2
#     return Ln

# def measure_currents(psi, a, ops):
#     Sp, Sm, Z = ops.sp(), ops.sm(), ops.z()
#     j0 = np.zeros(psi.N, dtype=np.float64)
#     j1 = np.zeros(psi.N // 2, dtype=np.float64)
#     Zn = mps.measure_1site(psi, Z, psi)
#     for n in range(psi.N):
#         j0[n] = (Zn[n].real + 1) / (2 * a)
#     SpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
#     for n in range(psi.N // 2):
#         j1[n] = SpSm[2 * n, 2 * n + 1].imag / (2 * a)
#     return j0, j1

