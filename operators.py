import numpy as np
import yastn
import yastn.tn.mps as mps

def HXXZ(N, a, m, ops=None):
    """
    Hopping and mass term
    1305.3765 is using x, mu; where x = 1/(g a)^2; mu = 2m/g^2 a (and a=1?)
    """
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    terms =  [mps.Hterm(1 / (2 * a), [n, n + 1], [Sp, Sm]) for n in range(N - 1)]
    terms += [mps.Hterm(1 / (2 * a), [n, n + 1], [Sm, Sp]) for n in range(N - 1)]
    terms += [mps.Hterm(m, [n], [d[n % 2]]) for n in range(N)]
    HI = mps.product_mpo(I, N)
    H = mps.generate_mpo(HI, terms)
    return H

def Qp(n, t, n0, a, v, Q):
    return Q * max(1 - abs(n0 + v * t / a - n), 0)

def Qm(n, t, n0, a, v, Q):
    return Q * max(1 - abs(n0 - v * t / a - n), 0)

def dQn(n, t, n0, L0, a, v, Q):
    return L0 + sum(Qp(k, t, n0, a, v, Q) - Qm(k, t, n0, a, v, Q) for k in range(n + 1))

def set_n0(N):
    return (N - 1) / 2

def Ln(n, N, t, L0, a, v, Q, ops=None):
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    terms = [mps.Hterm((-1) ** k, [k], [d[k % 2]]) for k in range(n + 1)]
    terms.append(mps.Hterm(dQn(n, t, set_n0(N), L0, a, v, Q), [0], [I]))
    HI = mps.product_mpo(I, N)
    return mps.generate_mpo(HI, terms)

def sumLn2(N, t, L0, a, v, Q, ops=None):
    """ sum_{n=0}^{N-2} Ln^2 """
    n0 = set_n0(N)
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d[0] = d[0].add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d[1] = d[1].add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H = mps.Mpo(N)
    #
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    #
    for n in range(N):
        An = ((-1) ** n) * d[n % 2]
        Bn = (((-1) ** n) * 2 * (N - 1 - n)) * d[n % 2]
        hn = N - 1 - n + 2 * ((-1) ** n) * sum(dQn(k, t, n0, L0, a, v, Q) for k in range(n, N - 1))
        Cn = hn * d[n % 2]
        if n < N - 1:
            Cn = Cn + (dQn(n, t, n0, L0, a, v, Q) ** 2) * I
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

def boost(N, a, g, m, L0, ops=None):
    """ n (sp sm + sm sp) + n * Z + sum_{n=0}^{N-2} n Ln^2 """
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    terms =  [mps.Hterm(n / (2 * a), (n, n + 1), (Sp, Sm)) for n in range(N - 1)]
    terms += [mps.Hterm(n / (2 * a), (n, n + 1), (Sm, Sp)) for n in range(N - 1)]
    terms += [mps.Hterm(n * m, (n,), (d[n % 2],)) for n in range(N)]
    HI = mps.product_mpo(I, N)
    H1 = mps.generate_mpo(HI, terms)
    #
    I = I.add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d[0] = d[0].add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    d[1] = d[1].add_leg(axis=0, s=-1).add_leg(axis=2, s=1)
    #
    H2 = mps.Mpo(N)
    #
    # This encodes Hamiltonian of the form sum_{n<n'} A_n B_n' + sum_n C_n
    #
    e0 = a * g * g / 2
    for n in range(N):
        An = ((-1) ** n) * d[n % 2]
        Bn = ((-1) ** n) * (N - 1 - n) * (N - 2 + n) * d[n % 2]
        Cn = (N - 1 - n) * (N - 2 + n) * (((-1) ** n) * L0 + 0.5)
        Cn = Cn * d[n % 2]
        if n < N - 1:
            Cn = Cn + (n * L0 * L0) * I
        if n == 0:
            H2[n] = yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn}, common_legs=(1, 3))
        elif n == N - 1:
            H2[n] = yastn.block({(0, 2): Cn,
                                 (1, 2): Bn,
                                 (2, 2): I}, common_legs=(1, 3))
        else:
            H2[n] =  yastn.block({(0, 0): I, (0, 1): An, (0, 2): Cn,
                                             (1, 1): I,  (1, 2): Bn,
                                                         (2, 2): I}, common_legs=(1, 3))
    return [H1, e0 * H2]


def momentum_total(N, ops=None):
    I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
    terms =  [mps.Hterm(-1j, (n, n + 1, n + 2), (Sm, Z, Sp)) for n in range(N - 2)]
    terms += [mps.Hterm(+1j, (n, n + 1, n + 2), (Sp, Z, Sm)) for n in range(N - 2)]
    HI = mps.product_mpo(I, N)
    return mps.generate_mpo(HI, terms)

def momentum_n(n, N, ops=None):
    I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
    terms = [mps.Hterm(-1j, (n - 1, n, n + 1), (Sm, Z, Sp)),
             mps.Hterm(+1j, (n - 1, n, n + 1), (Sp, Z, Sm))]
    HI = mps.product_mpo(I, N)
    return mps.generate_mpo(HI, terms)

def measure_energy_per_site(psi, a, g, m, t, L0, v, Q, ops):
    e0 = a * g * g / 2
    Sp, Sm = ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    dd = {n: d[n % 2] for n in range(psi.N)}
    n0 = set_n0(psi.N)
    eSpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
    eSmSp = mps.measure_2site(psi, Sm, Sp, psi, bonds='r1')
    ed = mps.measure_1site(psi, dd, psi)
    edd = mps.measure_2site(psi, dd, dd, psi, bonds='<')
    #
    engs = np.zeros(psi.N, dtype=np.float64)
    for n in range(psi.N):
        engs[n] += (1 / (4 * a)) * (eSpSm.get((n-1, n), 0) + eSpSm.get((n, n+1), 0)).real
        engs[n] += (1 / (4 * a)) * (eSmSp.get((n-1, n), 0) + eSmSp.get((n, n+1), 0)).real
        engs[n] += m * ed[n].real
    for n in range(psi.N - 1):
        cst = dQn(n, t, n0, L0, a, v, Q)
        tmp = cst * cst
        tmp += 2 * cst * sum(((-1) ** k) * ed[k] for k in range(n + 1))
        tmp += sum(ed[k] for k in range(n + 1))
        tmp += 2 * sum(((-1) ** (k + l)) * edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
        engs[n] += e0 * tmp.real
    return engs


def measure_T_per_site(psi, a, g, m, L0, ops):
    e0 = a * g * g / 2
    I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    dd = {n: d[n % 2] for n in range(psi.N)}
    eSpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
    eSmSp = mps.measure_2site(psi, Sm, Sp, psi, bonds='r1')
    ed = mps.measure_1site(psi, dd, psi)
    edd = mps.measure_2site(psi, dd, dd, psi, bonds='<')
    #
    T00 = np.zeros(psi.N, dtype=np.float64)
    T11 = np.zeros(psi.N, dtype=np.float64)
    for n in range(psi.N):
        T00[n] += (1 / (4 * a)) * (eSpSm.get((n-1, n), 0) + eSpSm.get((n, n+1), 0)).real
        T00[n] += (1 / (4 * a)) * (eSmSp.get((n-1, n), 0) + eSmSp.get((n, n+1), 0)).real
        T00[n] += m * ed[n].real
        T11[n] += (1 / (4 * a)) * (eSpSm.get((n-1, n), 0) + eSpSm.get((n, n+1), 0)).real
        T11[n] += (1 / (4 * a)) * (eSmSp.get((n-1, n), 0) + eSmSp.get((n, n+1), 0)).real
    for n in range(psi.N - 1):
        tmp = L0 * L0
        tmp += 2 * L0 * sum(((-1) ** k) * ed[k] for k in range(n + 1))
        tmp += sum(ed[k] for k in range(n + 1))
        tmp += 2 * sum(((-1) ** (k + l)) * edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
        T00[n] += e0 * tmp.real
        T11[n] -= e0 * tmp.real

    T01 = np.zeros(psi.N - 2, dtype=np.float64)
    HI = mps.product_mpo(I, psi.N)
    for n in range(1, psi.N - 2):
        Pn = mps.generate_mpo(HI, [mps.Hterm(1, (n - 1, n, n + 1), (Sm, Z, Sp))])
        T01[n - 1] = (1 / a) * mps.vdot(psi, Pn, psi).imag

    return T00, T11, T01

def measure_Ln(psi, L0, ops):
    Zn = mps.measure_1site(psi, ops.z(), psi)
    Ln = np.zeros(psi.N, dtype=np.float64)
    for n in range(psi.N):
        Ln[n] = L0 + sum(Zn[k] + ((-1) ** k) for k in range(n + 1)).real / 2
    return Ln

def measure_currents(psi, a, ops):
    Sp, Sm, Z = ops.sp(), ops.sm(), ops.z()
    j0 = np.zeros(psi.N, dtype=np.float64)
    j1 = np.zeros(psi.N // 2, dtype=np.float64)
    Zn = mps.measure_1site(psi, Z, psi)
    for n in range(psi.N):
        j0[n] = (Zn[n].real + 1) / (2 * a)
    SpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
    for n in range(psi.N // 2):
        j1[n] = SpSm[2 * n, 2 * n + 1].imag / (2 * a)
    return j0, j1


# def measure_energy_per_bond(psi, a, g, m, t, L0, v, Q, ops):
#     e0 = a * g * g / 2
#     Sp, Sm = ops.sp(), ops.sm()
#     d = {0: Sp @ Sm, 1: Sm @ Sp}
#     dd = {n: d[n % 2] for n in range(psi.N)}
#     N = psi.N
#     n0 = set_n0(N)
#     eSpSm = mps.measure_2site(psi, Sp, Sm, psi, bonds='r1')
#     eSmSp = mps.measure_2site(psi, Sm, Sp, psi, bonds='r1')
#     ed = mps.measure_1site(psi, dd, psi)
#     edd = mps.measure_2site(psi, dd, dd, psi, bonds='<')
#     #
#     engs = np.zeros(N - 1, dtype=np.float64)
#     for n in range(N - 1):
#         engs[n] += (1 / (2 * a)) * (eSpSm[n, n+1] + eSmSp[n, n+1]).real

#     engs[0] += m * ed[0].real
#     engs[N-2] += m * ed[N-1].real

#     for n in range(1, N-1):
#         engs[n] += (m / 2) * ed[n].real
#         engs[n-1] += (m / 2) * ed[n].real

#     for n in range(N - 1):
#         cst = dQn(n, t, n0, L0, a, v, Q)
#         tmp = cst * cst
#         tmp += 2 * cst * sum(((-1) ** k) * ed[k] for k in range(n + 1))
#         tmp += sum(ed[k] for k in range(n + 1))
#         tmp += 2 * sum(((-1) ** (k + l)) * edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
#         if n == 0:
#             engs[n] += e0 * tmp.real
#         else:
#             engs[n-1] += (e0 / 2) * tmp.real
#             engs[n] += (e0 / 2) * tmp.real
#     return engs

def axial_vector_source(N, P, a, ops=None):
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    HI = mps.product_mpo(I, N)
    terms =  [mps.Hterm( np.exp(1j * P * n / a), (n, n + 1), (Sp, Sm)) for n in range(N - 1)]
    terms += [mps.Hterm(-np.exp(1j * P * n / a), (n + 1, n), (Sp, Sm)) for n in range(N - 1)]
    pp = mps.generate_mpo(HI, terms)
    terms =  [mps.Hterm(-np.exp(-1j * P * n / a), (n, n + 1), (Sp, Sm)) for n in range(N - 1)]
    terms += [mps.Hterm( np.exp(-1j * P * n / a), (n + 1, n), (Sp, Sm)) for n in range(N - 1)]
    ppdag = mps.generate_mpo(HI, terms)
    return (1 / (a * a)) * (pp @ ppdag)
