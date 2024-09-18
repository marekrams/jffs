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
    terms =  [mps.Hterm(1 / (2 * a), (n, n + 1), (Sp, Sm)) for n in range(N - 1)]
    terms += [mps.Hterm(1 / (2 * a), (n, n + 1), (Sm, Sp)) for n in range(N - 1)]
    terms += [mps.Hterm(m, (n,), (d[n % 2],)) for n in range(N)]
    HI = mps.product_mpo(I, N)
    H = mps.generate_mpo(HI, terms)
    return H

def Qp(n, t, x0, v, Q):
    return Q * max(1 - abs(x0 + v * t - n), 0)

def Qm(n, t, x0, v, Q):
    return Q * max(1 - abs(x0 - v * t - n), 0)

def dQn(n, t, x0, v, Q):
    return sum(Qp(k, t, x0, v, Q) - Qm(k, t, x0, v, Q) for k in range(n + 1))

def hn(n, t, N, x0, L0, v, Q):
    return N - 1 - n + 2 * ((-1) ** n) * sum(L0 + dQn(k, t, x0, v, Q) for k in range(n, N - 1))

def set_x0(N):
    return (N - 1) / 2

def sumLn2(N, t=0, L0=0, v=1, Q=1, ops=None):
    """ sum_{n=0}^{N-2} Ln^2 """
    x0 = set_x0(N)
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
        Cn = hn(n, t, N, x0, L0, v, Q) * d[n % 2]
        if n < N - 1:
            Cn = Cn + ((L0 + dQn(n, t, x0, v, Q)) ** 2) * I
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

def Ln(n, N, t=0, L0=0, v=1, Q=1, ops=None):
    I, Sp, Sm = ops.I(), ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    terms = [mps.Hterm(((-1) ** k), [k], [d[k % 2]]) for k in range(n + 1)]
    terms.append(mps.Hterm(L0 + dQn(n, t, set_x0(N), v, Q), [0], [I]))
    HI = mps.product_mpo(I, N)
    return mps.generate_mpo(HI, terms)

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

def measure_energy_per_site(psi, a, e0, m, t, L0, v, Q, ops):
    Sp, Sm = ops.sp(), ops.sm()
    d = {0: Sp @ Sm, 1: Sm @ Sp}
    dd = {n: d[n % 2] for n in range(psi.N)}
    x0 = set_x0(psi.N)
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
        cst = L0 + dQn(n, t, x0, v, Q)
        tmp = cst * cst
        tmp += 2 * cst * sum(((-1) ** k) * ed[k] for k in range(n + 1))
        tmp += sum(ed[k] for k in range(n + 1))
        tmp += 2 * sum(((-1) ** (k + l)) * edd[k, l] for k in range(n + 1) for l in range(k + 1, n + 1))
        engs[n] += e0 * tmp.real
    return engs
