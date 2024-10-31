import numpy as np
from operators import cLn, cLns
from operators import measure_local_observables, measure_local_observables_old, sumLn2, Ln
import yastn
import yastn.tn.mps as mps


def test_cLn():
    t, x0, a, v, Q, N = 3, 20, 0.2, 0.75, 1, 40
    aa = [cLn(n, t, x0, a, v, Q) for n in range(N)]
    bb = cLns(N, t, x0, a, v, Q)
    assert np.allclose(aa, bb)

    cbb = np.cumsum(bb)
    cbb[1:] = cbb[:-1]
    cbb[0] = 0
    cbb = cbb[-1] - cbb
    for n in range(N):
        assert np.allclose(sum(bb[k] for k in range(n, N - 1)), cbb[n])

def test_measure_local_observables():
    t, a, g, m, v, Q, N = 3, 0.2, 0.2, 0.1, 0.75, 1, 40
    ops = yastn.operators.SpinlessFermions(sym='U1')
    I = ops.I()
    HI = mps.product_mpo(I, N=N)
    psi = mps.random_mps(HI, n=N//2, D_total=64, dtype='complex128')
    psi.canonize_(to='first').canonize_(to='last')

    oT00, oT11, oT01, oj0, oj1, onu, oLn = measure_local_observables_old(psi, t, a, g, m, v, Q, ops)
    T00, T11, T01, j0, j1, nu, Ln = measure_local_observables(psi, t, a, g, m, v, Q, ops)

    assert np.allclose(oT00, T00)
    assert np.allclose(oT11, T11)
    assert np.allclose(oT01, T01)
    assert np.allclose(oj0, j0)
    assert np.allclose(oj1, j1)
    assert np.allclose(onu, nu)
    assert np.allclose(oLn, Ln)


def test_sumLn(N):
    ops = yastn.operators.SpinlessFermions(sym='U1')
    for t, a, v, Q in [(2, 1, 1, 1), (4.5, 0.25, 0.3, 2.2)]:

        H1 = sumLn2(N, t=t, a=a, v=v, Q=Q, ops=ops)
        tmp = Ln(0, N, t=t, a=a, v=v, Q=Q, ops=ops)
        LL = tmp @ tmp
        for n in range(1, N-1):
            tmp = Ln(n, N, t=t, a=a, v=v, Q=Q, ops=ops)
            LL += tmp @ tmp
        print((H1 - LL).norm() / H1.norm())
        assert (H1 - LL).norm() < 1e-13 * H1.norm()



if __name__ == "__main__":
    test_cLn()
    test_measure_local_observables()
    test_sumLn(N=32)
