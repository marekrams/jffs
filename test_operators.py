import yastn
from operators import sumLn2, Ln, boost


def test_sumLn(N):
    ops = yastn.operators.Spin12(sym='U1')
    for t, a, L0 in [(2, 1, 0), (4.5, 0.25, 2)]:

        H1 = sumLn2(N, t=t, L0=L0, a=a, v=1, Q=1, ops=ops)
        tmp = Ln(0, N, t=t, L0=L0, a=a, v=1, Q=1, ops=ops)
        LL = tmp @ tmp
        for n in range(1, N-1):
            tmp = Ln(n, N, t=t, L0=L0, a=a, v=1, Q=1, ops=ops)
            LL += tmp @ tmp
        print((H1 - LL).norm())
        assert (H1 - LL).norm() < 1e-13 * H1.norm()


def test_boost(N):
    ops = yastn.operators.Spin12(sym='U1')
    for a, g, m, L0 in [(1, 1, 1, 0), (0.25, 0.2, 0.3, 2)]:
        [H1, H2] = boost(N, a, g, m, L0, ops=ops)

        e0 = a * g * g / 2

        tmp = Ln(0, N, t=0, L0=L0, a=a, v=0, Q=1, ops=ops)
        LL = 0 * (tmp @ tmp)
        for n in range(1, N-1):
            tmp = Ln(n, N, t=0, L0=L0, a=a, v=0, Q=1, ops=ops)
            LL += n * (tmp @ tmp)
        LL = e0 * LL
        print((H2 - LL).norm())
        assert (H2 - LL).norm() < 1e-13 * H2.norm()


def test_dn():
    ops = yastn.operators.Spin12(sym='U1')
    I, Sp, Sm, Z = ops.I(), ops.sp(), ops.sm(), ops.z()

    assert ((I + Z) / 2 - (Sp @ Sm)).norm() < 1e-13
    assert ((I - Z) / 2 - (Sm @ Sp)).norm() < 1e-13


if __name__ == "__main__":
    test_sumLn(N=10)
    test_boost(N=10)
    test_dn()
