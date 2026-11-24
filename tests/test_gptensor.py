"""Checks of the main mathematical statements of the paper. Run with pytest, or as a plain script:
    python tests/test_gptensor.py
"""
import os
import sys
from decimal import Decimal as D, localcontext

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gptensor as g                                                              # noqa: E402
from gptensor.kernels import c_avg, c_avg_hp, c_point, c_point_hp, mono_avg      # noqa: E402

G = float(np.sqrt(3/5)/2)        # Gauss point offset on a unit face


def test_kernels_float_vs_decimal():
    """The float64 closed forms agree with the 120-digit ones where float64 does not cancel."""
    with localcontext() as ctx:
        ctx.prec = 120
        for ell in (1.0, 3.0):
            for d in range(-4, 5):
                assert abs(c_avg(d, ell) - float(c_avg_hp(d, ell))) < 1e-13
                assert abs(c_point(d - 0.5, ell) - float(c_point_hp(D(d) - D('0.5'), ell))) < 1e-13


def test_cond_kronecker_identity():
    """Proposition 2: cond(C_1 (x) C_1) = cond(C_1)^2 (60-digit eigenvalues of the assembled 25 x 25)."""
    R, ell = 2, 3
    with mp.workdps(60):
        c1 = mp.matrix([[g.c_avg_mp(i - j, ell) for j in range(5)] for i in range(5)])
        C2 = mp.matrix(25, 25)
        for a in range(25):
            for b in range(25):
                C2[a, b] = c1[a // 5, b // 5]*c1[a % 5, b % 5]
        ev1 = [abs(v) for v in mp.eigsy(c1, eigvals_only=True)]
        ev2 = [abs(v) for v in mp.eigsy(C2, eigvals_only=True)]
        k1, k2 = max(ev1)/min(ev1), max(ev2)/min(ev2)
        assert abs(k2/k1**2 - 1) < mp.mpf(10)**-40
    assert abs(float(g.cond_C1(R, ell))/float(k1) - 1) < 1e-30


def test_table1_values():
    """Table 1: cond(C_1) at ell = 12 Delta and the factored double-precision limit for GP5."""
    assert abs(float(g.cond_C1(2, 12))/6.239e9 - 1) < 1e-3
    assert abs(float(g.cond_C1(3, 12))/7.896e13 - 1) < 1e-3
    assert abs(g.largest_ell(2, 1, g.EPS_DOUBLE) - 64.7) < 0.05
    assert abs(g.largest_ell(2, 3, g.EPS_QUAD) - 18.2) < 0.05


def test_factored_equals_assembled_pure_gp():
    """Proposition 1: for pure GP the factored weights are the assembled weights."""
    W = g.factored_weights(2, 3.0, (0.5, G, 0.0), arith='quad')
    _, Wa = g.assembled_weights(2, 3.0, (0.5, G, 0.0), arith='quad')
    assert np.max(np.abs(W - Wa.reshape(W.shape))) < 1e-15


def test_factored_lgp_reproduces_tensor_monomials():
    """The factored LGP (m = 2R-1 = 3) reproduces every x^a y^b with a, b <= 3 from cell averages."""
    x, y = 0.5, G
    W = g.factored_weights(2, 12.0, (x, y), lgp=True, arith='hp')
    for a in range(4):
        for b in range(4):
            got = sum(W[i, j]*mono_avg(a, i - 2)*mono_avg(b, j - 2) for i in range(5) for j in range(5))
            assert abs(got - x**a*y**b) < 1e-14


def test_factored_lgp_is_not_assembled_lgp():
    """Proposition 4: the factored and the assembled LGP differ (0.35 on weights of size 0.87 at 3 Delta)."""
    Wf = g.factored_weights(2, 3.0, (0.5, 0.0), lgp=True, arith='hp', digits=60)
    _, Wa = g.assembled_weights(2, 3.0, (0.5, 0.0), lgp=True, border='tensor', arith='hp', digits=60)
    assert abs(np.max(np.abs(Wa.reshape(5, 5) - Wf)) - 0.354) < 1e-3


def test_flat_limit():
    """The factored LGP tends to the degree-2R polynomial reconstruction, with a gap ~ (ell/Delta)^-2."""
    p = np.array([float(v) for v in g.poly_weights(2, 0.5)])
    gap = [np.max(np.abs(g.weights_1d(2, ell, 0.5, lgp=True, arith='hp') - p)) for ell in (24, 48)]
    assert gap[1] < 1e-3 and abs(gap[0]/gap[1] - 4) < 0.1


def test_sum_factorization():
    """Sum factorization gives the direct sum, at D(2R+1) instead of (2R+1)^D operations per point."""
    rng = np.random.default_rng(1)
    U = rng.standard_normal((12, 10, 8))
    ws = [g.weights_1d(2, 12.0, x, lgp=True) for x in (0.5, G, -G)]
    assert np.max(np.abs(g.apply_factored(U, ws) - g.apply_direct(U, g.outer(ws)))) < 1e-12
    assert g.cost_per_point(2, 3) == {'direct': 125, 'sum_factorized_sweep': 15, 'nested_single_cell': 155}


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'passed  {t.__name__}')
    print(f'all {len(tests)} tests passed')
