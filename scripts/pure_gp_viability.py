"""Sec. 3: pure GP (no LGP border) at a FIXED ell/Delta is only first order.

Moment defects E_k of the normalized pure-GP weights for the right face x_* = 1/2 (E_k = 0 means x^k is
reproduced exactly; the order of the operator is the index of the first nonzero E_k), and the 1D grid
convergence of the reconstruction of sin(2 pi x) from exact cell averages at ell/Delta = 6 and 12.
The weights are computed in double precision; cond(C_1) is printed in 120-digit arithmetic.
"""
import math
import numpy as np
import _setup  # noqa: F401
from gptensor import weights_1d, cond_C1
from gptensor.kernels import mono_avg as avg1d


def w1d(R, L, xs=0.5, normalize=True):
    w = weights_1d(R, L, xs, arith='double', normalize=False)
    return w/w.sum() if normalize else w


def moments(w, R, xs=0.5, kmax=6):
    off = np.arange(-R, R+1, dtype=float)
    return np.array([w @ np.array([avg1d(k, c) for c in off]) - xs**k for k in range(kmax+1)])


print("Moment defects E_k of the NORMALIZED pure-GP V->P weights (E_k=0 => degree k exact)")
print("order of the frozen operator = index of first nonzero E_k")
for R in (2, 3):
    print(f"\n  R={R} (GP{2*R+1}), nominal order {2*R+1}")
    print(f"    {'ell/D':>6} {'cond(C1)':>10} | " + "".join(f"{'E'+str(k):>10}" for k in range(1, 7)))
    for L in (3., 6., 12., 24., 48.):
        w = w1d(R, L)
        E = moments(w, R)
        print(f"    {L:6.0f} {float(cond_C1(R, L)):10.2e} | " + "".join(f"{E[k]:10.2e}" for k in range(1, 7)))

print("\n\n1D convergence of the PURE (normalized) GP reconstruction, V->P at x+h/2")
print("f = sin(2 pi x) on [0,1], exact cell averages in, point value out. ell/Delta FIXED.")
f    = lambda x: np.sin(2*np.pi*x)
favg = lambda x, h: f(x)*np.sin(np.pi*h)/(np.pi*h)
for R in (2, 3):
    for L in (6., 12.):
        w = w1d(R, L)
        print(f"\n  R={R} (GP{2*R+1}), ell={L:.0f}D, cond(C1)={float(cond_C1(R, L)):.1e}")
        print(f"    {'N':>6} {'Linf err':>12} {'rate':>7}")
        prev = None
        for N in (32, 64, 128, 256, 512, 1024, 2048):
            h = 1.0/N
            xc = (np.arange(N)+0.5)*h
            ub = favg(xc, h)
            got = sum(w[a+R]*np.roll(ub, -a) for a in range(-R, R+1))
            err = np.max(np.abs(got - f(xc+0.5*h)))
            r = '' if prev is None else f"{math.log2(prev/err):7.2f}"
            print(f"    {N:6d} {err:12.3e} {r:>7}")
            prev = err
