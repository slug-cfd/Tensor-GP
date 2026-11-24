"""Sec. 3: does the LGP border fix the fixed-ell/Delta order collapse of pure GP?
1D V->P, target x*=+1/2, cell averages on offsets -R..R.
  pure : w = C^{-1} t, normalized
  LGP  : bordered system [C P; P^T 0][w;lam] = [t;q], monomials of degree <= m
         (m = 2R-1 is the cap used in the paper: one GP degree of freedom left)
  LAG  : m = 2R, i.e. N constraints on N unknowns -> pure Lagrange, no GP left
Moment defects E_k = sum_a w_a <(a+.)^k> - x*^k; first nonzero E_k = order.
Weights solved in double AND in 120 digits (kernel included) to separate the
method from roundoff.
"""
import math
import numpy as np
from decimal import Decimal as D, getcontext
import _setup  # noqa: F401
from gptensor import weights_1d
from gptensor.kernels import mono_avg as avg1d
getcontext().prec = 120

def weights(R, L, kind, m=None, hp=False, xs=0.5):
    """1D weights at xs: 'pure' (normalized to sum one in float64) or 'lgp' with 1, x, ..., x^m, from a
    double-precision or a 120-digit solve."""
    w = weights_1d(R, L, xs, lgp=(kind != 'pure'), m=m, arith='hp' if hp else 'double', normalize=False)
    return w/w.sum() if kind == 'pure' else w


def moments(w, R, xs=0.5, kmax=7):
    off = np.arange(-R, R+1, dtype=float)
    return [w @ np.array([avg1d(k, c) for c in off]) - xs**k for k in range(kmax+1)]

def order_of(E, tol):
    for k, e in enumerate(E):
        if abs(e) > tol: return k
    return len(E)

# ---------------------------------------------------------------- moments
for R in (2, 3):
    print(f"\n=== R={R} (GP{2*R+1}), nominal order {2*R+1}.  E_k at fixed ell/Delta, 120-digit weights ===")
    print(f"  {'ell/D':>5} {'method':>12} | " + "".join(f"{'E'+str(k):>10}" for k in range(0, 2*R+3)) + " | order")
    for L in (3., 6., 12., 24.):
        for kind, m, lab in (('pure', None, 'pure GP'), ('lgp', 2*R-1, f'LGP m={2*R-1}'), ('lgp', 2*R, f'Lagr m={2*R}')):
            w = weights(R, L, kind, m, hp=True)
            E = moments(w, R, kmax=2*R+2)
            print(f"  {L:5.0f} {lab:>12} | " + "".join(f"{e:10.1e}" for e in E) + f" | {order_of(E, 1e-25)}")
        print()

# ---------------------------------------------------------------- double vs hp
print("\n=== Does double precision preserve LGP's moment conditions?  (max |E_k|, k<=m, and E_{m+1}) ===")
print(f"  {'R':>2} {'ell/D':>5} | {'pure dbl E1':>12} | {'LGP dbl max|E_0..m|':>20} {'LGP dbl E_m+1':>14} {'LGP hp E_m+1':>13} | {'w err dbl':>10}")
for R in (2, 3):
    m = 2*R-1
    for L in (6., 12., 24.):
        wp = weights(R, L, 'pure', hp=False)
        wd = weights(R, L, 'lgp', m, hp=False)
        wh = weights(R, L, 'lgp', m, hp=True)
        Ed = moments(wd, R, kmax=m+1); Eh = moments(wh, R, kmax=m+1)
        print(f"  {R:>2} {L:5.0f} | {moments(wp,R)[1]:12.1e} | {max(abs(e) for e in Ed[:m+1]):20.1e} "
              f"{Ed[m+1]:14.2e} {Eh[m+1]:13.2e} | {np.max(np.abs(wd-wh)):10.1e}")

# ---------------------------------------------------------------- convergence
print("\n=== 1D convergence at FIXED ell/Delta = 12, double-precision weights ===")
print("    f = sin(2 pi x), exact cell averages in, point value at x+h/2 out, L_inf")
f = lambda x: np.sin(2*np.pi*x); favg = lambda x, h: f(x)*np.sin(np.pi*h)/(np.pi*h)
Ns = (32, 64, 128, 256, 512, 1024, 2048)
for R in (2, 3):
    print(f"\n  R={R} (GP{2*R+1})")
    print(f"    {'N':>5} | {'pure GP':>10} {'rate':>6} | {'LGP m='+str(2*R-1):>10} {'rate':>6} | {'Lagr m='+str(2*R):>10} {'rate':>6}")
    W = {k: weights(R, 12., k2, m, hp=False) for k, k2, m in (('p', 'pure', None), ('l', 'lgp', 2*R-1), ('g', 'lgp', 2*R))}
    prev = {}
    for N in Ns:
        h = 1.0/N; xc = (np.arange(N)+0.5)*h; ub = favg(xc, h); ex = f(xc+0.5*h)
        row = f"    {N:>5} |"
        for k in ('p', 'l', 'g'):
            w = W[k]
            err = np.max(np.abs(sum(w[a+R]*np.roll(ub, -a) for a in range(-R, R+1)) - ex))
            r = '' if k not in prev else f"{math.log2(prev[k]/err):6.2f}"
            row += f" {err:10.2e} {r:>6} |"
            prev[k] = err
        print(row)
