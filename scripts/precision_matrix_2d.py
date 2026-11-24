"""Table 4 (Sec. 4.4): precision x route matrix for 2D GP5 weights at the length scales of the 2D Riemann problem.

Two-dimensional counterpart of precision_matrix.py (Table 2). GP5 (R=2), box stencil 5x5, V->P at the three
Gauss points of the face x = 1/2: targets (1/2, g), g = 0, +-sqrt(3/5)/2 (the largest error over the three
is reported). Routes: factored (5x5 solves, Kronecker product) and assembled (25x25, or 25+16 bordered).
Precisions: double (float64 kernel and solve) and quadruple (mpmath, 113-bit mantissa, kernel and solve).

Pure GP (normalized to sum 1): both routes are the same operator, so both are compared with the 120-digit
factored weights (max |w - w_true|, weights of size ~0.9).
LGP (m = 3 per direction): the factored and the assembled LGP are different operators (Proposition 4), so
each is compared with its own 120-digit solution; the assembled LGP error is relative to max |w_true|,
since its exact weights grow with ell/Delta.
"""
import math, itertools
import numpy as np
import mpmath as mp
from decimal import Decimal as D
from decimal import getcontext
import _setup  # noqa: F401
from gptensor import cond_C1
from gptensor.kernels import c_avg_hp as C1hp, c_point_hp, c_avg_mp, c_point_mp, c_avg as c_avg_dp, c_point as c_point_dp
from gptensor.linalg import solve_dec
getcontext().prec = 120

R, m = 2, 3
n = 2*R+1; off = list(range(-R, R+1))
pts = list(itertools.product(off, repeat=2)); mons = list(itertools.product(range(m+1), repeat=2))

# ---------- three arithmetics: kernel entries, monomial averages, linear solve ----------
class HP:                                    # 120-digit decimal (reference)
    half = D('0.5')
    @staticmethod
    def num(x): return D(x) if not isinstance(x, D) else x
    @staticmethod
    def C1(d, L): return C1hp(d, L)
    T1 = staticmethod(c_point_hp)
    solve = staticmethod(solve_dec)
    g = (D(3)/D(5)).sqrt()/2

class QP:                                    # quadruple: mpmath at 113 bits
    @staticmethod
    def num(x): return mp.mpf(x) if not isinstance(x, D) else mp.mpf(str(x))
    C1 = staticmethod(c_avg_mp)
    T1 = staticmethod(c_point_mp)
    @staticmethod
    def solve(A, b):
        try: return list(mp.lu_solve(mp.matrix(A), mp.matrix(b)))
        except ZeroDivisionError: return [mp.nan]*len(b)

class DP:                                    # double: float64
    @staticmethod
    def num(x): return float(x)
    C1 = staticmethod(c_avg_dp)
    T1 = staticmethod(c_point_dp)
    @staticmethod
    def solve(A, b):
        try: return list(np.linalg.solve(np.array(A, float), np.array(b, float)))
        except np.linalg.LinAlgError: return [float('nan')]*len(b)

def weights(K, L, tgt, lgp):
    """(factored, assembled) weights as float arrays [25] for the target tgt = (x, y) in arithmetic K."""
    num = K.num
    one = num(1); half = one/2
    avg = lambda e, c: ((num(c)+half)**(e+1)-(num(c)-half)**(e+1))/(e+1)
    pw = lambda x, e: one if e == 0 else x**e
    kv = {d: K.C1(d, L) for d in range(-2*R, 2*R+1)}
    t1 = [{a: K.T1(num(a)-tgt[k], L) for a in off} for k in range(2)]
    C1 = [[kv[a-b] for b in off] for a in off]
    w1 = []
    for k in range(2):
        t = [t1[k][a] for a in off]
        if lgp:
            A = [C1[i]+[avg(e, off[i]) for e in range(m+1)] for i in range(n)] \
              + [[avg(e, a) for a in off]+[num(0)]*(m+1) for e in range(m+1)]
            w = K.solve(A, t+[pw(tgt[k], e) for e in range(m+1)])[:n]
        else:
            w = K.solve(C1, t); s = sum(w); w = [v/s for v in w]
        w1.append(w)
    wf = np.array([float(w1[0][i]*w1[1][j]) for i in range(n) for j in range(n)])
    C = [[kv[a[0]-b[0]]*kv[a[1]-b[1]] for b in pts] for a in pts]
    t = [t1[0][a[0]]*t1[1][a[1]] for a in pts]
    if lgp:
        P = [[avg(e[0], a[0])*avg(e[1], a[1]) for e in mons] for a in pts]
        A = [C[i]+P[i] for i in range(len(pts))] + [[P[i][e] for i in range(len(pts))]+[num(0)]*len(mons) for e in range(len(mons))]
        w = K.solve(A, t+[pw(tgt[0], e[0])*pw(tgt[1], e[1]) for e in mons])[:len(pts)]
    else:
        w = K.solve(C, t); s = sum(w); w = [v/s for v in w]
    return wf, np.array([float(v) for v in w])

eps_d, eps_q = 2.0**-52, 2.0**-112
print("2D GP5, box stencil, face Gauss points: weight error against 120-digit weights")
print(f"1/eps: double {1/eps_d:.1e}, quadruple {1/eps_q:.1e}; assembled LGP error relative to max|w_true|")
print(f"{'ell/D':>5} {'cond(C1)':>9} {'cond(C1)^2':>10} | pure GP: {'dbl fac':>8} {'dbl asm':>8} {'quad fac':>8} {'quad asm':>8} |"
      f" LGP: {'dbl fac':>8} {'dbl asm':>8} {'quad fac':>8} {'quad asm':>8} | {'max|w| asm LGP':>14}")
for L in (12, 40, 90, 120, 400):
    mp.mp.prec = 113
    c1 = float(cond_C1(R, L))
    err = {}; wmax = 0.
    for lgp in (False, True):
        e = np.zeros(4)
        for gy in (D(0), HP.g, -HP.g):
            rf, ra = weights(HP, L, (D('0.5'), gy), lgp)
            sa = np.max(np.abs(ra)) if lgp else 1.0
            if lgp: wmax = max(wmax, np.max(np.abs(ra)))
            df, da = weights(DP, float(L), (0.5, float(gy)), lgp)
            qf, qa = weights(QP, L, (mp.mpf(1)/2, mp.mpf(str(gy))), lgp)
            e = np.fmax(e, [np.max(np.abs(df-rf)), np.max(np.abs(da-ra))/sa, np.max(np.abs(qf-rf)), np.max(np.abs(qa-ra))/sa])
            e = np.where(np.isnan([df.sum(), da.sum(), qf.sum(), qa.sum()]), np.inf, e)
        err[lgp] = e
    f = lambda v: f"{v:8.1e}"
    print(f"{L:5d} {c1:9.2e} {c1**2:10.2e} |          " + " ".join(f(v) for v in err[False]) + " |      "
          + " ".join(f(v) for v in err[True]) + f" | {wmax:14.2e}", flush=True)
