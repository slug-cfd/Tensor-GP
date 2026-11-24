"""Sec. 3: is the ASSEMBLED multi-D LGP solvable?  Compare, at ell = 12 Delta:
  - assembled 2D/3D LGP (tensor-degree border Q_m) solved in double, in emulated
    quad (34 digits) and in 60 digits (reference);
  - tensor-product LGP (1D factors) in double.
Report: polynomial-constraint residual and weight error vs that operator's own
60-digit solution.  V->P, target (1/2, 0[, 0]).
"""
import itertools, math, time
import numpy as np
from decimal import Decimal as D, getcontext, localcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, c_point_hp, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as _solve_dec

getcontext().prec = 60
T1 = c_point_hp


def solve_dec(A, b, prec):
    """Gaussian elimination with the entries rounded to, and the arithmetic carried out at, prec digits."""
    with localcontext() as ctx:
        ctx.prec = prec
        x = _solve_dec([[+v for v in r] for r in A], [+v for v in b])
        return [+v for v in x]


def build(R, L, m, Dim):
    off = list(range(-R, R+1))
    kv = {d: SE_C1_hp(d, L) for d in range(-2*R, 2*R+1)}          # 1D kernel table
    tg = [D('0.5')] + [D(0)]*(Dim-1)
    t1 = [{a: T1(D(a)-tg[k], L) for a in off} for k in range(Dim)]
    pts = list(itertools.product(off, repeat=Dim))
    N = len(pts)
    C = [[math.prod([kv[a[k]-b[k]] for k in range(Dim)], start=D(1)) for b in pts] for a in pts]
    t = [math.prod([t1[k][a[k]] for k in range(Dim)], start=D(1)) for a in pts]
    mons = list(itertools.product(range(m+1), repeat=Dim))           # Q_m
    P = [[math.prod([avg(e[k], a[k]) for k in range(Dim)], start=D(1)) for e in mons] for a in pts]
    q = [math.prod([pw(tg[k], e[k]) for k in range(Dim)], start=D(1)) for e in mons]
    Mn = len(mons)
    A = [C[i] + P[i] for i in range(N)] + [[P[i][e] for i in range(N)] + [D(0)]*Mn for e in range(Mn)]
    return A, t + q, N, P, q

def constraint_resid(w, P, q):
    w = np.array([float(v) for v in w]); Pn = np.array([[float(v) for v in r] for r in P])
    return np.max(np.abs(Pn.T @ w - np.array([float(v) for v in q])))

R, L = 2, 12.0; m = 2*R-1
print(f"GP{2*R+1}, ell = {L:.0f} Delta, Q_{m} border, V->P\n")
print(f"{'operator':>26} {'arith':>7} | {'constraint resid':>16} {'weight err':>11} {'cond(C)':>9}")
print("-"*80)
for Dim in (2, 3):
    t0 = time.time()
    A, b, N, P, q = build(R, L, m, Dim)
    ref = solve_dec(A, b, 60)[:N]
    refn = np.array([float(v) for v in ref])
    c1 = 6.239e9
    for lab, prec in (("double", None), ("quad", 34)):
        if prec is None:
            An = np.array([[float(v) for v in r] for r in A]); bn = np.array([float(v) for v in b])
            try:
                w = np.linalg.solve(An, bn)[:N]
            except np.linalg.LinAlgError:
                print(f"{'assembled '+str(Dim)+'D LGP':>26} {lab:>7} | singular"); continue
            wd = [D(repr(float(v))) for v in w]
        else:
            Aq = [[D(repr(float(v))) if False else v for v in r] for r in A]
            with localcontext() as ctx:
                ctx.prec = prec
                Aq = [[+v for v in r] for r in A]; bq = [+v for v in b]
            wd = solve_dec(Aq, bq, prec)[:N]
        wn = np.array([float(v) for v in wd])
        print(f"{'assembled '+str(Dim)+'D LGP':>26} {lab:>7} | {constraint_resid(wd, P, q):16.1e} "
              f"{np.max(np.abs(wn-refn)):11.1e} {c1**Dim:9.1e}")
    print(f"{'':>26} ({time.time()-t0:.0f}s)")

# tensor-product LGP from 1D factors, double, vs its own 60-digit solution
off = list(range(-R, R+1))
def lgp1d(target, prec):
    C = [[SE_C1_hp(a-b, L) for b in off] for a in off]
    t = [T1(D(a)-target, L) for a in off]
    P = [[avg(e, a) for e in range(m+1)] for a in off]
    q = [pw(target, e) for e in range(m+1)]
    A = [C[i]+P[i] for i in range(5)] + [[P[i][e] for i in range(5)]+[D(0)]*(m+1) for e in range(m+1)]
    if prec is None:
        return np.linalg.solve(np.array([[float(v) for v in r] for r in A]),
                               np.array([float(v) for v in t+q]))[:5]
    return np.array([float(v) for v in solve_dec(A, t+q, prec)[:5]])
for Dim in (2, 3):
    ws_d = [lgp1d(D('0.5'), None)] + [lgp1d(D(0), None)]*(Dim-1)
    ws_h = [lgp1d(D('0.5'), 60)] + [lgp1d(D(0), 60)]*(Dim-1)
    Wd = ws_d[0]; Wh = ws_h[0]
    for k in range(1, Dim):
        Wd = np.multiply.outer(Wd, ws_d[k]); Wh = np.multiply.outer(Wh, ws_h[k])
    print(f"{'tensor '+str(Dim)+'D LGP (FSA)':>26} {'double':>7} | {'(1D: ~1e-16)':>16} "
          f"{np.max(np.abs(Wd-Wh)):11.1e} {6.239e9:9.1e}")
