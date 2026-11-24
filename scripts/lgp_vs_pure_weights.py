"""Sec. 3: how close is the factored LGP to the (exact) assembled pure GP?
The assembled pure GP equals the tensor product of 1D pure-GP weights exactly, so
the gap is ||outer(LGP_1D...) - outer(pure_1D...)||, computed in 120 digits.
V->P, targets x*=+1/2 (normal) and 0 (tangential), cap m = 2R-1.
"""
import numpy as np
from decimal import Decimal as D, getcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, c_point_hp as T1, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as solve
getcontext().prec = 120

def w1d(R, L, target, lgp):
    off = list(range(-R, R+1)); n = len(off); m = 2*R-1
    C = [[SE_C1_hp(a-b, L) for b in off] for a in off]
    t = [T1(D(a)-target, L) for a in off]
    if not lgp:
        w = solve(C, t); s = sum(w); return np.array([float(v/s) for v in w])   # normalized
    P = [[avg(e, a) for e in range(m+1)] for a in off]
    q = [pw(target, e) for e in range(m+1)]
    A = [C[i]+P[i] for i in range(n)] + [[P[i][e] for i in range(n)]+[D(0)]*(m+1) for e in range(m+1)]
    return np.array([float(v) for v in solve(A, t+q)[:n]])

def outer(ws):
    W = ws[0]
    for w in ws[1:]: W = np.multiply.outer(W, w)
    return W

print("max |W_LGP,factored - W_pure,assembled|  (120-digit; pure GP normalized)")
print("relative = divided by max |W_pure|\n")
print(f"{'GP':>4} {'ell/D':>6} | {'1D, x*=1/2':>11} {'1D, x*=0':>10} | {'2D abs':>9} {'2D rel':>9} | {'3D abs':>9} {'3D rel':>9}")
print("-"*82)
for R in (2, 3):
    for L in (3., 6., 12., 24., 48.):
        px, py = w1d(R, L, D('0.5'), False), w1d(R, L, D(0), False)
        lx, ly = w1d(R, L, D('0.5'), True),  w1d(R, L, D(0), True)
        r = []
        for Dim in (2, 3):
            Wp = outer([px] + [py]*(Dim-1)); Wl = outer([lx] + [ly]*(Dim-1))
            d = np.max(np.abs(Wl-Wp)); r += [d, d/np.max(np.abs(Wp))]
        print(f"{'GP'+str(2*R+1):>4} {L:6.0f} | {np.max(np.abs(lx-px)):11.2e} {np.max(np.abs(ly-py)):10.2e} | "
              f"{r[0]:9.2e} {r[1]:9.2e} | {r[2]:9.2e} {r[3]:9.2e}")
    print()

print("Scaling of the gap with ell (successive exponents, 1D normal target):")
for R in (2, 3):
    Ls = [6., 12., 24., 48.]
    g = [np.max(np.abs(w1d(R, L, D('0.5'), True)-w1d(R, L, D('0.5'), False))) for L in Ls]
    print(f"  GP{2*R+1}: " + "  ".join(f"{np.log2(g[i]/g[i+1]):.2f}" for i in range(3)) + "   => gap ~ ell^-p")
