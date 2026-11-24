"""Table 2: three-dimensional GP5 pure-GP weights, precision x route.

Weights for the target (1/2, 0, 0), normalized by their sum, computed factored (5 x 5 solves and the
Kronecker product) and assembled (125 x 125 solve), with the kernel and the solve in double (float64) or in
quadruple precision (mpmath, 113-bit mantissa as in IEEE binary128). Error = max |w - w_true| against
120-digit factored weights (for pure GP the factored weights are exactly the assembled ones, so they are
the reference for both routes). Also reports cond(C_1) (120-digit) and whether Cholesky succeeds in double.
"""
import itertools
from decimal import Decimal as D, getcontext
import numpy as np
import mpmath as mp
import _setup  # noqa: F401
from gptensor import cond_C1
from gptensor.kernels import c_avg as SE_C1, c_point as SE_T1, c_avg_mp as SE_C1_mp, c_point_mp as SE_T1_mp
from gptensor.kernels import c_avg_hp as C1hp, c_point_hp as T1hp
from gptensor.linalg import solve_dec as solve_hp
getcontext().prec = 120

R = 2; n = 2*R+1; off = np.arange(-R, R+1.)
pts = list(itertools.product(range(-R, R+1), repeat=3))


def quad_weights(L):
    mp.mp.prec = 113
    kv = {d: SE_C1_mp(d, L) for d in range(-2*R, 2*R+1)}
    tx = {a: SE_T1_mp(mp.mpf(a)-mp.mpf(1)/2, L) for a in range(-R, R+1)}
    ty = {a: SE_T1_mp(a, L) for a in range(-R, R+1)}
    C1 = mp.matrix([[kv[a-b] for b in range(-R, R+1)] for a in range(-R, R+1)])
    wx = mp.lu_solve(C1, mp.matrix([tx[a] for a in range(-R, R+1)]))
    wy = mp.lu_solve(C1, mp.matrix([ty[a] for a in range(-R, R+1)]))
    sx, sy = sum(wx), sum(wy)
    wx = [v/sx for v in wx]; wy = [v/sy for v in wy]
    Wfac = np.array([[[float(wx[i]*wy[j]*wy[k]) for k in range(n)] for j in range(n)] for i in range(n)])
    Cb = mp.matrix([[kv[a[0]-b[0]]*kv[a[1]-b[1]]*kv[a[2]-b[2]] for b in pts] for a in pts])
    tb = mp.matrix([tx[a[0]]*ty[a[1]]*ty[a[2]] for a in pts])
    try:
        w = mp.lu_solve(Cb, tb); s = sum(w)
        Wasm = np.array([float(v/s) for v in w]).reshape(n, n, n)
    except ZeroDivisionError:
        Wasm = np.full((n, n, n), np.nan)
    return Wfac, Wasm


def double_weights(L):
    C = np.array([[SE_C1(a-b, L) for b in off] for a in off])
    wx = np.linalg.solve(C, np.array([SE_T1(a-0.5, L) for a in off])); wx /= wx.sum()
    wy = np.linalg.solve(C, np.array([SE_T1(a, L) for a in off]));     wy /= wy.sum()
    Wfac = np.einsum('i,j,k->ijk', wx, wy, wy)
    P = [tuple(float(v) for v in p) for p in pts]
    Cb = np.array([[np.prod([SE_C1(a[k]-b[k], L) for k in range(3)]) for b in P] for a in P])
    tb = np.array([SE_T1(a[0]-0.5, L)*SE_T1(a[1], L)*SE_T1(a[2], L) for a in P])
    Wasm = np.linalg.solve(Cb, tb); Wasm /= Wasm.sum()
    ch = []
    for M in (C, Cb):
        try: np.linalg.cholesky(M); ch.append('ok')
        except np.linalg.LinAlgError: ch.append('FAIL')
    return Wfac, Wasm.reshape(n, n, n), ch


print("3D GP5 pure GP: max weight error vs 120-digit weights")
print(f"{'ell/D':>5} {'cond(C1)':>9} {'cond(C1)^3':>10} | {'dbl fac':>9} {'dbl asm':>9} | {'quad fac':>9} {'quad asm':>9} | chol 1D/3D (dbl)")
for L in (3, 6, 12, 16, 24, 32, 64):
    A = [[C1hp(i-j, L) for j in range(-R, R+1)] for i in range(-R, R+1)]
    wx = np.array([float(v) for v in solve_hp(A, [T1hp(i-D('0.5'), L) for i in range(-R, R+1)])])
    wy = np.array([float(v) for v in solve_hp(A, [T1hp(D(i), L) for i in range(-R, R+1)])])
    wx /= wx.sum(); wy /= wy.sum()
    Wt = np.einsum('i,j,k->ijk', wx, wy, wy)
    c1 = float(cond_C1(R, L))
    df, da, ch = double_weights(float(L))
    qf, qa = quad_weights(L)
    e = [np.max(np.abs(W-Wt)) for W in (df, da, qf, qa)]
    print(f"{L:5d} {c1:9.2e} {c1**3:10.2e} | {e[0]:9.1e} {e[1]:9.1e} | {e[2]:9.1e} {e[3]:9.1e} | {ch[0]}/{ch[1]}", flush=True)
