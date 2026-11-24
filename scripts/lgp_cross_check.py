"""Proposition 4 / Appendix C, Eq. (C.1a). Check Eq. (cross): for the 2D assembled LGP, the factored trial w = w1 (x) w2
satisfies P^T w = q, but C w - T has a component outside range(P).
GP5 (R=2), m=3, ell=3 Delta, V->P, targets x*=1/2, y*=0. 60-digit arithmetic."""
import numpy as np
from decimal import Decimal as D, getcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, c_point_hp, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as solve, solve_bordered
getcontext().prec = 60
R, L, m = 2, 3.0, 3
off = list(range(-R, R+1))
def T1(d): return c_point_hp(d, L)
C1 = [[SE_C1_hp(a-b, L) for b in off] for a in off]
P1 = [[avg(e, a) for e in range(m+1)] for a in off]
def lgp1(target):
    t = [T1(D(a)-target) for a in off]; q = [pw(target, e) for e in range(m+1)]
    A = [C1[i]+P1[i] for i in range(5)] + [[P1[i][e] for i in range(5)]+[D(0)]*(m+1) for e in range(m+1)]
    x = solve(A, t+q)
    return np.array([float(v) for v in x[:5]]), np.array([float(v) for v in x[5:]]), np.array([float(v) for v in t]), np.array([float(v) for v in q])
w1, l1, t1, q1 = lgp1(D('0.5')); w2, l2, t2, q2 = lgp1(D(0))
C = np.kron(np.array([[float(v) for v in r] for r in C1]), np.array([[float(v) for v in r] for r in C1]))
Pn = np.array([[float(v) for v in r] for r in P1]); P = np.kron(Pn, Pn)
T = np.kron(t1, t2); q = np.kron(q1, q2); w = np.kron(w1, w2)
print(f"second row  |P^T w - q|_max              = {np.max(np.abs(P.T@w - q)):.1e}")
res = C@w - T
Q, _ = np.linalg.qr(P)                      # orthonormal basis of range(P)
out = res - Q@(Q.T@res)
print(f"first row   |C w - T|_max                = {np.max(np.abs(res)):.2e}")
print(f"            component outside range(P)   = {np.max(np.abs(out)):.2e}   (must be 0 for w to solve the 2D system)")
cross = -np.kron(t1, Pn@l2) - np.kron(Pn@l1, t2) + np.kron(Pn@l1, Pn@l2)
print(f"Eq.(cross) reproduces C w - T to           {np.max(np.abs(res-cross)):.1e}")
Q1, _ = np.linalg.qr(Pn)
print(f"T^(1) outside range(P^(1)): {np.linalg.norm(t1-Q1@(Q1.T@t1)):.2e}   T^(2): {np.linalg.norm(t2-Q1@(Q1.T@t2)):.2e}")
print(f"|lambda^(1)| = {np.linalg.norm(l1):.2e}   |lambda^(2)| = {np.linalg.norm(l2):.2e}")
