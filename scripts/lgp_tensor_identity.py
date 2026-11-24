"""Proposition 1 and Proposition 4 in 2D: is the tensor product of 1D LGP weights the same operator as an
LGP bordered solve on the ASSEMBLED 2D box?  (Pure GP: yes, exactly.  LGP: ?)
2D box R=2, V->P, target (1/2, 0), 60-digit arithmetic throughout so that
conditioning cannot be the explanation for any difference.
"""
import itertools
import numpy as np
from decimal import Decimal as D, getcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, c_point_hp, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as solve, solve_bordered
getcontext().prec = 60

T1 = c_point_hp
def bordered(C, t, P, q):
    return solve_bordered(C, t, P, q)[0]

R, L, m = 2, 3.0, 3          # moderate ell: 2D cond ~ 8.5e9, harmless at 60 digits
off = list(range(-R, R+1)); xs, ys = D('0.5'), D(0)

# --- 1D factors (pure and LGP) ---
C1 = [[SE_C1_hp(a-b, L) for b in off] for a in off]
P1 = [[avg(e, a) for e in range(m+1)] for a in off]
def w1(target, lgp):
    t = [T1(D(a)-target, L) for a in off]
    if not lgp: return solve(C1, t)
    return bordered(C1, t, P1, [pw(target, e) for e in range(m+1)])

# --- assembled 2D box ---
pts = list(itertools.product(off, off))
C2 = [[SE_C1_hp(a[0]-b[0], L)*SE_C1_hp(a[1]-b[1], L) for b in pts] for a in pts]
t2 = [T1(D(a[0])-xs, L)*T1(D(a[1])-ys, L) for a in pts]

def as_array(x): return np.array([float(v) for v in x]).reshape(5, 5)
def tensor(wx, wy): return np.outer([float(v) for v in wx], [float(v) for v in wy])

print("Same operator?  max |W_assembled - outer(w_x, w_y)|, 60-digit arithmetic\n")

W_asm = as_array(solve(C2, t2))
W_ten = tensor(w1(xs, False), w1(ys, False))
print(f"  pure GP                              : {np.max(np.abs(W_asm - W_ten)):.2e}")

# 2D bordered with the TENSOR-degree border Q_m: monomials x^a y^b, a,b <= m
monQ = [(a, b) for a in range(m+1) for b in range(m+1)]
PQ = [[avg(a, p[0])*avg(b, p[1]) for (a, b) in monQ] for p in pts]
qQ = [pw(xs, a)*pw(ys, b) for (a, b) in monQ]
W_lgpQ = as_array(bordered(C2, t2, PQ, qQ))
W_lgpT = tensor(w1(xs, True), w1(ys, True))
print(f"  LGP, assembled 2D with Q_{m} border    : {np.max(np.abs(W_lgpQ - W_lgpT)):.2e}"
      f"   ({len(monQ)} constraints)")

# 2D bordered with the TOTAL-degree border P_m: a+b <= m
monP = [(a, b) for a in range(m+1) for b in range(m+1) if a+b <= m]
PP = [[avg(a, p[0])*avg(b, p[1]) for (a, b) in monP] for p in pts]
qP = [pw(xs, a)*pw(ys, b) for (a, b) in monP]
W_lgpP = as_array(bordered(C2, t2, PP, qP))
print(f"  LGP, assembled 2D with P_{m} border    : {np.max(np.abs(W_lgpP - W_lgpT)):.2e}"
      f"   ({len(monP)} constraints)")
print(f"\n  scale of the weights                 : {np.max(np.abs(W_lgpT)):.2f}")

# rank-1 test: an outer product has sigma_2 = 0
for lab, W in (("assembled pure GP", W_asm), ("assembled LGP, Q_3", W_lgpQ), ("assembled LGP, P_3", W_lgpP)):
    s = np.linalg.svd(W, compute_uv=False)
    print(f"  {lab:22s} sigma_2/sigma_1 = {s[1]/s[0]:.2e}")

# which polynomials does each operator reproduce exactly (2D, cell-avg data, point target)?
def repro(W, a, b):
    got = sum(W[i, j]*float(avg(a, off[i]))*float(avg(b, off[j])) for i in range(5) for j in range(5))
    return abs(got - float(pw(xs, a)*pw(ys, b)))
print("\n  exact reproduction of x^a y^b ?   (residual)")
print(f"  {'monomial':>9} | {'tensor LGP':>11} {'assembled Q3':>13} {'assembled P3':>13}")
for (a, b) in ((3, 3), (4, 0), (2, 2), (3, 1), (4, 1)):
    print(f"  {'x^%d y^%d'%(a,b):>9} | {repro(W_lgpT,a,b):11.1e} {repro(W_lgpQ,a,b):13.1e} {repro(W_lgpP,a,b):13.1e}")
