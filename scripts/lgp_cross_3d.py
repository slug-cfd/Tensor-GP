"""Appendix C, Eq. (C.1b): three-dimensional version of lgp_cross_check.py and lgp_tensor_identity.py.

For the ASSEMBLED 3D LGP (tensor border x^a y^b z^c, a,b,c <= m), check that
  1. the factored trial w = w1 (x) w2 (x) w3 satisfies P^T w = q,
  2. C w - T is reproduced by the 7-term expansion of
         (T1 - P1 l1) (x) (T2 - P2 l2) (x) (T3 - P3 l3) - T1 (x) T2 (x) T3,
  3. C w - T has a component outside range(P), and each of the three terms with a
     single multiplier factor survives a projection that kills every other term,
  4. the assembled LGP weights differ from the factored ones, and are not rank one
     (singular values of the three unfoldings).
GP5 (R=2), m=3, ell=3 Delta, V->P, target (1/2, 0, 0). 60-digit arithmetic.
"""
import itertools
import numpy as np
from decimal import Decimal as D, getcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, c_point_hp, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as solve, solve_bordered
getcontext().prec = 60
R, L, m = 2, 3.0, 3
off = list(range(-R, R+1)); n1 = len(off)
tgt = (D('0.5'), D(0), D(0))


def T1(d): return c_point_hp(d, L)
bordered = solve_bordered
f = lambda v: np.array([float(x) for x in v])

# --- 1D factors ---
C1 = [[SE_C1_hp(a-b, L) for b in off] for a in off]
P1 = [[avg(e, a) for e in range(m+1)] for a in off]
fac = []
for x in tgt:
    t = [T1(D(a)-x) for a in off]; q = [pw(x, e) for e in range(m+1)]
    w, l = bordered(C1, t, P1, q)
    fac.append((f(w), f(l), f(t), f(q)))
Cn = np.array([[float(v) for v in r] for r in C1]); Pn = np.array([[float(v) for v in r] for r in P1])
(w1, l1, t1, q1), (w2, l2, t2, q2), (w3, l3, t3, q3) = fac
k3 = lambda a, b, c: np.kron(np.kron(a, b), c)
C = k3(Cn, Cn, Cn); P = k3(Pn, Pn, Pn); T = k3(t1, t2, t3); q = k3(q1, q2, q3); w = k3(w1, w2, w3)

print(f"1. second row  |P^T w - q|_max               = {np.max(np.abs(P.T@w - q)):.1e}")
res = C@w - T
a = (t1, t2, t3); b = (Pn@l1, Pn@l2, Pn@l3)
exp7 = np.zeros_like(res)
for s in itertools.product((0, 1), repeat=3):          # s_d = 1 picks b_d = P^(d) lambda^(d)
    if sum(s) == 0: continue
    term = k3(*(b[d] if s[d] else a[d] for d in range(3)))
    exp7 += (-1)**sum(s) * term
print(f"2. 7-term expansion reproduces C w - T to      {np.max(np.abs(res-exp7)):.1e}"
      f"   (|C w - T|_max = {np.max(np.abs(res)):.2e})")
Qf, _ = np.linalg.qr(P)
print(f"3. component of C w - T outside range(P)      = {np.max(np.abs(res - Qf@(Qf.T@res))):.2e}")
Q1, _ = np.linalg.qr(Pn); Pi = Q1@Q1.T; Qp = np.eye(n1) - Pi
for d in range(3):
    proj = k3(*(Pi if e == d else Qp for e in range(3)))   # range(P) along d, complement elsewhere
    single = k3(*(b[e] if e == d else a[e] for e in range(3)))
    print(f"   projection keeping only the term with P^({d+1})lambda^({d+1}): "
          f"|proj(Cw-T)| = {np.linalg.norm(proj@res):.2e}, |proj(that term)| = {np.linalg.norm(proj@single):.2e}")

# --- assembled 3D LGP, 125 + 64 = 189 unknowns ---
pts = list(itertools.product(off, off, off))
mons = list(itertools.product(range(m+1), repeat=3))
CA = [[SE_C1_hp(p[0]-r[0], L)*SE_C1_hp(p[1]-r[1], L)*SE_C1_hp(p[2]-r[2], L) for r in pts] for p in pts]
TA = [T1(D(p[0])-tgt[0])*T1(D(p[1])-tgt[1])*T1(D(p[2])-tgt[2]) for p in pts]
PA = [[avg(e[0], p[0])*avg(e[1], p[1])*avg(e[2], p[2]) for e in mons] for p in pts]
qA = [pw(tgt[0], e[0])*pw(tgt[1], e[1])*pw(tgt[2], e[2]) for e in mons]
print(f"\n4. assembled 3D system: {len(pts)} + {len(mons)} = {len(pts)+len(mons)} unknowns")
wA, _ = bordered(CA, TA, PA, qA)
WA = f(wA).reshape(n1, n1, n1); WF = w.reshape(n1, n1, n1)
print(f"   max |W_assembled - W_factored| = {np.max(np.abs(WA-WF)):.3f},  scale max|W_factored| = {np.max(np.abs(WF)):.3f},"
      f"  max|W_assembled| = {np.max(np.abs(WA)):.3f}")
for d in range(3):
    U = np.moveaxis(WA, d, 0).reshape(n1, -1)
    s = np.linalg.svd(U, compute_uv=False)
    sF = np.linalg.svd(np.moveaxis(WF, d, 0).reshape(n1, -1), compute_uv=False)
    print(f"   mode-{d+1} unfolding sigma_2/sigma_1: assembled {s[1]/s[0]:.2e}, factored {sF[1]/sF[0]:.1e}")
