"""Sec. 3: pure GP with ell fixed in PHYSICAL units (ell/Delta = ell N grows with N).

1D grid convergence of the reconstruction of sin(2 pi x) at the right face from exact cell averages, with
ell = 0.05 on the unit domain: GP5 keeps its fifth order, and GP7 its seventh until cond(C_1) passes the
double-precision limit of Table 1 (ell/Delta = 16.8). The weights are computed in double precision;
cond(C_1) is printed in 120-digit arithmetic. The precision limits themselves are in thresholds_true.py.
"""
import math
import numpy as np
import _setup  # noqa: F401
from gptensor import weights_1d, cond_C1


def w1d(R, L, xs=0.5):
    w = weights_1d(R, L, xs, arith='double', normalize=False)
    return w/w.sum()


f=lambda x: np.sin(2*np.pi*x); favg=lambda x,h: f(x)*np.sin(np.pi*h)/(np.pi*h)
print("(a) PURE GP, ell fixed in PHYSICAL units (ell/Delta = Lphys*N grows with N)")
for R in (2,3):
    for Lphys in (0.05,):
        print(f"\n  R={R} (GP{2*R+1}), ell_phys={Lphys} (domain 1)")
        print(f"    {'N':>6} {'ell/D':>7} {'cond(C1)':>10} {'Linf':>12} {'rate':>7}")
        prev=None
        for N in (32,64,128,256,512):
            h=1.0/N; L=Lphys*N
            w=w1d(R,L); xc=(np.arange(N)+0.5)*h; ub=favg(xc,h)
            got=sum(w[a+R]*np.roll(ub,-a) for a in range(-R,R+1))
            err=np.max(np.abs(got-f(xc+0.5*h)))
            r='' if prev is None else f"{math.log2(prev/err):7.2f}"
            print(f"    {N:6d} {L:7.1f} {float(cond_C1(R,L)):10.2e} {err:12.3e} {r:>7}")
            prev=err
