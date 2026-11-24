"""Quick start: factored GP5 weights in 3D, their conditioning, and the reconstruction of a smooth field at
the nine Gauss points of a cell face by sum factorization.

    python examples/quickstart.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gptensor as g                                                     # noqa: E402

R, ell = 2, 12.0                          # GP5, ell = 12 Delta
gp = np.sqrt(3/5)/2                       # Gauss points of a unit face: 0, +-sqrt(3/5)/2
print(f"GP{2*R+1} at ell = {ell:g} Delta")
print(f"  cond(C_1)                  = {float(g.cond_C1(R, ell)):.3e}   (the only matrix the factored route solves)")
print(f"  cond(assembled 3D box)     = {float(g.cond_assembled_box(R, ell, 3)):.3e}   (= cond(C_1)^3)")
print(f"  largest ell/Delta in double: factored {g.largest_ell(R, 1, g.EPS_DOUBLE):.1f}, "
      f"assembled 3D {g.largest_ell(R, 3, g.EPS_DOUBLE):.1f}")

# 1D factored-LGP weights (m = 2R - 1 = 3) for the face x = 1/2 and the three tangential Gauss points
w_face = g.weights_1d(R, ell, 0.5, lgp=True)
w_tan = {y: g.weights_1d(R, ell, y, lgp=True) for y in (-gp, 0.0, gp)}
print(f"\n  1D LGP weights at x = 1/2 : {np.array2string(w_face, precision=6)}")

# cell averages of f = sin(2 pi x) cos(2 pi y) sin(2 pi z) on a periodic N^3 grid
N = 48; h = 1.0/N
xc = (np.arange(N)+0.5)*h
X, Y, Z = np.meshgrid(xc, xc, xc, indexing='ij')
k = 2*np.pi; sinc = np.sin(np.pi*h)/(np.pi*h)
f = lambda x, y, z: np.sin(k*x)*np.cos(k*y)*np.sin(k*z)
U = f(X, Y, Z)*sinc**3                    # exact cell averages

err, t_sweep, t_direct = 0.0, 0.0, 0.0
for y in (-gp, 0.0, gp):
    for z in (-gp, 0.0, gp):
        ws = [w_face, w_tan[y], w_tan[z]]
        t0 = time.perf_counter(); V = g.apply_factored(U, ws); t_sweep += time.perf_counter()-t0
        t0 = time.perf_counter(); Vd = g.apply_direct(U, g.outer(ws)); t_direct += time.perf_counter()-t0
        assert np.max(np.abs(V-Vd)) < 1e-12
        err = max(err, np.max(np.abs(V - f(X+0.5*h, Y+y*h, Z+z*h))))
c = g.cost_per_point(R, 3)
print(f"\n  N = {N}^3, max error at the 9 Gauss points of the face x + Delta/2: {err:.2e}")
print(f"  operations per point: direct {c['direct']}, sum factorization {c['sum_factorized_sweep']}")
print(f"  time (numpy, 9 points per cell): direct {t_direct:.2f} s, sum factorization {t_sweep:.2f} s")
