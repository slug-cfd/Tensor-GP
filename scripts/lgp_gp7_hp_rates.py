"""Sec. 3: capped LGP (m = 2R-1) 1D convergence at ell/Delta = 12 with double vs 120-digit weights.
Shows that the GP7 rate drop with double weights (6.56 -> 6.08) is round-off in the 1D solve
(cond(C_1) = 7.9e13), not the method: 120-digit weights give 6.98, 6.99, 6.96."""
import math
import numpy as np
import _setup  # noqa: F401
from gptensor import weights_1d

f = lambda x: np.sin(2*np.pi*x); favg = lambda x, h: f(x)*np.sin(np.pi*h)/(np.pi*h)
for R in (2, 3):
    m = 2*R-1
    wd = weights_1d(R, 12., 0.5, lgp=True, m=m, arith='double')
    wh = weights_1d(R, 12., 0.5, lgp=True, m=m, arith='hp')
    print(f"R={R}: max|w_dbl - w_hp| = {np.max(np.abs(wd-wh)):.2e}")
    print(f"   {'N':>5} | {'dbl err':>10} {'rate':>6} | {'hp err':>10} {'rate':>6}")
    prev = {}
    for N in (32, 64, 128, 256, 512, 1024):
        h = 1.0/N; xc = (np.arange(N)+0.5)*h; ub = favg(xc, h); ex = f(xc+0.5*h)
        row = f"   {N:>5} |"
        for k, w in (('d', wd), ('h', wh)):
            err = np.max(np.abs(sum(w[a+R]*np.roll(ub, -a) for a in range(-R, R+1)) - ex))
            r = '' if k not in prev else f"{math.log2(prev[k]/err):6.2f}"
            row += f" {err:10.2e} {r:>6} |"; prev[k] = err
        print(row)
