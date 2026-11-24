"""Data for Fig. 2(a): cond(C_1) on a grid of ell/Delta for R = 1, 2, 3, in 120-digit arithmetic.
Writes output/cond_hp.npz, read by make_figures.py."""
import os
import numpy as np
import _setup
from gptensor import cond_C1
ells = np.concatenate([np.linspace(1.5, 14.0, 26)])
out = {}
for R in (1, 2, 3):
    out[f'R{R}'] = np.array([float(cond_C1(R, float(L))) for L in ells])
    print(R, out[f'R{R}'][[0, 10, -1]])
np.savez(os.path.join(_setup.OUT, 'cond_hp.npz'), ells=ells, **out)
