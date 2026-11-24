"""Sec. 3: pure GP (no border) in 2D and 3D, with ell fixed in physical units so that ell/Delta grows.

Factored weights (1D solves, applied by sum factorization) against assembled weights (the full (2R+1)^D
system, applied directly), both in float64, reconstructing sin(2 pi x) cos(2 pi y) [sin(2 pi z)] at the
face x + Delta/2 from exact cell averages on a periodic grid. Does the assembled route lose the order where
the factored one keeps it?
"""
import math
import numpy as np
import _setup  # noqa: F401
from gptensor import weights_1d, assembled_weights, apply_factored, apply_direct


def w_fac(R, L, xs, norm=True):
    w = weights_1d(R, L, xs, arith='double', normalize=False)
    return w/w.sum() if norm else w


def w_asm(R, L, targ, Dd, norm=True):
    """assemble the full (2R+1)^Dd system in float64 and solve it"""
    _, w = assembled_weights(R, L, tuple(targ), arith='double', normalize=False)
    if norm: w = w/w.sum()
    return w.reshape(*([2*R+1]*Dd))


# smooth periodic test field
def run(Dd, R, Lphys, Ns):
    out = []
    kk = 2*np.pi
    for N in Ns:
        h = 1.0/N; L = Lphys*N
        axes = [(np.arange(N)+0.5)*h]*Dd
        G = np.meshgrid(*axes, indexing='ij')
        sinc = (np.sin(np.pi*h)/(np.pi*h))**Dd
        if Dd == 2: fex = lambda X, Y: np.sin(kk*X)*np.cos(kk*Y)
        else:       fex = lambda X, Y, Z: np.sin(kk*X)*np.cos(kk*Y)*np.sin(kk*Z)
        ub = fex(*G)*sinc
        targ = [0.5]+[0.0]*(Dd-1)
        got = apply_factored(ub, [w_fac(R, L, targ[k]) for k in range(Dd)])     # sum factorization
        ga = apply_direct(ub, w_asm(R, L, targ, Dd))                              # direct sum
        shift = [G[0]+0.5*h]+[G[k] for k in range(1, Dd)]
        ex = fex(*shift)
        out.append((N, L, np.max(np.abs(got-ex)), np.max(np.abs(ga-ex))))
    return out


for Dd, R, Lphys, Ns in ((2, 2, 0.05, (32, 64, 128, 256)),
                         (3, 2, 0.05, (16, 32, 64, 96)),
                         (3, 3, 0.05, (16, 32, 64, 96))):
    print(f"\n{Dd}D, R={R} (GP{2*R+1}), PURE GP (normalized, no border), ell_phys={Lphys}")
    print(f"  {'N':>5} {'ell/D':>7} | {'factored err':>13} {'rate':>6} | {'assembled err':>14} {'rate':>6}")
    res = run(Dd, R, Lphys, Ns); pf = pa = None
    for N, L, ef, ea in res:
        rf = '' if pf is None else f"{math.log2(pf/ef):6.2f}"
        ra = '' if pa is None else f"{math.log2(pa/ea):6.2f}"
        print(f"  {N:>5} {L:7.1f} | {ef:13.3e} {rf:>6} | {ea:14.3e} {ra:>6}")
        pf, pa = ef, ea
