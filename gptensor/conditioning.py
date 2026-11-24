"""Condition numbers of the GP covariance matrices, computed reliably.

The 1D covariance C_1 of the 2R+1 cell averages has cond(C_1) ~ (ell/Delta)^(4R). On a box stencil in D
dimensions the assembled covariance is the Kronecker product of D copies of C_1, so that
cond(C) = cond(C_1)^D exactly, whereas the factored route only ever solves with C_1.

Both the kernel and the eigenvalues are computed in decimal arithmetic (120 digits by default). Two
float64 shortcuts give wrong answers and must not be used: the closed-form kernel cancels in float64 at
large ell, and numpy.linalg.cond saturates near 1e16. Both make the condition number appear to fall as
ell grows, which is impossible.
"""
from decimal import Decimal as D, localcontext

from .kernels import c_avg_hp
from .linalg import cond_sym_dec

EPS_DOUBLE = 2.22e-16      # float64 machine epsilon (2^-52), as quoted in the paper
EPS_QUAD = 1.93e-34        # IEEE binary128 machine epsilon (2^-112), as quoted in the paper


def C1_hp(R, ell):
    """The (2R+1) x (2R+1) covariance C_1 of the cell averages, in the current decimal context."""
    n = 2*R+1
    return [[c_avg_hp(i-j, ell) for j in range(n)] for i in range(n)]


def cond_C1(R, ell, digits=120):
    """cond_2(C_1) for GP of radius R (GP(2R+1)) at length scale ell (in units of Delta), as a Decimal.
    The kernel and the Jacobi eigensolve both run with `digits` significant digits."""
    with localcontext() as ctx:
        ctx.prec = digits
        return cond_sym_dec(C1_hp(R, ell))


def cond_assembled_box(R, ell, dim, digits=120):
    """cond_2 of the assembled (2R+1)^dim box covariance, = cond(C_1)^dim (exact identity)."""
    with localcontext() as ctx:
        ctx.prec = digits
        return cond_C1(R, ell, digits)**dim


def largest_ell(R, dim, eps, lo=1, hi=2000, iters=45, digits=120):
    """Largest ell/Delta in [lo, hi] for which cond(C_1)^dim <= 1/eps, by bisection on ell.
    dim = 1 is the factored route, dim = 2, 3 the assembled box. Returns hi if the bound holds there."""
    with localcontext() as ctx:
        ctx.prec = digits
        lo, hi = D(lo), D(hi); target = -D(repr(eps)).log10()
        for _ in range(iters):
            mid = (lo+hi)/2
            if dim*cond_C1(R, mid, digits).log10() <= target:
                lo = mid
            else:
                hi = mid
        return float(lo)
