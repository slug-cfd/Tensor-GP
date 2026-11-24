"""GP reconstruction weights: one-dimensional, factored (Kronecker), and assembled.

All coordinates are in units of Delta. The cell averages sit on the offsets -R..R of a stencil of radius R
(GP(2R+1)), and the target is the point x_* (1D) or (x_*, y_*, ...) (D dimensions) relative to the center
cell, e.g., x_* = 1/2 for the right face and the face Gauss points (1/2, g) in 2D.

Pure GP solves C w = t_*. The LGP borders it with the cell averages of a set of monomials,

    [[C, P], [P^T, 0]] [w; lam] = [t_*; q_*],

so that the weights reproduce those monomials exactly. In 1D the monomials are 1, x, ..., x^m, and the
paper caps m at 2R - 1.

  factored : solve the 1D system once per direction and take the Kronecker product of the 1D weights.
             For pure GP this is exactly the assembled weights (in exact arithmetic). For LGP it is a
             different operator, the factored LGP, which reproduces every x^a y^b z^c with a, b, c <= m.
  assembled: build and solve the full multidimensional system on the stencil (box or diamond).

Arithmetic, selected by `arith`:
  'double'  float64 kernel, numpy.linalg.solve,
  'quad'    mpmath kernel and LU solve at `bits` bits of mantissa (113 = IEEE quadruple),
  'hp'      decimal kernel and Gaussian elimination at `digits` significant digits (reference).
Weights are returned as float64 numpy arrays; in 'quad' and 'hp' the products of the factored route are
formed in the working precision before rounding.
"""
import itertools
from decimal import Decimal as D, localcontext

import numpy as np
import mpmath as mp

from .kernels import (c_avg, c_point, mono_avg, c_avg_mp, c_point_mp, mono_avg_mp,
                      c_avg_hp, c_point_hp, mono_avg_hp, power_hp)
from .linalg import solve_dec, solve_bordered


# ------------------------------------------------------------------------------------- stencils
def stencil(R, dim, kind='box'):
    """Offsets of the cells of a stencil of radius R in `dim` dimensions, in lexicographic order.
    kind = 'box' (max |o_d| <= R, (2R+1)^dim cells, the stencil that factors) or
           'diamond' (sum |o_d| <= R, the stencil of the original GP-MOOD, which does not factor)."""
    pts = list(itertools.product(range(-R, R+1), repeat=dim))
    if kind == 'diamond':
        pts = [p for p in pts if sum(map(abs, p)) <= R]
    elif kind != 'box':
        raise ValueError(kind)
    return pts


def monomials(m, dim, kind='tensor'):
    """Exponent tuples of the border monomials x^a y^b ...: 'tensor' (Q_m, every exponent <= m, the set
    the factored LGP reproduces) or 'total' (P_m, total degree <= m)."""
    mons = list(itertools.product(range(m+1), repeat=dim))
    if kind == 'total':
        mons = [e for e in mons if sum(e) <= m]
    elif kind != 'tensor':
        raise ValueError(kind)
    return mons


# ------------------------------------------------------------------------------ arithmetics
def _ops(arith, bits):
    """(number, kernel c, kernel t, monomial average, power, solver) in the requested arithmetic."""
    if arith == 'double':
        def solve(A, b):
            return list(np.linalg.solve(np.array(A, float), np.array(b, float)))
        return float, c_avg, c_point, mono_avg, (lambda x, e: x**e), solve
    if arith == 'quad':
        def solve(A, b):
            return list(mp.lu_solve(mp.matrix(A), mp.matrix(b)))
        return (lambda v: mp.mpf(str(v)) if isinstance(v, D) else mp.mpf(v)), c_avg_mp, c_point_mp, \
            mono_avg_mp, (lambda x, e: x**e), solve
    if arith == 'hp':
        return (lambda v: D(str(v))), c_avg_hp, c_point_hp, mono_avg_hp, power_hp, solve_dec
    raise ValueError(arith)


def _system(pts, target, ell, lgp, mons, ops):
    """Covariance C, prediction vector t, and (for LGP) border P and right-hand side q on the points."""
    num, cK, tK, avg, pw, _ = ops
    dim = len(target)
    tgt = [num(x) for x in target]
    kv = {}
    def c(d):
        if d not in kv:
            kv[d] = cK(d, ell)
        return kv[d]
    one = num(1)
    def prod(vals):
        r = one
        for v in vals:
            r = r*v
        return r
    C = [[prod(c(a[k]-b[k]) for k in range(dim)) for b in pts] for a in pts]
    t = [prod(tK(num(a[k])-tgt[k], ell) for k in range(dim)) for a in pts]
    if not lgp:
        return C, t, None, None
    P = [[prod(avg(e[k], a[k]) for k in range(dim)) for e in mons] for a in pts]
    q = [prod(pw(tgt[k], e[k]) for k in range(dim)) for e in mons]
    return C, t, P, q


def _solve(C, t, P, q, lgp, normalize, ops):
    num, *_, solve = ops
    n = len(t)
    if lgp:
        m = len(q)
        A = [list(C[i]) + list(P[i]) for i in range(n)] + [[P[i][e] for i in range(n)] + [num(0)]*m for e in range(m)]
        w = solve(A, list(t) + list(q))[:n]
    else:
        w = solve(C, t)
        if normalize:
            s = sum(w); w = [v/s for v in w]
    return w


def _run(fn, arith, bits, digits):
    if arith == 'quad':
        with mp.workprec(bits):
            return fn(_ops(arith, bits))
    if arith == 'hp':
        with localcontext() as ctx:
            ctx.prec = digits
            return fn(_ops(arith, bits))
    return fn(_ops(arith, bits))


# ---------------------------------------------------------------------------------- public API
def weights_1d(R, ell, x, lgp=False, m=None, arith='double', normalize=True, bits=113, digits=120):
    """1D weights w_k, k = -R..R, for the point x from the 2R+1 cell averages, as a float64 array.
    Pure GP (lgp=False) weights are normalized to sum to one unless normalize=False. LGP borders the
    system with 1, x, ..., x^m (default m = 2R - 1)."""
    m = 2*R-1 if m is None else m
    def fn(ops):
        num = ops[0]
        ellw = num(ell) if arith != 'double' else float(ell)
        pts = [(k,) for k in range(-R, R+1)]
        C, t, P, q = _system(pts, (x,), ellw, lgp, [(e,) for e in range(m+1)], ops)
        return np.array([float(v) for v in _solve(C, t, P, q, lgp, normalize, ops)])
    return _run(fn, arith, bits, digits)


def factored_weights(R, ell, target, lgp=False, m=None, arith='double', normalize=True, bits=113, digits=120):
    """Factored (Kronecker) weights on the box stencil: the outer product of the 1D weights for each
    coordinate of `target`, an array of shape (2R+1,)*len(target). Only (2R+1) x (2R+1) systems are
    solved, once per distinct coordinate."""
    m = 2*R-1 if m is None else m
    def fn(ops):
        num = ops[0]
        ellw = num(ell) if arith != 'double' else float(ell)
        pts = [(k,) for k in range(-R, R+1)]
        cache = {}
        ws = []
        for x in target:
            key = str(x)
            if key not in cache:
                C, t, P, q = _system(pts, (x,), ellw, lgp, [(e,) for e in range(m+1)], ops)
                cache[key] = _solve(C, t, P, q, lgp, normalize, ops)
            ws.append(cache[key])
        n = 2*R+1
        W = np.empty((n,)*len(target))
        for idx in itertools.product(range(n), repeat=len(target)):
            v = ws[0][idx[0]]
            for d in range(1, len(target)):
                v = v*ws[d][idx[d]]
            W[idx] = float(v)
        return W
    return _run(fn, arith, bits, digits)


def assembled_weights(R, ell, target, lgp=False, m=None, border='tensor', stencil_kind='box',
                      arith='double', normalize=True, bits=113, digits=120):
    """Weights from the full multidimensional system on the stencil.
    Returns (points, weights): the list of stencil offsets and a float64 array of their weights. For the
    box stencil, weights.reshape((2R+1,)*dim) matches factored_weights. LGP borders the system with the
    monomials of monomials(m, dim, border); note that the diamond stencil with a total-degree border is
    singular in 3D (the cell averages of xyz vanish on the R = 2 diamond)."""
    m = 2*R-1 if m is None else m
    dim = len(target)
    pts = stencil(R, dim, stencil_kind)
    def fn(ops):
        num = ops[0]
        ellw = num(ell) if arith != 'double' else float(ell)
        C, t, P, q = _system(pts, target, ellw, lgp, monomials(m, dim, border), ops)
        return np.array([float(v) for v in _solve(C, t, P, q, lgp, normalize, ops)])
    return pts, _run(fn, arith, bits, digits)


def lgp_1d_hp(R, ell, x, m=None):
    """1D LGP in the current decimal context: returns (w, lam, t, q, C, P) as Decimal lists, for the
    derivations that need the Lagrange multipliers (Proposition 4, Appendix C)."""
    m = 2*R-1 if m is None else m
    off = list(range(-R, R+1)); x = D(str(x)) if not isinstance(x, D) else x
    C = [[c_avg_hp(a-b, ell) for b in off] for a in off]
    P = [[mono_avg_hp(e, a) for e in range(m+1)] for a in off]
    t = [c_point_hp(D(a)-x, ell) for a in off]
    q = [power_hp(x, e) for e in range(m+1)]
    w, lam = solve_bordered(C, t, P, q)
    return w, lam, t, q, C, P
