"""Applying reconstruction weights to a periodic array of cell averages.

For a target point at the same offset in every cell (e.g., a face Gauss point), the reconstruction is a
convolution of the cell averages with the stencil weights. With factored weights
W = w^(1) (x) ... (x) w^(D), the D-fold sum can be nested (sum factorization),

    u_* = sum_l w^(1)_l ( sum_m w^(2)_m ( sum_n w^(3)_n u_{l,m,n} ) ),

and sweeping over a whole array, one direction at a time, shares the partial sums between neighboring
cells, so that the cost per point drops from (2R+1)^D to D (2R+1) multiplications and additions.
"""
import itertools

import numpy as np


def apply_direct(U, W):
    """Direct sum: out[i] = sum_o W[o] U[i + o] over all stencil offsets o, periodic in every axis.
    W has shape (2R+1,)*U.ndim and is indexed by o + R."""
    R = (W.shape[0]-1)//2
    out = np.zeros_like(U)
    for o in itertools.product(range(-R, R+1), repeat=U.ndim):
        V = U
        for ax, k in enumerate(o):
            V = np.roll(V, -k, axis=ax)
        out += W[tuple(k+R for k in o)]*V
    return out


def apply_factored(U, ws):
    """Sum factorization: one 1D weighted sum per axis, last axis first. ws[d] are the 1D weights along
    axis d, indexed by k + R. Equal to apply_direct(U, outer(ws)) up to round-off."""
    R = (len(ws[0])-1)//2
    V = U
    for ax in reversed(range(U.ndim)):
        V = sum(ws[ax][k+R]*np.roll(V, -k, axis=ax) for k in range(-R, R+1))
    return V


def outer(ws):
    """Kronecker (outer) product of 1D weight vectors, an array of shape (2R+1,)*len(ws)."""
    W = np.asarray(ws[0])
    for w in ws[1:]:
        W = np.multiply.outer(W, np.asarray(w))
    return W


def cost_per_point(R, dim):
    """Multiplication/addition pairs per reconstructed point:
    direct sum (2R+1)^D, sum factorization over a block D(2R+1), nested form at a single cell
    (2R+1)^D + ... + (2R+1)."""
    n = 2*R+1
    return {'direct': n**dim, 'sum_factorized_sweep': dim*n, 'nested_single_cell': sum(n**k for k in range(1, dim+1))}
