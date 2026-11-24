"""gptensor: Kronecker-factored Gaussian process (GP) reconstruction for finite volume methods.

Companion code to D. Lee and S. Riedel, "Kronecker-factored Gaussian process reconstruction for
high-order finite volume methods in multiple dimensions" (submitted to the Journal of Computational Physics).

Modules
  kernels       SE-kernel covariances of unit-cell averages and point values (float64, mpmath, decimal)
  weights       1D, factored (Kronecker) and assembled weights, pure GP and LGP, box and diamond stencils
  sumfac        applying the weights to an array of cell averages, directly or by sum factorization
  conditioning  condition numbers of C_1 and of the assembled box in 120-digit arithmetic, and the
                largest ell/Delta that double or quadruple precision can carry
  linalg        dense Gaussian elimination, bordered solves and a Jacobi eigensolver in decimal arithmetic
  polynomial    the flat limit ell/Delta -> infinity in exact rational arithmetic
"""
from .kernels import c_avg, c_point, mono_avg, c_avg_mp, c_point_mp, c_avg_hp, c_point_hp, mono_avg_hp
from .weights import stencil, monomials, weights_1d, factored_weights, assembled_weights, lgp_1d_hp
from .sumfac import apply_direct, apply_factored, outer, cost_per_point
from .conditioning import cond_C1, cond_assembled_box, largest_ell, EPS_DOUBLE, EPS_QUAD
from .polynomial import poly_weights

__version__ = '1.0.0'
