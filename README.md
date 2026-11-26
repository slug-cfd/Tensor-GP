# gptensor

Companion code for

> D. Lee and S. Riedel, *Kronecker-factored Gaussian process reconstruction for high-order finite volume
> methods in multiple dimensions*, submitted to Computaters and Fluids.

This repository contains the mathematics of the paper, namely the squared-exponential (SE) kernel for cell
averages, the one-dimensional, factored (Kronecker) and assembled GP systems with and without the
polynomial (LGP) border, their condition numbers in extended precision, the flat limit, and sum
factorization. It also contains the scripts that regenerate every table, figure and number of the paper that
does not come from a hydrodynamics simulation.

The finite volume simulations of Secs. 4.3 to 4.5 (Figs. 3 to 6, Table 5 and the run statistics quoted in
those sections) were computed with a GP-MOOD solver in the Flash-X code and are not part of this repository.

## Requirements

Python 3.9 or later with numpy and mpmath, and matplotlib for the figures.

```bash
pip install -r requirements.txt     # or: pip install -e .
```

The reference outputs were produced with Python 3.14.0, numpy 2.5.1, mpmath 1.4.1 and matplotlib 3.11.0.
The scripts and tests also run from a plain checkout without installing the package.

## Quick start

```python
import numpy as np
import gptensor as g

R, ell = 2, 12.0                                   # GP5, ell = 12 Delta (lengths in units of Delta)
print(float(g.cond_C1(R, ell)))                    # 6.239e9, the only matrix the factored route solves
print(float(g.cond_assembled_box(R, ell, 3)))      # 2.428e29 = cond(C_1)^3, the assembled 3D box
print(g.largest_ell(R, 1, g.EPS_DOUBLE))           # 64.7, largest ell/Delta for the factored route in double

w = g.weights_1d(R, ell, 0.5, lgp=True)            # 1D LGP weights for the right face x = 1/2
w0 = g.weights_1d(R, ell, 0.0, lgp=True)           # and for the face center in the tangential directions
W = g.factored_weights(R, ell, (0.5, 0.0, 0.0), lgp=True)            # 5x5x5 Kronecker product of the two
pts, Wa = g.assembled_weights(R, ell, (0.5, 0.0, 0.0), arith='quad')  # full 125x125 pure-GP system

U = np.random.default_rng(0).standard_normal((32, 32, 32))           # periodic array of cell averages
V = g.apply_factored(U, [w, w0, w0])               # sum factorization, 15 operations per point
print(np.max(np.abs(V - g.apply_direct(U, W))))    # direct sum, 125 operations per point: same result
```

`python examples/quickstart.py` reconstructs a smooth 3D field at the nine Gauss points of a cell face by sum
factorization and compares it with the direct sum.

## Reproducing the paper

```bash
./run_all.sh            # or: ./run_all.sh /path/to/python
```

This runs every script in `scripts/` (about one minute in total), writes their printed output to
`output/<script>.txt` and the figures to `output/`, and compares each output with `reference_output/`.

| Paper item | Script |
|---|---|
| Fig. 1, the diamond and box stencils (Sec. 2.2) | `make_figures.py` |
| Growth law cond(C_1) ~ (ell/Delta)^(4R) (Sec. 2.4) and Table 1, condition numbers and the largest ell/Delta in double and quadruple precision (Sec. 3) | `thresholds_true.py` |
| Fig. 2, conditioning of the factored and assembled systems (Sec. 3) | `cond_curve_hp.py`, then `make_figures.py` |
| Pure GP at fixed ell/Delta is first order, moment defects E_k (Sec. 3) | `pure_gp_viability.py` |
| Pure GP with ell fixed in physical units, GP5 and GP7 rates (Sec. 3) | `thresholds.py` |
| Capped LGP rates at ell = 12 Delta in double precision, GP5 4.93 to 4.99, GP7 6.56 to 6.08 (Sec. 3) | `lgp_vs_pure.py` |
| GP7 capped LGP rates with 120-digit weights, 6.98, 6.99, 6.96 (Sec. 3) | `lgp_gp7_hp_rates.py` |
| Factored LGP within 0.2% of the exact pure GP, gap ~ (ell/Delta)^-2 (Sec. 3) | `lgp_vs_pure_weights.py` |
| Assembled LGP unsolvable in double, weight errors 1.1 (2D) and 1.6e4 (3D) (Sec. 3) | `lgp_assembled_solvable.py` |
| Table 2, 3D GP5 pure-GP weight errors, factored and assembled, double and quadruple (Sec. 4.1) | `precision_matrix.py` |
| Table 3, exact assembled LGP weights grow as (ell/Delta)^6, factored ones converge to the polynomial limit (Sec. 4.2) | `lgp_assembled_growth.py` |
| Table 4, 2D GP5 weight errors at ell = 12 Delta and ell/Delta = 120 (Sec. 4.4) | `precision_matrix_2d.py` |
| Lemma B.1, t_* is not in range(P) (Appendix B) | `t_outside_range.py` |
| Proposition 4, the factored LGP is not the assembled LGP, 2D (0.35 on weights of 0.87, sigma_2/sigma_1 = 0.21) (Sec. 3, Appendix C) | `lgp_tensor_identity.py` |
| Eq. (C.1a), the 2D expansion of C w - t (Appendix C) | `lgp_cross_check.py` |
| Eq. (C.1b), the 3D expansion, and 3D assembled vs factored LGP (5.4 on 0.97, unfoldings 0.04 to 0.05) (Secs. 4.1, 4.2, Appendix C) | `lgp_cross_3d.py` |
| Supporting check, not quoted in the text: pure GP in 2D and 3D, factored with sum factorization against assembled with the direct sum | `pure_gp_2d3d.py` |

## Tests

```bash
python tests/test_gptensor.py      # or: pytest tests
```

The tests check the kernel, the identity cond(C_1 (x) C_1) = cond(C_1)^2, the Table 1 values, that the
factored pure GP equals the assembled one, that the factored LGP reproduces every x^a y^b with a, b <= 2R-1,
that it differs from the assembled LGP, the flat limit, and sum factorization.

## The package

| Module | Contents |
|---|---|
| `gptensor/kernels.py` | SE covariances of unit-cell averages (`c_avg`) and of a cell average and a point value (`c_point`), and cell averages of monomials, in float64, mpmath and decimal arithmetic |
| `gptensor/weights.py` | stencils (box, diamond), border monomials (tensor Q_m, total degree P_m), and the weights `weights_1d`, `factored_weights`, `assembled_weights` for pure GP and LGP in double (`'double'`), quadruple (`'quad'`, mpmath at 113 bits) or 120-digit (`'hp'`) arithmetic |
| `gptensor/sumfac.py` | applying weights to a periodic array of cell averages, directly (`apply_direct`) or by sum factorization (`apply_factored`), and the operation counts |
| `gptensor/conditioning.py` | `cond_C1`, `cond_assembled_box` and `largest_ell` in 120-digit arithmetic |
| `gptensor/linalg.py` | Gaussian elimination, bordered solves and a Jacobi eigensolver in decimal arithmetic |
| `gptensor/polynomial.py` | the flat limit ell/Delta -> infinity in exact rational arithmetic |

All lengths are in units of the grid spacing Delta, so the cells are the unit intervals [k - 1/2, k + 1/2] and a
target such as x = 1/2 is the right face of the center cell. Pure GP weights are normalized to sum to one, as
in the paper. The LGP border uses the monomials 1, x, ..., x^m in each direction with m = 2R - 1 by default.

Condition numbers must be computed with the kernel and the eigenvalues in extended precision. The closed
form of the cell-average kernel cancels in float64 at large ell, and `numpy.linalg.cond` saturates near
1e16, so both make the condition number appear to fall as ell grows.

## Relation to the computations in the paper

The scripts are those used for the paper, with their shared routines moved into the `gptensor` package.
Every number they print is identical to the output of the original scripts, with one deliberate exception.
`pure_gp_viability.py` and `thresholds.py` now print cond(C_1) in 120-digit arithmetic instead of the
float64 estimate of numpy, which saturates. The paper quotes only the convergence rates from these two
scripts, and those are unchanged.

## License

MIT, see `LICENSE`.

## Citation

Please cite the paper above. `CITATION.cff` holds the citation in machine-readable form, and GitHub
shows it under "Cite this repository". A DOI for this code will be added on publication.
