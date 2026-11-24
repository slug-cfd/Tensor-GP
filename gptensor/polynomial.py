"""The flat limit ell/Delta -> infinity in exact rational arithmetic.

As ell/Delta grows, the factored LGP weights converge to those of the degree-2R polynomial that matches
the 2R+1 cell averages, applied in each direction. These weights are rational numbers, e.g., for R = 2
(2, -13, 47, 27, -3)/60 at the face x = 1/2 and (9, -116, 2134, -116, 9)/1920 at the center x = 0.
"""
from fractions import Fraction as Fr


def poly_weights(R, x):
    """Exact weights w_k, k = -R..R, of the degree-2R polynomial reconstruction at x (in units of Delta)
    from the 2R+1 unit-cell averages, i.e., the solution of sum_k w_k <x^e>_k = x^e for e = 0..2R."""
    off = list(range(-R, R+1)); n = len(off); x = Fr(x)
    M = [[Fr((c+Fr(1, 2))**(e+1)-(c-Fr(1, 2))**(e+1), e+1) for c in off] + [x**e] for e in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0); M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]/M[c][c]; M[r] = [a-f*b for a, b in zip(M[r], M[c])]
    return [M[i][n]/M[i][i] for i in range(n)]
