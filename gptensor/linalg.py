"""Dense linear algebra in decimal arithmetic, at the precision of the current decimal context.

These routines are slow but exact enough to serve as references: the 1D covariance C_1 has condition
numbers up to 1e25 at the length scales of the paper, and the assembled 3D systems up to 1e60, beyond
what float64 or even quadruple precision can resolve.
"""
from decimal import Decimal as D


def solve_dec(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting. A is a list of rows."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        for r in range(c+1, n):
            f = M[r][c]/piv
            if f == 0:
                continue
            Mr, Mc = M[r], M[c]
            for k in range(c, n+1):
                Mr[k] -= f*Mc[k]
    x = [D(0)]*n
    for r in reversed(range(n)):
        x[r] = (M[r][n]-sum(M[r][k]*x[k] for k in range(r+1, n)))/M[r][r]
    return x


def bordered_matrix(C, P):
    """The bordered (LGP) matrix [[C, P], [P^T, 0]] as a list of rows."""
    n, m = len(C), len(P[0])
    return ([list(C[i]) + list(P[i]) for i in range(n)]
            + [[P[i][e] for i in range(n)] + [D(0)]*m for e in range(m)])


def solve_bordered(C, t, P, q):
    """Solve the bordered system [[C, P], [P^T, 0]] [w; lam] = [t; q]. Returns (w, lam)."""
    n = len(t)
    x = solve_dec(bordered_matrix(C, P), list(t) + list(q))
    return x[:n], x[n:]


def eig_sym_dec(A, max_rotations=600, tol_exp=-105):
    """Eigenvalues of a small symmetric matrix by the classical Jacobi method (largest off-diagonal entry
    first). Raises if the iteration has not converged, i.e., if an off-diagonal entry above 10^tol_exp
    remains after max_rotations rotations. Meant for the (2R+1) x (2R+1) matrix C_1."""
    n = len(A)
    A = [list(r) for r in A]
    tol = D(10)**tol_exp
    for _ in range(max_rotations):
        off = D(0); p = q = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i][j]) > off:
                    off, p, q = abs(A[i][j]), i, j
        if off < tol:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        th = (aqq-app)/(2*apq); t = (D(1) if th >= 0 else D(-1))/(abs(th)+(th*th+1).sqrt())
        c = 1/(t*t+1).sqrt(); s = t*c
        for k in range(n):
            akp, akq = A[k][p], A[k][q]; A[k][p] = c*akp-s*akq; A[k][q] = s*akp+c*akq
        for k in range(n):
            apk, aqk = A[p][k], A[q][k]; A[p][k] = c*apk-s*aqk; A[q][k] = s*apk+c*aqk
    else:
        off = max((abs(A[i][j]) for i in range(n) for j in range(i+1, n)), default=D(0))
        if off >= tol:
            raise RuntimeError(f'Jacobi eigensolver did not converge in {max_rotations} rotations '
                               f'(largest off-diagonal entry {off:.2e})')
    return [A[i][i] for i in range(n)]


def cond_sym_dec(A, **kw):
    """2-norm condition number max|eig| / min|eig| of a small symmetric matrix, in decimal arithmetic."""
    ev = [abs(v) for v in eig_sym_dec(A, **kw)]
    return max(ev)/min(ev)
