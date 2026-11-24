"""Squared-exponential (SE) kernel for cell averages on a uniform grid.

Everything is measured in units of the grid spacing Delta, so the cells are the unit intervals
I_k = [k - 1/2, k + 1/2] and the length scale is ell = ell/Delta. For the SE kernel
k(x, y) = exp(-(x - y)^2 / (2 ell^2)) the two one-dimensional covariances needed by finite-volume GP
reconstruction have closed forms:

  c_avg(d, ell)    covariance of the averages over two unit cells whose centers are d apart
                   (the entries c(k - k') of the 1D covariance matrix C_1),
  c_point(d, ell)  covariance of the average over a unit cell and the point value at distance d from
                   its center (the entries t(x_* - k) of the 1D prediction vector t_*).

In D dimensions on a box stencil both are products of these 1D factors, which is the separability that
the Kronecker factorization rests on.

Each quantity is provided in three arithmetics:
  float64   (suffix none)   math.erf, for the double-precision computations,
  mpmath    (suffix _mp)    any precision set through mpmath.mp (113 bits = IEEE quadruple),
  decimal   (suffix _hp)    the decimal module at the precision of the current decimal context
                            (120 digits for the reference computations of the paper).

The closed form of c_avg cancels badly in float64 when ell is large, so condition numbers must never be
computed from the float64 kernel (see conditioning.py).
"""
import math
from decimal import Decimal as D

import mpmath as mp

# ----------------------------------------------------------------------------------------- float64
_SQRT_PI = math.sqrt(math.pi)
_RT2 = math.sqrt(2.0)


def c_avg(d, ell):
    """float64 covariance of two unit-cell averages whose centers are d apart."""
    yp, ym, y0 = (d+1)/(_RT2*ell), (d-1)/(_RT2*ell), d/(_RT2*ell)
    return _SQRT_PI*ell**2*(yp*math.erf(yp)+ym*math.erf(ym)
                            -2*(y0*math.erf(y0)+math.exp(-y0**2)/_SQRT_PI)
                            +(math.exp(-yp**2)+math.exp(-ym**2))/_SQRT_PI)


def c_point(d, ell):
    """float64 covariance of a unit-cell average and the point value at distance d from its center."""
    return ell*math.sqrt(0.5)*_SQRT_PI*(math.erf((d+0.5)/(_RT2*ell))-math.erf((d-0.5)/(_RT2*ell)))


def mono_avg(e, c):
    """float64 average of x^e over the unit cell centered at c."""
    return ((c+0.5)**(e+1)-(c-0.5)**(e+1))/(e+1)


# ------------------------------------------------------------------------------------------ mpmath
def c_avg_mp(d, ell):
    """mpmath version of c_avg, at the current mpmath precision."""
    d = mp.mpf(d); ell = mp.mpf(ell); r2 = mp.sqrt(2); sp = mp.sqrt(mp.pi)
    yp, ym, y0 = (d+1)/(r2*ell), (d-1)/(r2*ell), d/(r2*ell)
    return sp*ell**2*(yp*mp.erf(yp)+ym*mp.erf(ym)-2*(y0*mp.erf(y0)+mp.exp(-y0**2)/sp)
                      +(mp.exp(-yp**2)+mp.exp(-ym**2))/sp)


def c_point_mp(d, ell):
    """mpmath version of c_point, at the current mpmath precision."""
    d = mp.mpf(d); ell = mp.mpf(ell); r2 = mp.sqrt(2); h = mp.mpf(1)/2
    return ell*mp.sqrt(h)*mp.sqrt(mp.pi)*(mp.erf((d+h)/(r2*ell))-mp.erf((d-h)/(r2*ell)))


def mono_avg_mp(e, c):
    """mpmath average of x^e over the unit cell centered at c."""
    c = mp.mpf(c); h = mp.mpf(1)/2
    return ((c+h)**(e+1)-(c-h)**(e+1))/(e+1)


# ----------------------------------------------------------------------------------------- decimal
# pi to about 150 digits, enough for the 120-digit reference computations
PI = D('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706'
       '798214808651328230664709384460955058223172535940812848')


def erf_hp(x):
    """erf in decimal arithmetic (Maclaurin series, terms below 1e-100 dropped)."""
    x = D(x)
    if x < 0:
        return -erf_hp(-x)
    t = s = x; tol = D(10)**-100
    for n in range(1, 900):
        t = t*(-x*x)/D(n); term = t/D(2*n+1); s += term
        if abs(term) < tol:
            break
    return s*2/PI.sqrt()


def c_avg_hp(d, ell):
    """decimal version of c_avg, at the precision of the current decimal context.
    A float ell enters with its exact binary value, D(ell)."""
    d = D(d); ell = D(ell); r2 = D(2).sqrt()
    yp, ym, y0 = (d+1)/(r2*ell), (d-1)/(r2*ell), d/(r2*ell)
    sp = PI.sqrt()
    return sp*ell*ell*(yp*erf_hp(yp) + ym*erf_hp(ym)
                       - 2*(y0*erf_hp(y0) + (-y0*y0).exp()/sp)
                       + ((-yp*yp).exp() + (-ym*ym).exp())/sp)


def c_point_hp(d, ell):
    """decimal version of c_point, at the precision of the current decimal context.
    A float ell enters through its shortest decimal representation, D(str(ell))."""
    d = D(d); ell = D(str(ell)); r2 = D(2).sqrt()
    return ell*D('0.5').sqrt()*PI.sqrt()*(erf_hp((d+D('0.5'))/(r2*ell))-erf_hp((d-D('0.5'))/(r2*ell)))


def mono_avg_hp(e, c):
    """decimal average of x^e over the unit cell centered at c."""
    c = D(c); h = D('0.5')
    return ((c+h)**(e+1)-(c-h)**(e+1))/D(e+1)


def power_hp(x, e):
    """x^e in decimal arithmetic, with x^0 = 1."""
    return D(1) if e == 0 else D(x)**e
