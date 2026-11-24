"""Table 3 (Sec. 4.2): exact-arithmetic size of the assembled LGP weights against ell/Delta.

GP5 (R=2), tensor border x^a y^b [z^c] with a,b,c <= m=3, V->P, target (1/2, 0[, 0]).
The assembled bordered system is solved in 120-digit decimal arithmetic, with the
kernel of gptensor.kernels (the same as for Table 2), and compared with the factored LGP
(Kronecker product of the 1D LGP weights).

Check column: relative difference from an independent 200-digit reference, in which
the erf series and pi follow the working precision (the erf of gptensor.kernels stops at 1e-100 and
its pi has ~150 digits, so a check that only raises the working precision would not
test the inputs). The 3D solve loses ~12 digits at ell = 3 Delta and ~40 at 48 Delta,
about 7 per doubling, as cond(C_1)^3 ~ (ell/Delta)^24, so 120 digits leave ~80.

Reports max|w| and sum|w| (the latter bounds the amplification of errors in the
cell averages), and the growth exponent log2(ratio) per doubling of ell/Delta.
"""
import itertools, math
from fractions import Fraction as Fr
import numpy as np
from decimal import Decimal as D, getcontext, localcontext
import _setup  # noqa: F401
from gptensor.kernels import c_avg_hp as SE_C1_hp, erf_hp, PI, mono_avg_hp as avg, power_hp as pw
from gptensor.linalg import solve_dec as solve, solve_bordered
from gptensor.polynomial import poly_weights as _poly_weights
getcontext().prec = 120

R, m = 2, 3
PREC, PREC_REF = 120, 200
off = list(range(-R, R+1))


def pi_dec():
    """pi to the current precision (recipe from the decimal module docs)."""
    with localcontext() as ctx:
        ctx.prec += 2
        three = D(3); lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
        while s != lasts:
            lasts = s; n, na = n+na, na+8; d, da = d+da, da+32; t = (t*n)/d; s += t
    return +s


def kernels(full):
    """(C1, T1). full=False: gptensor.kernels. full=True: erf and pi at the working precision."""
    if not full:
        pi, erf, C1 = PI, erf_hp, SE_C1_hp
    else:
        pi = pi_dec(); sp = pi.sqrt(); tol = D(10)**-(getcontext().prec+5)
        def erf(x):
            if x < 0: return -erf(-x)
            t = s = x
            for n in range(1, 5000):
                t = t*(-x*x)/D(n); term = t/D(2*n+1); s += term
                if abs(term) < tol: break
            return s*2/sp
        def C1(d, L):
            d = D(d); L = D(L); r2 = D(2).sqrt()
            yp, ym, y0 = (d+1)/(r2*L), (d-1)/(r2*L), d/(r2*L)
            return sp*L*L*(yp*erf(yp)+ym*erf(ym)-2*(y0*erf(y0)+(-y0*y0).exp()/sp)
                           +((-yp*yp).exp()+(-ym*ym).exp())/sp)
    def T1(d, L):
        d = D(d); LL = D(str(L)); r2 = D(2).sqrt()
        return LL*D('0.5').sqrt()*pi.sqrt()*(erf((d+D('0.5'))/(r2*LL))-erf((d-D('0.5'))/(r2*LL)))
    return C1, T1


def bordered(C, t, P, q):
    return solve_bordered(C, t, P, q)[0]


def poly_weights(x):
    """Exact weights of the degree-2R polynomial matching the 2R+1 cell averages, at x."""
    return _poly_weights(R, x)


def weights(L, Dim, prec, full=False, factored=True):
    """Assembled LGP weights (Decimal) and, optionally, factored LGP weights (float)."""
    with localcontext() as ctx:
        ctx.prec = prec
        C1, T1 = kernels(full)
        tg = [D('0.5')] + [D(0)]*(Dim-1)
        kv = {d: C1(d, L) for d in range(-2*R, 2*R+1)}
        t1 = [{a: T1(D(a)-tg[k], L) for a in off} for k in range(Dim)]
        wf = None
        if factored:
            Cf = [[kv[a-b] for b in off] for a in off]; Pf = [[avg(e, a) for e in range(m+1)] for a in off]
            for k in range(Dim):
                w1 = np.array([float(v) for v in bordered(Cf, [t1[k][a] for a in off], Pf, [pw(tg[k], e) for e in range(m+1)])])
                wf = w1 if wf is None else np.multiply.outer(wf, w1)
            wf = wf.ravel()
        pts = list(itertools.product(off, repeat=Dim)); mons = list(itertools.product(range(m+1), repeat=Dim))
        C = [[math.prod([kv[a[k]-b[k]] for k in range(Dim)], start=D(1)) for b in pts] for a in pts]
        t = [math.prod([t1[k][a[k]] for k in range(Dim)], start=D(1)) for a in pts]
        P = [[math.prod([avg(e[k], a[k]) for k in range(Dim)], start=D(1)) for e in mons] for a in pts]
        q = [math.prod([pw(tg[k], e[k]) for k in range(Dim)], start=D(1)) for e in mons]
        wa = bordered(C, t, P, q)
    return wa, wf


for Dim in (2, 3):
    print(f"\n{Dim}D GP5, assembled vs factored LGP, {PREC}-digit weights "
          f"(check: vs {PREC_REF}-digit reference with erf and pi at full precision)")
    print(f"{'ell/Delta':>9} {'check':>8} | {'max|w| asm':>11} {'sum|w| asm':>11} {'growth':>6} | {'max|w| fac':>10} {'sum|w| fac':>10} {'growth':>7}")
    prev = prevf = None
    for L in (3, 6, 12, 24, 48):
        wa, wf = weights(float(L), Dim, PREC)
        wr, _ = weights(float(L), Dim, PREC_REF, full=True, factored=False)
        with localcontext() as ctx:
            ctx.prec = PREC_REF + 10
            sc = max(abs(v) for v in wr)
            chk = float(max(abs(a-b) for a, b in zip(wa, wr))/sc)
        wa = np.array([float(v) for v in wa])
        s = np.abs(wa).sum()
        g = f"{math.log2(s/prev):6.2f}" if prev else f"{'':>6}"
        prev = s
        sf = np.abs(wf).sum()
        gf = f"{math.log2(sf/prevf):7.4f}" if prevf else f"{'':>7}"
        prevf = sf
        print(f"{L:9d} {chk:8.0e} | {np.max(np.abs(wa)):11.3e} {s:11.3e} {g} | {np.max(np.abs(wf)):10.3f} {sf:10.4f} {gf}", flush=True)
    # ell/Delta -> infinity: degree-4 polynomial reconstruction from 5 cell averages, exact fractions
    lim = [poly_weights(Fr(1, 2))] + [poly_weights(Fr(0))]*(Dim-1)
    mx = math.prod(max(abs(v) for v in w) for w in lim); sm = math.prod(sum(abs(v) for v in w) for w in lim)
    print(f"{'inf':>9} {'':>8} | {'':>11} {'':>11} {'':>6} | {float(mx):10.3f} {float(sm):10.4f}"
          f"   (= {mx} and {sm})")
    if Dim == 3:
        print(f"   1D polynomial weights: face {[str(v) for v in lim[0]]}, center {[str(v) for v in lim[1]]}")
