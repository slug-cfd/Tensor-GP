"""Lemma B.1 / Appendix B: is t_* (the cell averages of the Gaussian centered at x_*) in range(P) (the cell
averages of 1, ..., x^(2R-1))? range(P) is a hyperplane whose normal n holds the 2R-th difference
coefficients, so the check is n.t != 0, over 0.1 <= ell/Delta <= 50 and |x_*| <= 1/2, in 60 digits."""
import mpmath as mp
import _setup  # noqa: F401
from gptensor.kernels import mono_avg_mp, c_point_mp
mp.mp.dps = 60
from math import comb
for R in (2, 3):
    m = 2*R-1; ks = range(-R, R+1)
    n = [(-1)**j*comb(2*R, j) for j in range(2*R+1)]
    avg = mono_avg_mp
    print(f"R={R}: max |n^T P| =", mp.nstr(max(abs(sum(n[i]*avg(e, k) for i, k in enumerate(ks))) for e in range(m+1)), 3))
    # smallest positive root of He_{2R}
    He = lambda z: mp.hermite(2*R, z/mp.sqrt(2))   # physicists H_n(z/sqrt2) = 2^{n/2} He_n(z)
    z1 = mp.findroot(He, 0.7 if R == 2 else 0.6)
    print(f"   z1(He_{2*R}) = {mp.nstr(z1, 6)},  Rolle bound ell >= (R+1)/z1 = {mp.nstr((R+1)/z1, 4)}")
    worst = None
    for L in [mp.mpf(x)/20 for x in range(2, 1001)]:          # ell = 0.1 .. 50
        for xs in [mp.mpf(j)/20 - mp.mpf(1)/2 for j in range(21)]:
            t = [c_point_mp(k-xs, L) for k in ks]
            nt = sum(n[i]*t[i] for i in range(len(t)))
            # sign change detection along ell for fixed xs handled by recording sign
            rel = abs(nt)/mp.sqrt(sum(x*x for x in t))/mp.sqrt(sum(c*c for c in n))
            if worst is None or rel < worst[0]: worst = (rel, L, xs, nt)
    print(f"   min over ell in [0.1,50], x* in [-1/2,1/2] of |n.t|/(|n||t|) = {mp.nstr(worst[0],3)} at ell={mp.nstr(worst[1],4)}, x*={mp.nstr(worst[2],3)}")
    # sign of n.t across the scan
    signs = set()
    for L in [mp.mpf(x)/20 for x in range(2, 1001)]:
        for xs in [mp.mpf(j)/20 - mp.mpf(1)/2 for j in range(21)]:
            t = [c_point_mp(k-xs, L) for k in ks]
            signs.add(mp.sign(sum(n[i]*t[i] for i in range(len(t)))))
    print("   signs of n.t seen:", signs)
