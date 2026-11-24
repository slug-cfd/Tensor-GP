"""Table 1 and the growth law cond(C_1) ~ (ell/Delta)^(4R) of Sec. 3.

cond(C_1) for R = 1, 2, 3 at ell/Delta = 3 .. 48, with the kernel AND the eigensolve in 120-digit
arithmetic, and the largest ell/Delta for which cond(C_1)^D <= 1/eps in double and quadruple precision
(D = 1 is the factored route, D = 2, 3 the assembled box).
"""
import math
from decimal import getcontext
import _setup  # noqa: F401
from gptensor import cond_C1, largest_ell
getcontext().prec = 120

print("TRUE cond(C_1) (120-digit kernel AND eigensolve)")
print(f"{'R':>3} | " + "".join(f"{'ell='+str(L):>12}" for L in (3,6,12,24,48)))
for R in (1,2,3):
    row = [cond_C1(R, L) for L in (3,6,12,24,48)]
    print(f"{R:>3} | " + "".join(f"{float(c):12.3e}" for c in row))
    print(f"    | growth exponents: " + " ".join(
        f"{math.log2(float(row[i+1]/row[i])):.2f}" for i in range(4)))

print()
print("Largest ell/Delta a precision can carry:  cond(C_1)^D <= 1/eps")
print(f"{'R':>3} {'GP':>4} {'route':>15} | {'float64':>9} {'quad':>9}")
print("-"*48)
for R in (1,2,3):
    for Dd, lab in ((1,'factored (1D)'), (2,'assembled 2D'), (3,'assembled 3D')):
        print(f"{R:>3} {2*R+1:>4} {lab:>15} | {largest_ell(R,Dd,2.22e-16):9.1f} {largest_ell(R,Dd,1.93e-34):9.1f}")
