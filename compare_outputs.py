"""Compare output/*.txt with reference_output/*.txt line by line, ignoring the run-time lines "(Ns)".
Exit status 0 when every output matches its reference."""
import glob
import os
import re
import sys

root = os.path.dirname(os.path.abspath(__file__))
skip = re.compile(r'^\s*\(\d+s\)\s*$')
refs = sorted(glob.glob(os.path.join(root, 'reference_output', '*.txt')))
if not refs:
    sys.exit('no reference outputs found in reference_output/')
bad = 0
for ref in refs:
    name = os.path.basename(ref)
    out = os.path.join(root, 'output', name)
    if not os.path.exists(out):
        print(f'MISSING  {name}'); bad += 1; continue
    a = [l for l in open(out).read().splitlines() if not skip.match(l)]
    b = [l for l in open(ref).read().splitlines() if not skip.match(l)]
    if a == b:
        print(f'match    {name}')
    else:
        bad += 1
        print(f'DIFFERS  {name}')
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f'   line {i+1}\n     got: {x}\n     ref: {y}')
                break
print('all outputs match the reference' if bad == 0 else f'{bad} output(s) differ from the reference')
sys.exit(1 if bad else 0)
