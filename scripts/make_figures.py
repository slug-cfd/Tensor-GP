"""Figs. 1 and 2.

Fig. 1: the 2D diamond, 2D box and 3D box stencils of radius R = 2.
Fig. 2: (a) cond(C_1) and cond(C_1)^3 against ell/Delta (data from cond_curve_hp.py, run it first), with the
double- and quadruple-precision limits 1/eps; (b) cond(C) = cond(C_1)^D at ell = 12 Delta for D = 1, 2, 3.
Writes output/fig_stencils.pdf and output/fig_cond.pdf.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _setup
from gptensor import cond_C1

OUT = _setup.OUT
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

# ---------------- Fig 1 : stencils ----------------
fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.95))
R = 2
from matplotlib.patches import Polygon
def cube3d(ax, R, a=0.35, b=0.35):
    """5x5x5 box stencil in cabinet projection, P(x,y,z) = (x + a z, y + b z), z into the page.
    The middle slab of cells (the layer containing the center cell) is drawn inside translucent outer
    faces, so the dark center cell sits in the middle of the slab grid; its face i+1/2 (x = 1/2) is red."""
    h, c = R + .5, .5
    P = lambda x, y, z: (x + a*z, y + b*z)
    poly = lambda pts, **kw: ax.add_patch(Polygon([P(*p) for p in pts], closed=True, **kw))
    line = lambda p, q, **kw: ax.plot(*zip(P(*p), P(*q)), **kw)
    # middle slab z in [-1/2, 1/2]: front face with its grid, and its bands on the top and right faces
    poly([(-h,h,-c), (h,h,-c), (h,h,c), (-h,h,c)], fc='#c9e8c4', ec='0.3', lw=.6, zorder=1)
    poly([(h,-h,-c), (h,-h,c), (h,h,c), (h,h,-c)], fc='#b3dcad', ec='0.3', lw=.6, zorder=1)
    poly([(-h,-h,-c), (h,-h,-c), (h,h,-c), (-h,h,-c)], fc='#dff2dc', ec='0.3', lw=.7, zorder=1)
    gs = dict(color='0.35', lw=.45, zorder=1.5)
    for t in np.arange(-h+1, h):
        line((t,-h,-c), (t,h,-c), **gs); line((-h,t,-c), (h,t,-c), **gs)
        line((t,h,-c), (t,h,c), **gs);   line((h,t,-c), (h,t,c), **gs)
    # center cell [-1/2,1/2]^3: front, top, and the red face i+1/2, above the translucent faces
    poly([(-c,-c,-c), (c,-c,-c), (c,c,-c), (-c,c,-c)], fc='0.25', ec='k', lw=.8, zorder=4)
    poly([(-c,c,-c), (c,c,-c), (c,c,c), (-c,c,c)], fc='0.38', ec='k', lw=.8, zorder=4)
    poly([(c,-c,-c), (c,-c,c), (c,c,c), (c,c,-c)], fc='crimson', ec='k', lw=.8, zorder=4)
    # translucent outer faces: top (y = h) and right (x = h) with grids, front (z = -h) outline only
    for f in ([(-h,h,-h), (h,h,-h), (h,h,h), (-h,h,h)], [(h,-h,-h), (h,-h,h), (h,h,h), (h,h,-h)],
              [(-h,-h,-h), (h,-h,-h), (h,h,-h), (-h,h,-h)]):
        poly(f, fc='#eff9ed', alpha=.28, ec='0.3', lw=.8, zorder=3)
    g = dict(color='0.55', lw=.4, zorder=3)
    for t in np.arange(-h+1, h):
        line((t,h,-h), (t,h,h), **g);   line((-h,h,t), (h,h,t), **g)       # top
        line((h,t,-h), (h,t,h), **g);   line((h,-h,t), (h,h,t), **g)       # right
    m = h*(1 + a)
    ax.set_xlim(-m-.25, m+.25); ax.set_ylim(-m-.25, m+.25)

for ax, (kind, D, title) in zip(axes, [('diamond', 2, r'2D diamond, $R{=}2$ (13)'),
                                       ('box', 2, r'2D box, $R{=}2$ (25)'),
                                       ('box3', 3, r'3D box, $R{=}2$ (125)')]):
    if kind == 'box3':
        cube3d(ax, R)
    else:
        if kind == 'diamond':
            pts = [(i, j) for i in range(-R, R+1) for j in range(-R, R+1) if abs(i)+abs(j) <= R]
        else:
            pts = [(i, j) for i in range(-R, R+1) for j in range(-R, R+1)]
        for i in range(-R, R+1):
            for j in range(-R, R+1):
                ax.add_patch(plt.Rectangle((i-.5, j-.5), 1, 1, fc='white', ec='0.85', lw=.6, zorder=0))
        for (i, j) in pts:
            ax.add_patch(plt.Rectangle((i-.5, j-.5), 1, 1, fc='#dff2dc', ec='0.45', lw=.8, zorder=1))
        ax.add_patch(plt.Rectangle((-.5, -.5), 1, 1, fc='0.25', ec='k', lw=1.0, zorder=2))
        ax.plot([.5, .5], [-.5, .5], color='crimson', lw=2.2, zorder=3)     # face i+1/2
        ax.set_xlim(-R-.8, R+.8); ax.set_ylim(-R-.8, R+.8)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=8.5)
fig.tight_layout(pad=0.4)
fig.savefig(f'{OUT}/fig_stencils.pdf', bbox_inches='tight')
print('fig_stencils.pdf')

# ---------------- Fig 2 : conditioning ----------------
fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95))
HP = np.load(os.path.join(OUT, 'cond_hp.npz'))
ells = HP['ells']
ax = axes[0]
cols = {1: '#1f77b4', 2: '#d62728', 3: '#2ca02c'}
for R in (1, 2, 3):
    c1 = HP[f'R{R}']
    ax.semilogy(ells, c1,    '-',  color=cols[R], lw=1.5, label=rf'$R{{=}}{R}$ (GP{2*R+1})')
    ax.semilogy(ells, c1**3, '--', color=cols[R], lw=1.2)
ax.axhline(1/2.22e-16, color='k', lw=.9, ls=':')
ax.axhline(1/1.93e-34, color='k', lw=.9, ls='-.')
LBL = dict(fontsize=7, fontweight='bold', ha='center', va='center', zorder=5,
           bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none'))
ax.text(5.0, 1/2.22e-16, 'float64 limit', **LBL)
ax.text(10.6, 1/1.93e-34, 'quad limit', **LBL)
ax.axvline(12, color='0.6', lw=.8)
ax.text(12, 10**2.5, r'$\boldsymbol{\ell=12\Delta}$', **LBL)
ax.set_xlabel(r'$\ell/\Delta$'); ax.set_ylabel(r'$\mathrm{cond}_2$')
ax.set_ylim(1e1, 1e45)
ax.set_title(r'(a) factored vs assembled', fontsize=8.5)

ax = axes[1]
Ds = np.array([1, 2, 3])
C12 = {R: float(cond_C1(R, 12)) for R in (1, 2, 3)}   # 120-digit, ell = 12 Delta
for R, mk in ((1, 'o'), (2, 's'), (3, '^')):
    c1 = C12[R]
    ax.semilogy(Ds, c1**Ds, mk+'-', color=cols[R], lw=1.4, ms=5, label=rf'$R{{=}}{R}$ (GP{2*R+1})')
ax.axhline(1/2.22e-16, color='k', lw=.9, ls=':')
ax.axhline(1/1.93e-34, color='k', lw=.9, ls='-.')
ax.text(2.2, 1/2.22e-16, 'float64 limit', **LBL)
ax.text(2.85, 1/1.93e-34, 'quad limit', **LBL)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['1D factor', '2D box', '3D box']); ax.set_xlim(0.8, 3.15)
ax.set_ylabel(r'$\mathrm{cond}_2(\mathbf{C})=\mathrm{cond}_2(\mathbf{C}_1)^{D}$')
ax.set_ylim(1e1, 1e45); ax.legend(fontsize=7, loc='upper left', framealpha=1)
ax.set_title(r'(b) $\ell=12\Delta$', fontsize=8.5)
fig.tight_layout(pad=0.4, w_pad=3.0)
fig.savefig(f'{OUT}/fig_cond.pdf', bbox_inches='tight')
print('fig_cond.pdf')
