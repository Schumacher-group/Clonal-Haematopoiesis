import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import binom, norm

np.random.seed(42)

# ── DATA ─────────────────────────────────────────────────────────────────────
S_TRUE  = 0.3
N_W     = 1e5
X0      = 1.0
T_MAX   = 50
VAF_THR = 0.02
READ_D  = 800   # reduced for wider error bars

t_full   = np.linspace(0, T_MAX, 1000)
x_full   = X0 * np.exp(S_TRUE * t_full)
vaf_full = x_full / (2 * (N_W + x_full))

first_idx  = np.where(vaf_full >= VAF_THR)[0][0]
t_detect   = t_full[first_idx]
timepoints = np.array([t_detect + 3*i for i in range(4)])

vaf_at_t = np.interp(timepoints, t_full, vaf_full)
dp       = np.random.poisson(READ_D, 4)
ao       = np.random.binomial(dp, vaf_at_t)
vaf_obs  = ao / dp

# ── Posterior: Normal, 95% CI = 0.1 ─────────────────────────────────────────
ci_width = 0.1
sigma    = ci_width / (2 * 1.96)
mu       = 0.30
s_lo     = mu - 1.96 * sigma
s_hi     = mu + 1.96 * sigma

# ── PALETTE ───────────────────────────────────────────────────────────────────
BLUE       = '#2563EB'
BLUE_LIGHT = '#DBEAFE'
ORANGE     = '#EA580C'
GREEN      = '#16A34A'
RED        = '#DC2626'
GREY_LINE  = '#CBD5E1'
GREY_FILL  = '#F1F5F9'
INK        = '#0F172A'
SUBINK     = '#475569'
BG         = '#FFFFFF'

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'axes.labelsize':     11,
    'axes.labelcolor':    SUBINK,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'xtick.color':        SUBINK,
    'ytick.color':        SUBINK,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'text.color':         INK,
    'figure.facecolor':   BG,
    'axes.facecolor':     BG,
    'grid.color':         '#E2E8F0',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.7,
})

fig = plt.figure(figsize=(20, 6))
gs  = GridSpec(1, 4, figure=fig, wspace=0.42,
               left=0.055, right=0.975, top=0.78, bottom=0.16)
axes = [fig.add_subplot(gs[i]) for i in range(4)]
ax1, ax2, ax3, ax4 = axes

YLIM = (-0.5, 48)

# Panel labels
for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, chr(65+i), transform=ax.transAxes,
            fontsize=15, fontweight='bold', color=INK, va='top')

subtitles = [
    'VAF from clonal expansion model',
    'Sampled timepoints',
    'Observed data with sequencing error',
    'Posterior inference',
]
for ax, st in zip(axes, subtitles):
    ax.set_title(st, fontsize=11, fontweight='semibold', color=INK, pad=10, loc='left')

# ── PANEL 1 ──────────────────────────────────────────────────────────────────
ax1.yaxis.grid(True); ax1.set_axisbelow(True)

ax1.plot(t_full, vaf_full * 100, color=BLUE, lw=2.5, zorder=3)

ax1.axhline(VAF_THR*100, color=RED, lw=1.4, ls=(0,(5,3)), alpha=0.85, zorder=2)
ax1.text(T_MAX*0.98, VAF_THR*100 + 1.2, 'Detection threshold',
         color=RED, fontsize=11, ha='right', va='bottom', alpha=0.9, fontweight='medium')

ax1.text(0.04, 0.96, r'$x(t)=x_0\,e^{st}$',
         transform=ax1.transAxes, fontsize=14, color=BLUE, va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE_LIGHT, lw=0.8))
ax1.text(0.04, 0.78, r'$\mathrm{VAF}(t)=\dfrac{x(t)}{2(N_w + x(t))}$',
         transform=ax1.transAxes, fontsize=13, color=BLUE, va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE_LIGHT, lw=0.8))

ax1.set_xlabel('Time (years)'); ax1.set_ylabel('VAF (%)')
ax1.set_xlim(0, T_MAX); ax1.set_ylim(*YLIM)

# ── PANEL 2 ──────────────────────────────────────────────────────────────────
ax2.yaxis.grid(True); ax2.set_axisbelow(True)
ax2.plot(t_full, vaf_full*100, color=BLUE, lw=2.5, alpha=0.18, zorder=1)

for tp, v in zip(timepoints, vaf_at_t):
    ax2.plot([tp, tp], [0, v*100], color=BLUE, lw=1.0, ls=':', alpha=0.5, zorder=2)

ax2.scatter(timepoints, vaf_at_t*100, color=BLUE, s=90, zorder=5,
            edgecolors='white', linewidths=1.5)

for i, (tp, v) in enumerate(zip(timepoints, vaf_at_t)):
    ax2.text(tp - 0.9, v*100, f'Sample {i+1}', ha='right', va='center',
             fontsize=10, color=BLUE, fontweight='medium')

ax2.set_xlabel('Time (years)'); ax2.set_ylabel('VAF (%)')
ax2.set_xlim(0, T_MAX); ax2.set_ylim(*YLIM)

# ── PANEL 3 ──────────────────────────────────────────────────────────────────
ax3.yaxis.grid(True); ax3.set_axisbelow(True)

ax3.scatter(timepoints, vaf_at_t*100, color=BLUE, s=90, zorder=5,
            edgecolors='white', linewidths=1.5, label='True VAF', marker='o')

se = np.sqrt(vaf_obs*(1-vaf_obs)/dp)*100
ax3.errorbar(timepoints, vaf_obs*100, yerr=1.96*se,
             fmt='none', ecolor=ORANGE, elinewidth=1.8, capsize=5, capthick=1.8, zorder=6)
ax3.scatter(timepoints, vaf_obs*100, color=ORANGE, s=110, marker='D',
            edgecolors='white', linewidths=1.5, label='Observed VAF (±1.96 SE)', zorder=7)

ax3.set_xlabel('Time (years)'); ax3.set_ylabel('VAF (%)')
ax3.set_xlim(0, T_MAX); ax3.set_ylim(*YLIM)
ax3.legend(fontsize=10, loc='upper left',
           frameon=True, framealpha=0.9, edgecolor='#E2E8F0')

# ── PANEL 4 ──────────────────────────────────────────────────────────────────
ax4.yaxis.grid(True); ax4.set_axisbelow(True)

s_plot = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 800)
p_plot = norm.pdf(s_plot, mu, sigma)
ymax   = p_plot.max()

ax4.fill_between(s_plot, p_plot, alpha=0.10, color=BLUE)
ax4.plot(s_plot, p_plot, color=BLUE, lw=2.5)

ci_mask = (s_plot >= s_lo) & (s_plot <= s_hi)
ax4.fill_between(s_plot, p_plot, where=ci_mask, alpha=0.38, color=GREEN,
                 label=f'95% CI  [{s_lo:.2f}, {s_hi:.2f}]')

for xv in [s_lo, s_hi]:
    ax4.plot([xv, xv], [0, norm.pdf(xv, mu, sigma)],
             color=GREEN, lw=1.2, ls='--', alpha=0.8)

ax4.axvline(mu, color=BLUE, lw=1.8, ls=(0,(5,3)),
            label=f'$\\hat{{s}} = {mu:.2f}$')

bracket_y = ymax * 1.07

ax4.set_xlabel('Selection coefficient $s$')
ax4.set_ylabel('Posterior density')
ax4.set_xlim(mu - 4.5*sigma, mu + 4.5*sigma)
ax4.set_ylim(bottom=0, top=ymax * 1.25)
ax4.legend(fontsize=9.5, loc='upper left', frameon=True,
           framealpha=0.9, edgecolor='#E2E8F0')

# ── SUPTITLE ─────────────────────────────────────────────────────────────────
fig.suptitle(
    'Synthetic patient data generation and Bayesian recovery of clonal fitness',
    fontsize=14, fontweight='bold', color=INK, y=0.97, x=0.515
)

plt.savefig('pipeline_v4.png', dpi=200, bbox_inches='tight', facecolor=BG)
print("Done.")