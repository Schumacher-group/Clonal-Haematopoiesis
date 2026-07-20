import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import FancyBboxPatch

# Parameters
s = 0.5          # fitness
N_w = 1e5        # wildtype cell count
x0 = 1.0         # initial mutant cell count
t = np.linspace(0, 50, 1000)

# ODE: dx/dt = s * x
def model(x, t, s):
    return s * x

# Solve ODE
x = odeint(model, x0, t, args=(s,))
x = x.flatten()

# VAF = x / (2 * (N_w + x))
vaf = x / (2 * (N_w + x))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cell count plot
axes[0].plot(t, x, color='steelblue', linewidth=2.5,
             label=r'$\frac{dx}{dt} = s \cdot x,\quad s=0.5$')
axes[0].set_xlabel('Time', fontsize=18)
axes[0].set_ylabel('Cell Count (x)', fontsize=18)
axes[0].set_title('Mutant Cell Count Over Time', fontsize=16, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=18, loc='upper left',
               framealpha=0.9, edgecolor='steelblue', fancybox=True)
axes[0].tick_params(labelsize=16)

# VAF plot
axes[1].plot(t, vaf, color='darkorange', linewidth=2.5,
             label=r'$\mathrm{VAF} = \dfrac{x}{2(N_w + x)},\quad N_w=10^5$')
axes[1].set_xlabel('Time', fontsize=18)
axes[1].set_ylabel('VAF', fontsize=18)
axes[1].set_title('Variant Allele Frequency Over Time', fontsize=16, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=18, loc='upper left',
               framealpha=0.9, edgecolor='darkorange', fancybox=True)
axes[1].tick_params(labelsize=16)

plt.tight_layout(pad=2.0)
plt.savefig('exports/cell_count_vaf_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
s = 0.5
N_w = 1e5
t = np.linspace(0, 50, 1000)

def model(x, t, s):
    return s * x

# Each scenario defines starting proportions of het and hom clones
# Both clone types grow at the same rate s
# VAF = (x_het + 2*x_hom) / (2*(N_w + x_all))  where x_all = x_het + x_hom

scenarios = [
    {
        'label': 'Heterozygous only',
        'x_het_0': 1.0, 'x_hom_0': 0.0,
        'color': 'steelblue', 'linestyle': '-',
    },
    {
        'label': 'Homozygous only',
        'x_het_0': 0.0, 'x_hom_0': 1.0,
        'color': 'darkorange', 'linestyle': '-',
    },
    {
        'label': 'Mixed (50% het, 50% hom)',
        'x_het_0': 0.5, 'x_hom_0': 0.5,
        'color': 'mediumpurple', 'linestyle': '--',
    },
    {
        'label': 'Mixed (80% het, 20% hom)',
        'x_het_0': 0.8, 'x_hom_0': 0.2,
        'color': 'mediumseagreen', 'linestyle': '--',
    },
    {
        'label': 'Mixed (20% het, 80% hom)',
        'x_het_0': 0.2, 'x_hom_0': 0.8,
        'color': 'tomato', 'linestyle': '--',
    },
]

fig, ax = plt.subplots(figsize=(13, 7))

for sc in scenarios:
    x_het = odeint(model, sc['x_het_0'], t, args=(s,)).flatten()
    x_hom = odeint(model, sc['x_hom_0'], t, args=(s,)).flatten()
    x_all = x_het + x_hom

    vaf = (x_het + 2 * x_hom) / (2 * (N_w + x_all))

    # Theoretical saturation: (f_het + 2*f_hom) / 2
    f_het = sc['x_het_0'] / (sc['x_het_0'] + sc['x_hom_0']) if (sc['x_het_0'] + sc['x_hom_0']) > 0 else 0
    f_hom = sc['x_hom_0'] / (sc['x_het_0'] + sc['x_hom_0']) if (sc['x_het_0'] + sc['x_hom_0']) > 0 else 0
    sat = (f_het + 2 * f_hom) / 2

    ax.plot(t, vaf, color=sc['color'], linestyle=sc['linestyle'],
            linewidth=2.5, label=sc['label'])
    ax.axhline(sat, color=sc['color'], linestyle=':', linewidth=1.2, alpha=0.5)
    ax.text(25.2, sat, f'sat={sat:.2f}', fontsize=9, color=sc['color'], va='center')

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('VAF', fontsize=14)
ax.set_title(
    'VAF Over Time by Zygosity\n'
    f',   $s={s}$,   $N_w=10^5$',
    fontsize=15, fontweight='bold'
)
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13, loc='upper left', framealpha=0.9, edgecolor='grey', fancybox=True)

plt.tight_layout()
plt.savefig('exports/vaf_zygosity.png', dpi=150, bbox_inches='tight')
plt.show()
print("Done")

import os
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.15,
    "grid.linestyle":     "--",
    "axes.facecolor":     "#F8F9FA",
    "figure.facecolor":   "white",
})

# ── Parameters ────────────────────────────────────────────────────────────────
S    = 0.5
E    = 2e5      # carrying capacity
N_W0 = 1e5      # initial wildtypes
X0   = 1.0      # initial mutants
T    = np.linspace(0, 60, 3000)

# ── ODE (2-equation, conserved: x + Nw + F = E) ──────────────────────────────
def model(t, y, s, E, B):
    x, Nw = y
    x   = max(x,   0.)
    Nw  = max(Nw,  0.)
    F   = max(E - x - Nw, 0.)
    dxdt  = s * x * (1 - (x + Nw) / E)
    denom = F + Nw * B
    dNwdt = -dxdt * (Nw / denom) if denom > 1e-9 else -dxdt
    if Nw < 1e-6 and dNwdt < 0:
        dNwdt = 0.
    return [dxdt, dNwdt]

def solve(B):
    sol = solve_ivp(model, (T[0], T[-1]), [X0, N_W0], t_eval=T,
                    args=(S, E, B), max_step=0.02, rtol=1e-8, atol=1e-10)
    x, Nw = sol.y
    F     = np.maximum(E - x - Nw, 0)
    vaf   = x / (2 * (Nw + x))
    return x, Nw, F, vaf

# Simple logistic (no displacement model) for reference
def simple_vaf(t, s=S, Nw=N_W0):
    x = X0 * np.exp(s * t)
    return x / (2 * (Nw + x))

# ── B values to show ─────────────────────────────────────────────────────────
B_VALUES = [0.1, 0.5, 1.0, 5.0, 20.0, 100.0]

CMAP   = plt.cm.RdYlBu_r
COLORS = [CMAP(v) for v in np.linspace(0.1, 0.9, len(B_VALUES))]

solutions = {B: solve(B) for B in B_VALUES}

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 12))
gs = gridspec.GridSpec(2, 3,
                       top=0.92, bottom=0.07,
                       left=0.07, right=0.97,
                       hspace=0.38, wspace=0.28)

# ══ Panel 1: VAF over time, all B values ═════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2])   # wide left

# Simple model reference
ax1.plot(T, simple_vaf(T), color="black", lw=1.8, ls=":", alpha=0.55,
         label="Simple model\n(no displacement, fixed $N_w$)")

for B, col in zip(B_VALUES, COLORS):
    x, Nw, F, vaf = solutions[B]
    lbl = f"$B = {B}$"
    ax1.plot(T, vaf, color=col, lw=2.2, label=lbl)

# Saturation lines
ax1.axhline(0.5, color="grey", lw=0.9, ls="--", alpha=0.4)
ax1.text(61, 0.502, "VAF = 0.5\n(het sat.)", fontsize=8, color="grey", va="bottom")

ax1.set_xlabel("Time (years)", fontsize=13)
ax1.set_ylabel("VAF", fontsize=13)
ax1.set_xlim(0, 60); ax1.set_ylim(-0.01, 0.62)
ax1.set_title("VAF Over Time — Varying Displacement Efficiency $B$\n"
              r"$\frac{dx}{dt}=s\cdot x\left(1-\frac{x+N_w}{E}\right),$"
              r"$\quad\frac{dN_w}{dt}=-\frac{dx}{dt}\cdot\frac{N_w}{F+N_w B}$",
              fontsize=12, fontweight="bold", pad=8)
ax1.legend(fontsize=9, loc="upper left", framealpha=0.92,
           edgecolor="#ccc", ncol=2)

# ══ Panel 2: Final VAF vs B (summary curve) ══════════════════════════════════
ax2 = fig.add_subplot(gs[0, 2])

B_sweep  = np.logspace(-2, 3, 200)
vaf_final = []
for B in B_sweep:
    _, _, _, vaf = solve(B)
    vaf_final.append(vaf[-1])

ax2.semilogx(B_sweep, vaf_final, color="#3A7FD5", lw=2.5)
ax2.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.5)
ax2.axhline(simple_vaf(T[-1]), color="black", lw=1, ls=":", alpha=0.5)

# Mark the B=1 point
_, _, _, vaf_B1 = solutions[1.0]
ax2.scatter([1.0], [vaf_B1[-1]], color="red", s=80, zorder=5)
ax2.text(1.2, vaf_B1[-1]+0.01, "$B=1$", fontsize=9, color="red")

for B, col in zip(B_VALUES, COLORS):
    _, _, _, vaf = solutions[B]
    ax2.scatter([B], [vaf[-1]], color=col, s=50, zorder=5)

ax2.set_xlabel("Displacement efficiency $B$ (log scale)", fontsize=11)
ax2.set_ylabel("Final VAF (at $t=60$)", fontsize=11)
ax2.set_title("Final VAF vs $B$", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 0.6)

# Regime annotations
ax2.text(0.015, 0.54, "Displacement\ndominated", fontsize=8,
         color="#CC4400", style="italic")
ax2.text(10, 0.27, "Free space\nfilling", fontsize=8,
         color="#006699", style="italic")

# ══ Panel 3: Cell composition at B=1 ═════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])

B_show = 1.0
x, Nw, F, vaf = solutions[B_show]

ax3.stackplot(T,
              x   / E,
              Nw  / E,
              F   / E,
              labels=["Mutant ($x$)", "Wildtype ($N_w$)", "Free space ($F$)"],
              colors=["#E07B39", "#3A7FD5", "#CCCCCC"],
              alpha=0.85)
ax3.set_xlabel("Time (years)", fontsize=11)
ax3.set_ylabel("Fraction of capacity $E$", fontsize=11)
ax3.set_title(f"Cell composition ($B={B_show}$)", fontsize=12, fontweight="bold")
ax3.set_xlim(0, 60); ax3.set_ylim(0, 1)
ax3.legend(fontsize=9, loc="center left", framealpha=0.9)

# ══ Panel 4: Cell composition at B=20 ════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])

B_show2 = 20.0
x2, Nw2, F2, vaf2 = solutions[B_show2]

ax4.stackplot(T,
              x2  / E,
              Nw2 / E,
              F2  / E,
              labels=["Mutant ($x$)", "Wildtype ($N_w$)", "Free space ($F$)"],
              colors=["#E07B39", "#3A7FD5", "#CCCCCC"],
              alpha=0.85)
ax4.set_xlabel("Time (years)", fontsize=11)
ax4.set_ylabel("Fraction of capacity $E$", fontsize=11)
ax4.set_title(f"Cell composition ($B={B_show2}$)", fontsize=12, fontweight="bold")
ax4.set_xlim(0, 60); ax4.set_ylim(0, 1)
ax4.legend(fontsize=9, loc="center left", framealpha=0.9)

# ══ Panel 5: VAF comparison — simple vs expanded at B=1 ═══════════════════════
ax5 = fig.add_subplot(gs[1, 2])

_, _, _, vaf_B1 = solutions[1.0]
vaf_simple      = simple_vaf(T)

ax5.plot(T, vaf_simple, color="black", lw=2, ls=":", label="Simple het model")
ax5.plot(T, vaf_B1,     color="red",   lw=2.2,       label="Expanded model ($B=1$)")

ax5.fill_between(T, vaf_simple, vaf_B1,
                 where=(vaf_B1 > vaf_simple),
                 color="red", alpha=0.12, label="VAF difference")
ax5.fill_between(T, vaf_simple, vaf_B1,
                 where=(vaf_B1 < vaf_simple),
                 color="blue", alpha=0.12)

ax5.set_xlabel("Time (years)", fontsize=11)
ax5.set_ylabel("VAF", fontsize=11)
ax5.set_title("Simple vs Expanded model ($B=1$)", fontsize=12, fontweight="bold")
ax5.set_xlim(0, 60); ax5.set_ylim(-0.01, 0.62)
ax5.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.4)
ax5.legend(fontsize=9, loc="upper left", framealpha=0.9)

# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle("Expanded Mutant Growth Model — Displacement vs Free Space Filling",
             fontsize=14, fontweight="bold", y=0.97)

plt.savefig("exports/expanded_growth_model_vaf.png",
            dpi=180, bbox_inches="tight")
print("Saved.")