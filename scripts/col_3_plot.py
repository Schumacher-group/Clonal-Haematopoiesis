import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os; os.makedirs("exports", exist_ok=True)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

K, x0 = 1e5, 1.0
INK = "#0F172A"; BG = "#FFFFFF"; BLOODRED = "#8A0303"
C_EXP = "#D94F00"; C_MUT = "#1A4F8C"; C_WT = "#2A7A3B"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "text.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "grid.color": "#E2E8F0", "grid.linewidth": 0.5,
})

def disp_clone(t_arr, s, E, B):
    n0, F0 = K, max(E - K, 0.)
    def rhs(t, y):
        x, Nw, F = max(y[0],0), max(y[1],1e-9), max(y[2],0)
        dxdt = s * x * (1 - (x+Nw)/E)
        sd = max(F + B*Nw, 1e-12)
        return [dxdt, -dxdt*B*Nw/sd, -dxdt*F/sd]
    sol = solve_ivp(rhs, [0, t_arr[-1]+1], [x0, n0, F0],
                    dense_output=True, max_step=0.5, rtol=1e-6, atol=1e-8)
    v = sol.sol(t_arr)
    x, Nw = np.maximum(v[0], 0), np.maximum(v[1], 0)
    return x, Nw, x / (2*(Nw + x + 1e-12))

t = np.linspace(0, 100, 4000); s = 0.3

combos = [
    (6e5, 0.5,  "#60A5FA", "-",   r"$E=6\!\times\!10^5$, $B=0.5$"),
    (6e5, 5.0,  "#1E3A8A", "-",   r"$E=6\!\times\!10^5$, $B=5.0$"),
    (1e6, 2.5,  "#15803D", "--",  r"$E=10\!\times\!10^5$, $B=2.5$"),
    (4e5, 2.5,  "#86EFAC", "--",  r"$E=4\!\times\!10^5$, $B=2.5$"),
]

exp_x   = x0 * np.exp(s * t)
exp_vaf = exp_x / (2*(K + exp_x))

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(14, 5.5),
    gridspec_kw={"wspace": 0.25}
)
fig.patch.set_facecolor(BG)

# ── PANEL 1: Clone size (normalised, y-clipped at niche capacity) ─────────────
ax1.set_axisbelow(True); ax1.yaxis.grid(True, alpha=0.5)

# Exponential — clip display to 3× max niche so curve is visible before it rockets
E_max = 1e6
y_ceil = 3.2 * E_max / K   # 3.2 × max niche in units of K

exp_x_norm = np.clip(exp_x / K, None, y_ceil)
ax1.plot(t, exp_x_norm, color=C_EXP, lw=2.5, ls="--", zorder=5, label="Exponential")

for E, B, col, ls, lab in combos:
    x_d, _, _ = disp_clone(t, s, E, B)
    ax1.plot(t, x_d / K, color=col, lw=2.0, ls=ls, zorder=3, label=lab)
    # horizontal carrying capacity line
    ax1.axhline(E / K, color=col, lw=0.6, ls=":", alpha=0.45)

# Arrow indicating exponential keeps growing
ax1.annotate("unbounded\ngrowth →", xy=(97, y_ceil * 0.94), xytext=(80, y_ceil * 0.70),
    fontsize=14, color=C_EXP, ha="center",
    arrowprops=dict(arrowstyle="->", color=C_EXP, lw=1.2))

# Displacement equation — top left
ax1.text(0.03, 0.97,
    r"$\dfrac{dx}{dt} = s \cdot x\!\left(1-\dfrac{x+N_W}{E}\right)$",
    transform=ax1.transAxes, ha="left", va="top", fontsize=14, color=C_MUT)

# # Exponential equation — below it
# ax1.text(0.03, 0.66,
#     r"$\dfrac{dx}{dt} = sx$  (exponential)",
#     transform=ax1.transAxes, ha="left", va="top", fontsize=11.5, color=C_EXP,
#     bbox=dict(boxstyle="round,pad=0.4", fc="#FFF7ED", ec=C_EXP, lw=1.3, alpha=0.96))

ax1.set_xlabel("Time (yr)", fontsize=12)
ax1.set_ylabel(r"Clone size  $x(t)\cdot 10^5$", fontsize=12)
ax1.set_xlim(0, 100); ax1.set_ylim(0, y_ceil)
ax1.set_title("Mutant clone growth: displacement model vs exponential",
    fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, loc="center right", framealpha=0.95,
    edgecolor="#CBD5E1", handlelength=2.2)

# ── PANEL 2: VAF ──────────────────────────────────────────────────────────────
ax2.set_axisbelow(True); ax2.yaxis.grid(True, alpha=0.5)

ax2.axhline(0.5, color=INK, lw=0.9, ls=":", alpha=0.4)
ax2.text(1.0, 0.513, "VAF = 0.5", fontsize=8.5, color=INK, alpha=0.5, va="bottom")

ax2.plot(t, exp_vaf, color=C_EXP, lw=2.5, ls="--", zorder=5, label="Exponential")

for E, B, col, ls, lab in combos:
    _, _, vaf_d = disp_clone(t, s, E, B)
    ax2.plot(t, vaf_d, color=col, lw=2.0, ls=ls, zorder=3, label=lab)

# ── Inset: clone size dynamics ────────────────────────────────────────────────
axins = inset_axes(
    ax2,
    width="38%",
    height="38%",
    loc="upper right",
    borderpad=1.2
)

axins.plot(
    t,
    exp_x_norm,
    color=C_EXP,
    lw=1.5,
    ls="--"
)

for E, B, col, ls, lab in combos:
    x_d, _, _ = disp_clone(t, s, E, B)
    axins.plot(
        t,
        x_d / K,
        color=col,
        lw=1.2,
        ls=ls
    )

axins.set_xlim(0, 100)
axins.set_ylim(0,100)
    
# WT depletion equation — top left
ax2.text(0.03, 0.97,
    r"$\dfrac{dN_W}{dt} = -\dfrac{dx}{dt}\cdot\dfrac{B \cdot N_W}{F+B \cdot N_W}$",
    transform=ax2.transAxes, ha="left", va="top", fontsize=14, color=C_WT)

# VAF equation — bottom right, clear of legend
ax2.text(0.3, 0.4,
    r"$V(t) = \dfrac{x}{2(N_W + x)}$",
    transform=ax2.transAxes, ha="right", va="bottom", fontsize=12,
    color=BLOODRED, fontweight="bold")

ax2.set_xlabel("Time (yr)", fontsize=12)
ax2.set_ylabel("VAF", fontsize=12)
ax2.set_xlim(0, 100); ax2.set_ylim(-0.01, 0.70)
ax2.set_title(r"VAF dynamics: displacement model vs exponential  ($s=0.3$)",
    fontsize=12, fontweight="bold")
# Legend bottom-centre, between the two equation boxes
ax2.legend(fontsize=9, loc="lower left", framealpha=0.95,
    edgecolor="#CBD5E1", ncol=2, handlelength=2.2, columnspacing=1.0,
    bbox_to_anchor=(0.44, 0.01))

plt.savefig("exports/col3_A_clean2.pdf", dpi=220, bbox_inches="tight")
print("Saved.")
plt.close()