import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "figure.dpi": 120,
})

# ---- colours (colour-blind friendly) ----
C_HET, C_HOM, C_MIX = "#0072B2", "#D55E00", "#009E73"
GREEN_BOX, RED_BOX = "#E4F5EC", "#FBE6E1"
GREEN_EDGE, RED_EDGE = "#009E73", "#D55E00"

# ---- model ----
Nw, s = 1.0, 0.5
t = np.linspace(0, 35, 600)
x0 = 0.04167 / np.exp(s * 16.5)          # so V_het = 2% at t=16.5
x = x0 * np.exp(s * t)

V_het   = x / (2 * (Nw + x))              # -> 0.50
V_hom   = x / (Nw + x)                    # -> 1.00
V_mix   = 1.5 * x / (2 * (Nw + x))        # -> 0.75

# observation window
t_obs = np.array([16.5, 18.5, 20.5])
def V(kind):
    xx = x0 * np.exp(s * t_obs)
    return {"het": xx/(2*(Nw+xx)), "hom": xx/(Nw+xx),
            "mix": 1.5*xx/(2*(Nw+xx))}[kind]

# ---- figure ----
fig = plt.figure(figsize=(13, 8))
gs = GridSpec(2, 3, height_ratios=[1.25, 1], hspace=0.45, wspace=0.28)

# === TOP: the truth ===
ax = fig.add_subplot(gs[0, :])
ax.axvspan(16, 21, color="0.85", alpha=0.6, zorder=0)
ax.plot(t, V_hom, color=C_HOM, lw=3.5, label="Homozygous  (ceiling 1.0)")
ax.plot(t, V_mix, color=C_MIX, lw=3.5, label="Mixed 50/50  (ceiling 0.75)")
ax.plot(t, V_het, color=C_HET, lw=3.5, label="Heterozygous  (ceiling 0.5)")
ax.set_ylim(0, 1.05); ax.set_xlim(0, 35)
ax.set_xlabel("Time (years)"); ax.set_ylabel("VAF")
ax.set_title("Same clone size, same fitness — but zygosity sets the VAF ceiling",
             fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=13)
ax.text(0.5, 0.93, r"$V_{het}(t)=\dfrac{x(t)}{2\,(N_w+x(t))}$",
        transform=ax.transAxes, fontsize=15, color=C_HET)

# === BOTTOM: fit het model to each ===
panels = [
    ("het", C_HET, "Heterozygous data", 0.501, +0.001, True),
    ("hom", C_HOM, "Homozygous data",   0.547, +0.047, False),
    ("mix", C_MIX, "Mixed data",        0.527, +0.027, False),
]
for j, (kind, col, title, shat, bias, ok) in enumerate(panels):
    a = fig.add_subplot(gs[1, j])
    Vtrue = {"het": V_het, "hom": V_hom, "mix": V_mix}[kind]
    a.axvspan(16, 21, color="0.85", alpha=0.6, zorder=0)
    a.plot(t, Vtrue, color=col, lw=3, label="Truth")
    a.plot(t, V_het, "k--", lw=2.5, label="Het-model fit")   # always saturates at 0.5
    a.scatter(t_obs, V(kind), color=col, s=55, zorder=5, edgecolor="k", linewidth=0.5)
    a.set_xlim(0, 35); a.set_ylim(0, 1.05)
    a.set_title(title, color=col, fontweight="bold")
    a.set_xlabel("Time (years)")
    if j == 0: a.set_ylabel("VAF")
    if j == 0: a.legend(loc="upper left", frameon=False, fontsize=11)

    # big verdict box
    fc, ec = (GREEN_BOX, GREEN_EDGE) if ok else (RED_BOX, RED_EDGE)
    mark = "✓ unbiased" if ok else "✗ biased"
    a.text(0.97, 0.03,
           f"$\\hat s$ = {shat:.3f}\nbias {bias:+.3f}\n{mark}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=13,
           fontweight="bold", color=ec,
           bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=2))

fig.suptitle("Assuming heterozygosity biases fitness inference when the data isn't",
             fontsize=19, fontweight="bold", y=0.99)
fig.savefig("zygosity_bias.png", bbox_inches="tight", dpi=200)
plt.show()
