import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ----------------------------------------------------------------------
# Minimal deterministic model + het-only MLE.  Fit on EARLY data,
# then PREDICT forward.  het VAF caps at 0.5 -> catastrophic on hom.
# ----------------------------------------------------------------------
N_w, s_true, x0, depth = 1e5, 0.5, 1.0, 3000

fit_t = np.array([16.0, 18.0, 20.0])          # early: het looks fine here
val_t = np.array([26.0, 30.0, 34.0])          # later: the reveal

C = {"het": "#0072B2", "hom": "#E69F00"}

def vaf(t, s, m_eff, t0=0.0):
    x = x0 * np.exp(s * (t - t0))
    return m_eff * x / (2 * (N_w + x))

# het-only MLE (m_eff=1) fit to the EARLY points only
s_grid  = np.linspace(0.20, 1.20, 500)
t0_grid = np.linspace(-4.0, 14.0, 160)
SS, TT  = np.meshgrid(s_grid, t0_grid, indexing="ij")
Pfit = np.stack([np.clip(vaf(t, SS, 1.0, TT), 1e-9, 1-1e-9) for t in fit_t], -1)

def fit_het(a):
    ll = sum(binom.logpmf(a[k], depth, Pfit[..., k]) for k in range(len(fit_t)))
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    return s_grid[i], t0_grid[j]

# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 21, "axes.labelsize": 17,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 14,
    "axes.linewidth": 1.3, "lines.linewidth": 3.5,
})

t = np.linspace(0, 38, 1000)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

cases = [("Heterozygous clone", 1.0, "het", True),
         ("Homozygous clone",   2.0, "hom", False)]

for ax, (name, m_eff, ck, good) in zip(axes, cases):
    # data
    v_fit = vaf(fit_t, s_true, m_eff);  a_fit = np.round(depth*v_fit).astype(int)
    v_val = vaf(val_t, s_true, m_eff)
    s_hat, t0_hat = fit_het(a_fit)

    true_curve = vaf(t, s_true, m_eff)
    pred_curve = vaf(t, s_hat, 1.0, t0_hat)     # het extrapolation

    # het ceiling
    ax.axhline(0.5, color="0.4", ls=":", lw=2.0, zorder=1)
    ax.text(0.5, 0.515, "het model ceiling (VAF = 0.5)",
            color="0.4", fontsize=12, ha="left")

    # fit vs predict regions
    ax.axvspan(0, fit_t[-1], color="0.90", zorder=0)
    ax.text(fit_t[-1]/2, 0.96, "FIT", ha="center", color="0.45",
            fontsize=14, fontweight="bold")
    ax.text((fit_t[-1]+38)/2, 0.96, "PREDICT", ha="center", color="0.45",
            fontsize=14, fontweight="bold")

    # prediction-error wedge (only where het under-predicts, in predict zone)
    mask = t > fit_t[-1]
    ax.fill_between(t[mask], pred_curve[mask], true_curve[mask],
                    where=true_curve[mask] > pred_curve[mask],
                    color="#c02020", alpha=0.18, zorder=1)

    # curves
    ax.plot(t, true_curve, color=C[ck], label="Truth", zorder=3)
    ax.plot(t, pred_curve, "k--", lw=3.0, label="Het model", zorder=4)

    # observations
    ax.scatter(fit_t, v_fit, color=C[ck], s=110, edgecolor="k", lw=1.4,
               zorder=5, label="Fitted data")
    ax.scatter(val_t, v_val, marker="*", color=C[ck], s=320, edgecolor="k",
               lw=1.4, zorder=6, label="Future data")

    if not good:                                # arrow driving the point home
        ax.annotate("real clone keeps growing",
                    xy=(val_t[1], v_val[1]), xytext=(22, 0.85),
                    fontsize=13, color="#c02020", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#c02020", lw=2.2))
        ax.annotate("het prediction stalls\nat the ceiling",
                    xy=(34, vaf(34, s_hat, 1.0, t0_hat)), xytext=(24, 0.30),
                    fontsize=13, color="0.25", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="0.25", lw=2.2))

    ax.set_title(name + ("  ✓ prediction holds" if good
                         else "  ✗ prediction fails"),
                 color=("#1a7d1a" if good else "#c02020"), fontweight="bold")
    ax.set_xlabel("Time (years)")
    ax.set_xlim(0, 38); ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Variant Allele Frequency (VAF)")
fig.suptitle("Fit early, predict forward: the het-only model collapses on homozygous clones",
             fontsize=20, fontweight="bold", y=1.00)

plt.tight_layout()
fig.savefig("het_predict_fail.png", dpi=200, bbox_inches="tight")
print("SAVED:", __import__("os").path.abspath("het_predict_fail.png"))
plt.show()
