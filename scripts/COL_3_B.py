"""
Column 3 poster plots
 Plot A: VAF dynamics — extended displacement model vs exponential, varying E and B for fixed s
 Plot B: Fitness recovery — minimal exponential model accuracy across E×B space
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os

os.makedirs("exports", exist_ok=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
K          = 1e5        # baseline niche size (wild-type cells at t=0)
T_MAX      = 150
N_POINTS   = 3000
VAF_THRESH = 0.02
x0_TRUE    = 1.0
N_OBS      = 6

# ── AESTHETICS ───────────────────────────────────────────────────────────────
INK       = "#0F172A"
BG        = "#FFFFFF"
BLOODRED  = "#8A0303"
HETBLUE   = "#2171B5"
C_EXP     = "#D94F00"   # exponential model — burnt orange
C_DISP    = "#1A6B3C"   # displacement model — forest green

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "text.color": INK,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.5,
    "axes.grid": True,
    "axes.axisbelow": True,
})

# ── ODE MODELS ───────────────────────────────────────────────────────────────

def exponential_vaf(t_arr, s, x0=x0_TRUE):
    """Minimal exponential model: clone grows freely against fixed background."""
    x = x0 * np.exp(s * t_arr)
    Nw = K  # constant
    return x / (2. * (Nw + x))


def _disp_rhs(t, y, s, E, B):
    x, Nw, F = y
    x  = max(x,  0.)
    Nw = max(Nw, 1e-9)
    F  = max(F,  0.)
    dxdt = s * x * (1. - (x + Nw) / E)
    sd = max(F + B * Nw, 1e-12)
    return [dxdt, -dxdt * B * Nw / sd, -dxdt * F / sd]


def displacement_vaf(t_arr, s, E, B, x0=x0_TRUE):
    """Full niche-competition displacement model."""
    n0 = K
    F0 = max(E - n0, 0.)
    sol = solve_ivp(
        _disp_rhs, [0., t_arr[-1] + 1.], [x0, n0, F0],
        args=(s, E, B), method='RK45', dense_output=True,
        max_step=0.5, rtol=1e-6, atol=1e-8
    )
    vals = sol.sol(t_arr)
    x, Nw, _ = np.maximum(vals, 0.)
    return x / (2. * (Nw + x + 1e-12))


# ── SYNTHETIC DATA (for plot B) ───────────────────────────────────────────────

def generate_synth(s, E, B, seed=42, n_obs=N_OBS):
    rng = np.random.default_rng(seed)
    t_arr = np.linspace(0., T_MAX, N_POINTS)
    vaf_true = displacement_vaf(t_arr, s, E, B)

    above = np.where(vaf_true >= VAF_THRESH)[0]
    if len(above) == 0:
        return None
    t_detect = t_arr[above[0]]

    dt = float(np.clip(1. / s, 1., 20.))
    t_obs = t_detect + np.arange(n_obs) * dt
    idx = np.array([np.argmin(np.abs(t_arr - to)) for to in t_obs])
    idx = np.unique(idx[t_obs <= T_MAX])
    if len(idx) < 2:
        return None

    DP  = np.maximum(rng.normal(5000., 1500., len(idx)).astype(int), 100)
    AO  = rng.binomial(DP, np.clip(vaf_true[idx], 1e-9, 1 - 1e-9))
    return dict(t=t_arr[idx], VAF=AO/DP, DP=DP, AO=AO)


def run_map(data):
    t_rel = data["t"] - data["t"][0]
    DP, AO = data["DP"].astype(float), data["AO"].astype(float)
    vaf0   = float(np.clip(data["VAF"][0], 1e-4, 0.499))
    x0i    = float(np.clip(2.*K*vaf0/(1.-2.*vaf0), 10., 4e5))

    # OLS init
    try:
        lv = np.log(np.clip(data["VAF"], 1e-6, None))
        w  = np.where(data["VAF"] < 0.3, 1., 0.2)
        A  = np.column_stack([np.ones_like(t_rel), t_rel])
        W  = np.diag(w)
        s_init = float(np.clip(np.linalg.lstsq(W@A, W@lv, rcond=None)[0][1], 0.02, 0.99))
    except Exception:
        s_init = 0.3

    def neg_ll(params):
        s, lx0 = params
        x = np.exp(lx0) * np.exp(s * t_rel)
        p = np.clip(x / (2.*(K + x)), 1e-12, 1-1e-12)
        return -np.sum(AO*np.log(p) + (DP-AO)*np.log(1-p))

    res = minimize(neg_ll, [s_init, np.log(x0i)], method='L-BFGS-B',
                   bounds=[(0.001, 0.999), (np.log(10.), np.log(5e5))],
                   options={'maxiter': 500, 'ftol': 1e-9})
    if not res.success:
        return None
    return float(res.x[0])


# ─────────────────────────────────────────────────────────────────────────────
# PLOT A: VAF DYNAMICS COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_A_vaf_dynamics(save_path="exports/col3_A_vaf_dynamics.pdf"):
    """
    3-panel figure for the poster.
    Panel 1: Vary E (B fixed, s fixed) — show how niche size alters trajectory
    Panel 2: Vary B (E fixed, s fixed) — show how displacement bias alters trajectory
    Panel 3: Side-by-side for a single realistic (E,B): exponential vs displacement
    """

    s_fixed = 0.3
    t_arr   = np.linspace(0., T_MAX, N_POINTS)

    # Colour ramps
    E_vals   = [2e5, 4e5, 6e5, 8e5, 1e6]
    B_vals   = [0.5, 1.0, 2.5, 5.0, 10.0]
    E_cmap   = plt.cm.Blues
    B_cmap   = plt.cm.Greens
    E_norm   = Normalize(vmin=0, vmax=len(E_vals)-1)
    B_norm   = Normalize(vmin=0, vmax=len(B_vals)-1)

    B_fixed = 2.5   # held constant in panel 1
    E_fixed = 6e5   # held constant in panel 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.84, bottom=0.15, wspace=0.38)

    # ── Panel 1: vary E ──────────────────────────────────────────────────────
    ax = axes[0]
    for i, E in enumerate(E_vals):
        col   = E_cmap(0.3 + 0.7 * i / (len(E_vals)-1))
        label = f"$E={E/K:.0f}$K"
        vaf_d = displacement_vaf(t_arr, s_fixed, E, B_fixed)
        ax.plot(t_arr, vaf_d, color=col, lw=2.0, label=label, zorder=3)

    # Exponential reference
    vaf_exp = exponential_vaf(t_arr, s_fixed)
    ax.plot(t_arr, vaf_exp, color=C_EXP, lw=2.0, ls="--",
            label="Exponential", zorder=4)

    ax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.5)
    ax.text(3, 0.515, "VAF = 0.5", fontsize=8, color=INK, alpha=0.6)
    ax.set_xlim(0, T_MAX); ax.set_ylim(0, 0.85)
    ax.set_xlabel("Time (yr)", fontsize=11)
    ax.set_ylabel("VAF", fontsize=11)
    ax.set_title(f"Effect of niche size $E$\n($s={s_fixed}$, $B={B_fixed}$)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
              edgecolor="#E2E8F0", ncol=1)

    # ── Panel 2: vary B ──────────────────────────────────────────────────────
    ax = axes[1]
    for i, B in enumerate(B_vals):
        col   = B_cmap(0.25 + 0.75 * i / (len(B_vals)-1))
        label = f"$B={B}$"
        vaf_d = displacement_vaf(t_arr, s_fixed, E_fixed, B)
        ax.plot(t_arr, vaf_d, color=col, lw=2.0, label=label, zorder=3)

    ax.plot(t_arr, vaf_exp, color=C_EXP, lw=2.0, ls="--",
            label="Exponential", zorder=4)

    ax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.5)
    ax.text(3, 0.515, "VAF = 0.5", fontsize=8, color=INK, alpha=0.6)
    ax.set_xlim(0, T_MAX); ax.set_ylim(0, 0.85)
    ax.set_xlabel("Time (yr)", fontsize=11)
    ax.set_ylabel("VAF", fontsize=11)
    ax.set_title(f"Effect of displacement bias $B$\n($s={s_fixed}$, $E={E_fixed/K:.0f}$K)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9,
              edgecolor="#E2E8F0", ncol=1)

    # ── Panel 3: s_hat comparison (exp vs displacement) ──────────────────────
    ax = axes[2]

    # Show realistic case: 4 different s values, compare exp model s_hat
    # to displacement model s_hat when data is generated from displacement
    s_test   = [0.1, 0.2, 0.4, 0.7]
    E_r, B_r = 6e5, 2.5
    markers  = ["o", "s", "^", "D"]

    ax.set_xlim(0, T_MAX); ax.set_ylim(0, 0.85)

    # overlay trajectories for each s
    s_cmap = plt.cm.plasma
    s_norm = Normalize(vmin=min(s_test), vmax=max(s_test))

    for s_val, mk in zip(s_test, markers):
        col     = s_cmap(s_norm(s_val))
        vaf_exp = exponential_vaf(t_arr, s_val)
        vaf_disp = displacement_vaf(t_arr, s_val, E_r, B_r)
        ax.plot(t_arr, vaf_disp, color=col, lw=2.2, zorder=3,
                label=f"$s={s_val}$ — displacement")
        ax.plot(t_arr, vaf_exp, color=col, lw=1.5, ls="--",
                alpha=0.55, zorder=2)

    ax.axhline(0.5, color=INK, lw=0.8, ls=":", alpha=0.5)
    ax.text(3, 0.515, "VAF = 0.5", fontsize=8, color=INK, alpha=0.6)

    # Legend entries for line styles
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="grey", lw=2.2, label="Displacement model"),
        Line2D([0], [0], color="grey", lw=1.5, ls="--", alpha=0.55,
               label="Exponential model"),
    ]
    sm = ScalarMappable(cmap=s_cmap, norm=s_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("$s$", fontsize=10)
    cb.set_ticks(s_test)

    ax.legend(handles=legend_lines, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#E2E8F0")
    ax.set_xlabel("Time (yr)", fontsize=11)
    ax.set_ylabel("VAF", fontsize=11)
    ax.set_title(f"Displacement vs Exponential VAF\n($E={E_r/K:.0f}$K, $B={B_r}$)",
                 fontsize=11, fontweight="bold")

    fig.suptitle(
        "Extended niche-competition model alters VAF dynamics relative to the exponential approximation",
        fontsize=12, fontweight="bold", color=INK, y=0.99
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"Plot A → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT B: FITNESS RECOVERY — scatter + heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_B_fitness_recovery(save_path="exports/col3_B_fitness_recovery.pdf"):
    """
    2-panel figure.
    Left:  s_true vs s_hat scatter for realistic (E,B) — coloured by s_true
    Right: heatmap of mean |rel error| across E×B grid (averaged over s values)
           with biological realistic region marked
    """

    s_values = list(np.round(np.arange(0.1, 1.1, 0.1), 2))
    E_values = [2e5, 4e5, 6e5, 8e5, 1e6]
    B_values = [0.5, 1.41, 2.5, 5.0, 10.0]

    def is_realistic(E, B):
        return (5e5 <= E <= 1e6) and (1.5 <= B <= 7.0)

    print("Running inference grid for Plot B...")
    all_results = []
    for E in E_values:
        for B in B_values:
            for s_true in s_values:
                seed = abs(hash((42, s_true, E, B))) % (2**31)
                data = generate_synth(s_true, E, B, seed=seed)
                if data is None:
                    continue
                s_hat = run_map(data)
                if s_hat is None:
                    continue
                rel_err = (s_hat - s_true) / s_true * 100.
                all_results.append(dict(
                    s_true=s_true, E=E, B=B, s_hat=s_hat,
                    rel_err=rel_err, realistic=is_realistic(E, B)
                ))
                print(f"  E={E/K:.0f}K B={B} s={s_true} → ŝ={s_hat:.3f} ({rel_err:+.1f}%)")

    realistic = [r for r in all_results if r["realistic"]]
    print(f"\nGrid: {len(all_results)} fits, {len(realistic)} in realistic region")

    fig = plt.figure(figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.15, wspace=0.40)
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ── Left panel: scatter ──────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[0])

    s_all   = sorted(set(r["s_true"] for r in realistic))
    s_cmap  = plt.cm.plasma
    s_norm  = Normalize(vmin=min(s_all), vmax=max(s_all))

    s_range = np.linspace(0.05, 1.05, 200)
    ax_sc.fill_between(s_range, s_range*0.9, s_range*1.1,
                       color="#DBEAFE", alpha=0.55, zorder=1, label="±10% band")
    ax_sc.plot(s_range, s_range, color=INK, lw=1.4, zorder=2, label="Identity")

    for r in realistic:
        col = s_cmap(s_norm(r["s_true"]))
        ax_sc.scatter(r["s_true"], r["s_hat"], color=col, s=60,
                      edgecolors="white", lw=0.7, zorder=4)

    sm = ScalarMappable(cmap=s_cmap, norm=s_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_sc, fraction=0.04, pad=0.03)
    cb.set_label("$s_{\\rm true}$", fontsize=10)

    n_rec  = sum(abs(r["rel_err"]) < 10 for r in realistic)
    biases = [abs(r["rel_err"]) for r in realistic]
    ax_sc.text(0.05, 0.97,
               f"{n_rec}/{len(realistic)} within ±10%\n"
               f"mean |error| = {np.mean(biases):.1f}%",
               transform=ax_sc.transAxes, fontsize=9.5, va="top",
               color="#1A4F8C",
               bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF",
                         ec="#2563EB", lw=1.0))

    ax_sc.set_xlim(0.05, 1.05); ax_sc.set_ylim(0.05, 1.05)
    ax_sc.set_xlabel("$s_{\\rm true}$", fontsize=12)
    ax_sc.set_ylabel("$\\hat{s}$ (exponential model)", fontsize=12)
    ax_sc.set_title("Fitness recovery — realistic parameters\n"
                    r"($5\times10^5 \leq E \leq 10^6$, $1.5 \leq B \leq 7$)",
                    fontsize=11, fontweight="bold")
    ax_sc.legend(fontsize=9, loc="lower right", framealpha=0.95,
                 edgecolor="#E2E8F0")

    # ── Right panel: heatmap ─────────────────────────────────────────────────
    ax_hm = fig.add_subplot(gs[1])

    E_arr = np.array(E_values)
    B_arr = np.array(B_values)
    grid  = np.full((len(E_arr), len(B_arr)), np.nan)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            vals = [abs(r["rel_err"]) for r in all_results
                    if np.isclose(r["E"], E) and np.isclose(r["B"], B)]
            if vals:
                grid[i, j] = np.mean(vals)

    hm_cmap = LinearSegmentedColormap.from_list(
        "bias", [(0.0, "#2A9D5C"), (0.30, "#B0D9B1"),
                 (0.50, "#FFFDE7"), (0.75, "#F4A261"), (1.0, "#C0392B")], N=256
    )
    im = ax_hm.imshow(grid, aspect="auto", origin="lower",
                      cmap=hm_cmap, vmin=0, vmax=50)
    cb2 = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cb2.set_label("Mean |relative error| in $\\hat{s}$ (%)", fontsize=9.5)
    cb2.set_ticks([0, 10, 20, 30, 40, 50])

    # Annotate cells
    for i in range(len(E_arr)):
        for j in range(len(B_arr)):
            v = grid[i, j]
            if not np.isnan(v):
                c = "white" if v > 28 else "black"
                ax_hm.text(j, i, f"{v:.0f}%", ha="center", va="center",
                           fontsize=8.5, color=c, fontweight="bold")

    # Highlight realistic region with gold border
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            if is_realistic(E, B):
                ax_hm.add_patch(mpatches.Rectangle(
                    (j - 0.48, i - 0.48), 0.96, 0.96,
                    fill=False, edgecolor="gold", lw=2.8, zorder=5
                ))

    B_labels = [f"{b:.2g}" for b in B_arr]
    E_labels = [f"${E/1e5:.0f}\\times10^5$" for E in E_arr]
    ax_hm.set_xticks(range(len(B_arr))); ax_hm.set_xticklabels(B_labels, fontsize=9)
    ax_hm.set_yticks(range(len(E_arr))); ax_hm.set_yticklabels(E_labels, fontsize=8.5)
    ax_hm.set_xlabel(r"Displacement bias $B$", fontsize=11)
    ax_hm.set_ylabel(r"Niche size $E$", fontsize=11)
    ax_hm.set_title("Accuracy across niche parameter space\n"
                    r"(gold border = biologically realistic)",
                    fontsize=11, fontweight="bold")
    ax_hm.grid(False)

    fig.suptitle(
        "Minimal exponential model accurately recovers fitness within the biologically realistic parameter regime",
        fontsize=12, fontweight="bold", color=INK, y=0.99
    )

    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"Plot B → {save_path}")
    plt.close()


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Plot A: VAF dynamics comparison ===")
    plot_A_vaf_dynamics()

    print("\n=== Plot B: Fitness recovery ===")
    plot_B_fitness_recovery()

    print("\nDone. Check exports/")