"""
Optimized CHIP fitness recovery pipeline — CLEAN VERSION (patched)
  - Parallel grid inference via multiprocessing
  - Direct scipy.optimize (no PyMC5 compilation overhead)
  - solve_ivp with dense_output for faster ODE integration
  - lru_cache on ODE solutions for plotting
  - All 4 publication-quality plots

Patches:
  * plot_realistic_slice now snaps E_real (and B_real) to the actual grid,
    fixing the empty-slice crash when a hardcoded target isn't a grid point.
  * E labels now use _format_E() everywhere (600K, 1M) instead of E/K "K".
  * Defensive guards added for empty slices / off-grid s_fixed.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from functools import lru_cache
from multiprocessing import Pool, cpu_count
import csv, os, time

os.makedirs("exports", exist_ok=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
K          = 1e5
T_MAX      = 200
N_POINTS   = 2000
VAF_THRESH = 0.05
x0_TRUE    = 1.0
N_OBS      = 5

# Biologically realistic parameter window (shared by heatmap + regime diagram)
E_REALISTIC = (5e5, 1e6)
B_REALISTIC = (1.5, 7.0)

# Plotting aesthetics
INK    = "#0F172A"
BG     = "#FFFFFF"
C_REAL = "#2563EB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "text.color": INK,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "grid.color": "#E2E8F0", "grid.linewidth": 0.5,
})


# ── HELPERS (defined early so all plotters can use them) ──────────────────────

def _format_E(E):
    """Human-readable niche-size label for talk slides: 200K, 1M, etc."""
    if E >= 1e6:
        v = E / 1e6
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    v = E / 1e3
    return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"


def _text_color_for(rgba):
    """Perceptual-luminance-based black/white text choice across colormaps."""
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.55 else "white"


def _is_realistic(E, B):
    return (E_REALISTIC[0] <= E <= E_REALISTIC[1]) and (B_REALISTIC[0] <= B <= B_REALISTIC[1])


def _snap(value, grid):
    """Snap a target value to the nearest available grid point."""
    return min(grid, key=lambda g: abs(g - value))


# ── TRUE (EXPANDED) ODE — solve_ivp version ──────────────────────────────────

def _ode_rhs(t, y, s, E, B):
    """Vectorized RHS for solve_ivp."""
    x, Nw, F = y
    x  = max(x, 0.)
    Nw = max(Nw, 1e-6)
    F  = max(F, 0.)
    dxdt = s * x * (1. - (x + Nw) / E)
    sd = max(F + Nw * B, 1e-6)
    return [dxdt, -dxdt * Nw * B / sd, -dxdt * F / sd]


def run_true_model(s, E, B, x0=x0_TRUE):
    """Fast ODE integration using solve_ivp with dense_output."""
    n0 = K
    F0 = max(E - n0, 0.)
    sol = solve_ivp(
        _ode_rhs, [0., T_MAX], [x0, n0, F0],
        args=(s, E, B),
        method='RK45',
        dense_output=True,
        max_step=1.0,
        rtol=1e-5, atol=1e-7
    )
    t = np.linspace(0., T_MAX, N_POINTS)
    x, Nw, F = sol.sol(t)
    return t, np.maximum(x, 0.), np.maximum(Nw, 1.), F


# ── CACHED ODE FOR PLOTTING ──────────────────────────────────────────────────

@lru_cache(maxsize=512)
def run_true_model_cached(s, E, B):
    """Deterministic ODE wrapper — safe to cache for repeated plotting calls."""
    return run_true_model(float(s), float(E), float(B))


# ── SYNTHETIC DATA GENERATION ─────────────────────────────────────────────────

def generate_synth(s, E, B, seed=None, n_obs=5):
    rng = np.random.default_rng(seed)
    t, x, Nw, F = run_true_model(s, E, B)

    denom    = 2. * (Nw + x)
    VAF_true = np.where(denom > 0., x / denom, 0.)
    VAF_true = np.clip(VAF_true, 1e-9, 1. - 1e-9)

    if not np.all(np.isfinite(VAF_true)):
        return None

    above = np.where(VAF_true >= VAF_THRESH)[0]
    if len(above) == 0:
        return None
    t_detect = t[above[0]]

    dt_sample = float(np.clip(1. / s, 1., 20.))
    t_obs = t_detect + np.arange(n_obs) * dt_sample

    idx = np.array([np.argmin(np.abs(t - to)) for to in t_obs])
    idx = np.unique(idx)
    if len(idx) < 2:
        return None

    DP_out  = np.maximum(rng.normal(5000., 2000., size=len(idx)).astype(int), 100)
    AO_out  = rng.binomial(DP_out, VAF_true[idx])
    VAF_out = AO_out / DP_out
    t_out   = t[idx]

    if len(t_out) < 2:
        return None

    return dict(t=t_out, VAF=VAF_out, DP=DP_out, AO=AO_out)


# ── FAST MAP INFERENCE — direct scipy.optimize ────────────────────────────────

def _estimate_s_ols(t_rel, VAF_obs):
    """OLS init for s."""
    try:
        lv = np.log(np.clip(VAF_obs, 1e-6, None))
        w  = np.where(VAF_obs < 0.3, 1., 0.2)
        A  = np.column_stack([np.ones_like(t_rel), t_rel])
        W  = np.diag(w)
        coef = np.linalg.lstsq(W @ A, W @ lv, rcond=None)[0]
        return float(np.clip(coef[1], 0.01, 0.99))
    except Exception:
        return 0.3


def run_map_fast(data):
    """
    Direct scipy.optimize.minimize — no PyMC5 compilation overhead.
    Uses exact same binomial log-likelihood as PyMC5.
    """
    t_rel = data["t"] - data["t"][0]
    DP    = data["DP"].astype(float)
    AO    = data["AO"].astype(float)

    vaf0    = float(np.clip(data["VAF"][0], 1e-4, 0.499))
    x0_init = float(np.clip(2. * K * vaf0 / (1. - 2. * vaf0), 10., 4e5))
    s_init  = _estimate_s_ols(t_rel, data["VAF"])

    def neg_loglike(params):
        s, log_x0 = params
        x0 = np.exp(log_x0)
        x_t = x0 * np.exp(s * t_rel)
        p = np.clip(x_t / (2. * (K + x_t)), 1e-12, 1. - 1e-12)
        # Binomial log-likelihood (constant terms omitted — don't affect optimum)
        ll = AO * np.log(p) + (DP - AO) * np.log(1. - p)
        return -np.sum(ll)

    res = minimize(
        neg_loglike,
        x0=[s_init, np.log(x0_init)],
        method='L-BFGS-B',
        bounds=[(0.001, 0.999), (np.log(10.), np.log(5e5))],
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    return float(res.x[0])


# ── PARALLEL GRID ─────────────────────────────────────────────────────────────

def _run_single(args):
    """Worker function for multiprocessing. Must be top-level."""
    E, B, s_true, seed_base = args
    seed = abs(hash((seed_base, s_true, E, B))) % (2**31)

    data = generate_synth(s_true, E, B, seed=seed, n_obs=N_OBS)
    if data is None:
        return None

    try:
        s_hat    = run_map_fast(data)
        rel_bias = (s_hat - s_true) / s_true * 100.
        abs_bias = s_hat - s_true
        recovered = abs(s_hat - s_true) / s_true < 0.10
        realistic = (E_REALISTIC[0] <= E <= E_REALISTIC[1]) and (B_REALISTIC[0] <= B <= B_REALISTIC[1])

        return dict(
            s_true    = s_true,
            E         = E,
            B         = B,
            s_hat     = s_hat,
            rel_bias  = rel_bias,
            abs_bias  = abs_bias,
            recovered = recovered,
            realistic = realistic,
        )
    except Exception as e:
        return dict(error=str(e), E=E, B=B, s_true=s_true)


def run_grid_parallel(E_values, B_values, s_values, n_workers=None, seed_base=42):
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    tasks = [(E, B, s, seed_base)
             for E in E_values for B in B_values for s in s_values]
    total = len(tasks)

    print(f"Running {total} fits on {n_workers} workers...")
    t0 = time.time()

    with Pool(n_workers) as pool:
        raw_results = pool.map(_run_single, tasks, chunksize=1)

    elapsed = time.time() - t0
    results = [r for r in raw_results if r is not None and "error" not in r]
    errors  = [r for r in raw_results if r is not None and "error" in r]
    skipped = total - len(raw_results) + len(errors)

    print(f"Done in {elapsed:.1f}s ({total/elapsed:.1f} fits/sec)")
    print(f"  Success: {len(results)}  Skipped/Failed: {skipped}")
    if errors:
        print(f"  First error: {errors[0]['error']}")

    return results


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def save_results_csv(results, path="exports/s_recovery_results.csv"):
    fields = ["s_true", "E", "B", "s_hat", "rel_bias", "abs_bias", "recovered", "realistic"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Results saved -> {path}")


def load_csv(path="exports/s_recovery_results.csv"):
    results = []
    with open(path) as f:
        for row in csv.DictReader(f):
            results.append({
                "s_true":    float(row["s_true"]),
                "E":         float(row["E"]),
                "B":         float(row["B"]),
                "s_hat":     float(row["s_hat"]),
                "rel_bias":  float(row["rel_bias"]),
                "abs_bias":  float(row["abs_bias"]),
                "recovered": row["recovered"] == "True",
                "realistic": row["realistic"] == "True",
            })
    return results


# ── PLOT 1: Realistic-parameter VAF slice ────────────────────────────────────

def plot_realistic_slice(results, E_real=6e5, B_real=2.512,
                         save_path="exports/plot1_realistic_slice.png"):

    # FIX: snap BOTH targets to the actual grid (E was never snapped -> crash)
    E_vals = sorted(set(r["E"] for r in results))
    B_vals = sorted(set(r["B"] for r in results))
    E_real = _snap(E_real, E_vals)
    B_real = _snap(B_real, B_vals)

    slice_r = sorted(
        [r for r in results
         if np.isclose(r["E"], E_real, rtol=1e-3)
         and np.isclose(r["B"], B_real, rtol=1e-3)],
        key=lambda r: r["s_true"]
    )

    # FIX: defensive guard instead of crashing on min([])
    if not slice_r:
        print(f"WARNING: no points at E={_format_E(E_real)}, B={B_real:.3g} — skipping slice plot")
        return
    print(f"Slice: {len(slice_r)} points at E={_format_E(E_real)}, B={B_real:.3g}")

    s_values = [r["s_true"] for r in slice_r]
    cmap     = plt.cm.plasma
    s_norm   = plt.Normalize(vmin=min(s_values), vmax=max(s_values))

    fig = plt.figure(figsize=(14, 5.5))
    gs  = gridspec.GridSpec(1, 2, left=0.06, right=0.97,
                            top=0.88, bottom=0.13, wspace=0.32)
    ax_vaf  = fig.add_subplot(gs[0])
    ax_scat = fig.add_subplot(gs[1])

    # Left: VAF trajectories
    ax_vaf.yaxis.grid(True, alpha=0.4); ax_vaf.set_axisbelow(True)

    for r in slice_r:
        s_t = r["s_true"]
        col = cmap(s_norm(s_t))

        t_f, x_f, Nw_f, _ = run_true_model_cached(s_t, E_real, B_real)
        d = 2.*(Nw_f + x_f)
        ax_vaf.plot(t_f, np.where(d>0., x_f/d, 0.),
                    color=col, lw=1.8, alpha=0.9, zorder=3)

        data = generate_synth(s_t, E_real, B_real, seed=42+int(s_t*1000))
        if data is not None:
            t_r  = data["t"] - data["t"][0]
            vaf0 = float(np.clip(data["VAF"][0], 1e-4, 0.499))
            x0f  = float(2.*K*vaf0/(1.-2.*vaf0))
            xf   = x0f * np.exp(r["s_hat"] * t_r)
            vf   = xf / (2.*(K + xf))
            ax_vaf.plot(data["t"], vf,
                        color=col, lw=2.8, alpha=0.45, zorder=2)
            ax_vaf.scatter(data["t"], data["VAF"],
                           color=col, s=22, zorder=5,
                           edgecolors="white", lw=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=s_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_vaf, fraction=0.035, pad=0.02)
    cb.set_label("$s_{\\rm true}$", fontsize=11)
    cb.set_ticks(s_values[::max(1, len(s_values)//6)])

    ax_vaf.plot([], [], color="grey", lw=1.8, label="True ODE trajectory")
    ax_vaf.plot([], [], color="grey", lw=2.8, alpha=0.45, label="Minimal model fit")
    ax_vaf.scatter([], [], color="grey", s=22, edgecolors="white",
                   lw=0.6, label="Observed samples")
    ax_vaf.legend(fontsize=9, loc="upper left", frameon=True,
                  framealpha=0.92, edgecolor="#E2E8F0")

    ax_vaf.set_xlim(0, T_MAX); ax_vaf.set_ylim(0, 1.0)
    ax_vaf.set_xlabel("Time (yr)", fontsize=12)
    ax_vaf.set_ylabel("VAF", fontsize=12)
    # FIX: use _format_E instead of E_real/K "K" (was printing 600K as "6K")
    ax_vaf.set_title(f"VAF trajectories — realistic parameters\n"
                     f"$E = {_format_E(E_real)}$,  $B = {B_real:.3g}$",
                     fontsize=12, fontweight="bold")

    # Right: s_true vs s_hat
    ax_scat.yaxis.grid(True, alpha=0.4); ax_scat.set_axisbelow(True)

    s_range = np.linspace(0.05, 1.05, 300)
    ax_scat.fill_between(s_range, s_range*0.9, s_range*1.1,
                         color="#DBEAFE", alpha=0.6, zorder=1, label="±10% band")
    ax_scat.plot(s_range, s_range, color=INK, lw=1.5, zorder=2, label="Identity")

    for r in slice_r:
        col = cmap(s_norm(r["s_true"]))
        ax_scat.scatter(r["s_true"], r["s_hat"],
                        color=col, s=75, zorder=5,
                        edgecolors="white", lw=0.8)

    n_rec  = sum(abs(r["rel_bias"]) < 10 for r in slice_r)
    biases = [abs(r["rel_bias"]) for r in slice_r]
    ax_scat.text(0.05, 0.97,
                 f"{n_rec}/{len(slice_r)} within ±10%\n"
                 f"mean |bias| = {np.mean(biases):.1f}%\n"
                 f"max  |bias| = {np.max(biases):.1f}%",
                 transform=ax_scat.transAxes,
                 fontsize=10, va="top", color=C_REAL,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF",
                           ec=C_REAL, lw=1.1))

    ax_scat.set_xlim(0.05, 1.05); ax_scat.set_ylim(0.05, 1.05)
    ax_scat.set_xlabel("$s_{\\rm true}$", fontsize=13)
    ax_scat.set_ylabel("$\\hat{s}$", fontsize=13)
    ax_scat.set_title("Fitness recovery — realistic parameters",
                      fontsize=12, fontweight="bold")
    ax_scat.legend(fontsize=10, loc="lower right", frameon=True,
                   framealpha=0.95, edgecolor="#E2E8F0")

    fig.suptitle(
        "Minimal exponential model accurately recovers fitness at biologically realistic HSC niche parameters",
        fontsize=12, fontweight="bold", color=INK, y=0.99)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot 1 -> {save_path}")
    plt.close()


# ── PLOT 2: Recovery heatmap (single panel, talk version) ────────────────────

def plot_heatmap(results, save_path="exports/plot2_heatmap.png"):
    """
    Single-panel recovery-rate heatmap for the talk slide: for each
    (E, B) cell, the fraction of s_true values recovered within 10%.
    """
    B_arr = np.array(sorted(set(r["B"] for r in results)))
    E_arr = np.array(sorted(set(r["E"] for r in results)))

    rec10_g = np.full((len(E_arr), len(B_arr)), np.nan)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            vals = [abs(r["rel_bias"]) < 10. for r in results
                    if np.isclose(r["E"], E) and np.isclose(r["B"], B)]
            if vals:
                rec10_g[i, j] = np.mean(vals) * 100.

    B_labels = [f"{b:.2g}" for b in B_arr]
    E_labels = [_format_E(E) for E in E_arr]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.suptitle("Fitness recovery within 10% across niche parameter space",
                 fontsize=13, fontweight="bold", y=0.98)

    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=0, vmax=100)
    im = ax.imshow(rec10_g, aspect="auto", origin="lower",
                    cmap=cmap, norm=norm, interpolation="bicubic")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Recovery rate (%)", fontsize=10)

    for i in range(len(E_arr)):
        for j in range(len(B_arr)):
            v = rec10_g[i, j]
            if not np.isnan(v):
                col = _text_color_for(cmap(norm(v)))
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(range(len(B_arr))); ax.set_xticklabels(B_labels, fontsize=9)
    ax.set_yticks(range(len(E_arr))); ax.set_yticklabels(E_labels, fontsize=9)
    ax.set_xlabel(r"$B$  (wildtype displacement bias)", fontsize=10)
    ax.set_ylabel(r"$E$  (effective niche size)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot 2 -> {save_path}")
    plt.close()


def plot_heatmap_error_backup(results, save_path="exports/plot2b_heatmap_error_backup.png"):
    """Mean-relative-error panel, kept as a standalone Q&A backup slide."""
    B_arr = np.array(sorted(set(r["B"] for r in results)))
    E_arr = np.array(sorted(set(r["E"] for r in results)))

    bias_g = np.full((len(E_arr), len(B_arr)), np.nan)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            vals = [r["rel_bias"] for r in results
                    if np.isclose(r["E"], E) and np.isclose(r["B"], B)]
            if vals:
                bias_g[i, j] = np.mean(vals)

    B_labels = [f"{b:.2g}" for b in B_arr]
    E_labels = [_format_E(E) for E in E_arr]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    fig.suptitle("Mean relative error in $\\hat{s}$ across niche parameter space",
                 fontsize=13, fontweight="bold", y=0.98)

    cmap = plt.cm.RdBu_r
    vabs = min(max(abs(np.nanmin(bias_g)), abs(np.nanmax(bias_g))), 60.)
    norm = mcolors.Normalize(vmin=-vabs, vmax=vabs)
    im = ax.imshow(bias_g, aspect="auto", origin="lower",
                    cmap=cmap, norm=norm, interpolation="bicubic")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Error in $\\hat{s}$ (%)", fontsize=10)

    for i in range(len(E_arr)):
        for j in range(len(B_arr)):
            v = bias_g[i, j]
            if not np.isnan(v):
                col = _text_color_for(cmap(norm(v)))
                ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                        fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(range(len(B_arr))); ax.set_xticklabels(B_labels, fontsize=9)
    ax.set_yticks(range(len(E_arr))); ax.set_yticklabels(E_labels, fontsize=9)
    ax.set_xlabel(r"$B$  (wildtype displacement bias)", fontsize=10)
    ax.set_ylabel(r"$E$  (effective niche size)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot 2 (backup) -> {save_path}")
    plt.close()


# ── PLOT 3: VAF examples ──────────────────────────────────────────────────────

def plot_vaf_examples(results, s_fixed=0.3, seed=42,
                      save_path="exports/plot3_vaf_examples.png"):
    E_vals = sorted(set(r["E"] for r in results))
    B_vals = sorted(set(r["B"] for r in results))
    s_vals = sorted(set(r["s_true"] for r in results))

    # FIX: snap s_fixed to the grid so the fit-lookup below can't silently miss
    s_fixed = _snap(s_fixed, s_vals)

    E_show = [E_vals[0], E_vals[len(E_vals)//2], E_vals[-1]]
    B_show = [B_vals[0], B_vals[-1]]

    fig, axes = plt.subplots(len(E_show), len(B_show),
                             figsize=(10, 9), sharex=False, sharey=False)
    fig.suptitle(
        f"Example VAF trajectories  ($s = {s_fixed:.2g}$)\\n"
        r"True ODE (line) $\cdot$ Observed samples (dots) $\cdot$ "
        r"Minimal model fit $\hat{s}$ (dashed)",
        fontsize=12, fontweight="bold"
    )

    for row, E in enumerate(E_show):
        for col, B in enumerate(B_show):
            ax = axes[row][col]
            t_full, x_full, Nw_full, _ = run_true_model_cached(s_fixed, E, B)
            denom_full = 2. * (Nw_full + x_full)
            VAF_full   = np.where(denom_full > 0., x_full / denom_full, 0.)
            ax.plot(t_full, VAF_full, color="steelblue", lw=1.8,
                    label="True VAF (ODE)", zorder=2)

            data = generate_synth(s_fixed, E, B, seed=seed, n_obs=N_OBS)
            if data is not None:
                ax.scatter(data["t"], data["VAF"],
                           color="tomato", s=22, zorder=3,
                           alpha=0.85, edgecolor="k", lw=0.3,
                           label="Observed (binomial)")
                try:
                    match = [r for r in results
                             if np.isclose(r["E"], E) and np.isclose(r["B"], B)
                             and np.isclose(r["s_true"], s_fixed)]
                    s_hat = match[0]["s_hat"] if match else run_map_fast(data)

                    t_fit   = data["t"] - data["t"][0]
                    vaf0    = float(np.clip(data["VAF"][0], 1e-4, 0.499))
                    x0_fit  = float(2. * K * vaf0 / (1. - 2. * vaf0))
                    x_fit   = x0_fit * np.exp(s_hat * t_fit)
                    vaf_fit = x_fit / (2. * (K + x_fit))
                    ax.plot(data["t"], vaf_fit, color="darkorange",
                            lw=1.6, ls="--", zorder=4,
                            label=f"Minimal fit  $\\hat{{s}}={s_hat:.2f}$")
                except Exception:
                    pass

            # FIX: use _format_E for the per-panel title too
            ax.set_title(f"E = {_format_E(E)},  B = {B:.2g}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Time (yr)", fontsize=9)
            ax.set_ylabel("VAF", fontsize=9)
            ax.set_ylim(0, None)
            ax.legend(fontsize=7.5, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot 3 -> {save_path}")
    plt.close()


# ── PLOT 4: Regime diagram ────────────────────────────────────────────────────

def plot_regime_diagram(results, save_path="exports/plot4_regime_diagram.png"):
    B_arr = np.array(sorted(set(r["B"] for r in results)))
    E_arr = np.array(sorted(set(r["E"] for r in results)))
    nB, nE = len(B_arr), len(E_arr)

    grid = np.full((nE, nB), np.nan)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            vals = [abs(r["rel_bias"]) for r in results
                    if np.isclose(r["E"], E) and np.isclose(r["B"], B)]
            if vals:
                grid[i, j] = np.mean(vals)

    cmap = LinearSegmentedColormap.from_list(
        "bias_cmap",
        [(0.00, "#4CAF7D"), (0.25, "#A8D8A8"),
         (0.45, "#FFFDE7"), (0.70, "#FFAB76"), (1.00, "#C0392B")],
        N=256
    )
    norm = mcolors.Normalize(vmin=0, vmax=60)

    fig = plt.figure(figsize=(8.5, 5.8))
    ax  = fig.add_axes([0.18, 0.18, 0.60, 0.68])
    cax = fig.add_axes([0.82, 0.18, 0.03, 0.68])

    cell_w = 0.88
    for i in range(nE):
        for j in range(nB):
            v = grid[i, j]
            color = cmap(norm(v)) if not np.isnan(v) else "#DDDDDD"
            rect = mpatches.FancyBboxPatch(
                (j - cell_w/2, i - cell_w/2), cell_w, cell_w,
                boxstyle="round,pad=0.06",
                facecolor=color, edgecolor="white", linewidth=2.5,
                zorder=2, transform=ax.transData)
            ax.add_patch(rect)

    # Biologically realistic region highlight (gold border on cells)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            if _is_realistic(E, B):
                rect = mpatches.FancyBboxPatch(
                    (j - cell_w/2, i - cell_w/2), cell_w, cell_w,
                    boxstyle="round,pad=0.02",
                    facecolor="none", edgecolor="gold", linewidth=3.0,
                    zorder=3, transform=ax.transData)
                ax.add_patch(rect)

    ax.set_xlim(-0.6, nB - 0.4)
    ax.set_ylim(-0.6, nE - 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    cb = mcolorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Mean |error| in " + r"$\hat{s}$" + " (%)",
                 fontsize=10, labelpad=8)
    cb.set_ticks([0, 20, 40, 60])
    cb.ax.tick_params(labelsize=9)
    cax.axhline(norm(20), color="white", lw=1.5, linestyle="--", alpha=0.8)

    B_labels = [f"{b:.2g}" for b in B_arr]
    for j, bl in enumerate(B_labels):
        ax.text(j, -0.75, bl, ha="center", va="top",
                fontsize=10, color="#333333", transform=ax.transData)

    ax.annotate("", xy=(nB - 0.4, -1.35), xytext=(-0.4, -1.35),
                xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3))
    ax.text((nB - 1) / 2, -1.72,
            r"Wildtype displacement bias  $B$",
            ha="center", va="top", fontsize=11, color="#333333",
            transform=ax.transData)

    E_labels = [_format_E(E) for E in E_arr]
    for i, el in enumerate(E_labels):
        ax.text(-0.75, i, el, ha="right", va="center",
                fontsize=10, color="#333333", transform=ax.transData)

    ax.annotate("", xy=(-1.35, nE - 0.4), xytext=(-1.35, -0.4),
                xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3))
    # FIX: readable axis label (old one was an unparseable "E/1e5 x ..." string)
    ax.text(-1.72, (nE - 1) / 2,
            r"Effective niche size  $E$",
            ha="center", va="center", fontsize=11, color="#333333",
            rotation=90, transform=ax.transData)

    fig.text(0.49, 0.94,
             "Minimal model accuracy across niche parameter space",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="#222222")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Plot 4 -> {save_path}")
    plt.close()


# ── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    E_values = list(np.linspace(2e5, 1e6, 7))
    B_values = list(np.round(np.logspace(np.log10(0.8), np.log10(10.), 7), 3))
    s_values = list(np.round(np.arange(0.1, 1.1, 0.1), 2))

    print(f"Grid: {len(E_values)} E x {len(B_values)} B x {len(s_values)} s"
          f" = {len(E_values)*len(B_values)*len(s_values)} fits")
    print(f"B values: {B_values}\n")

    # Phase 1: Parallel inference
    results = run_grid_parallel(E_values, B_values, s_values)

    print(f"\n{len(results)} fits completed.")
    save_results_csv(results)

    # Phase 2: Generate plots
    print("\n--- Generating plots ---")
    plot_realistic_slice(results)
    plot_heatmap(results)
    plot_heatmap_error_backup(results)
    plot_vaf_examples(results)
    plot_regime_diagram(results)

    print("\nAll done. Check exports/ directory for outputs.")
