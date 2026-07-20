"""
sensitivity_benchmark.py
------------------------
Tests the deterministic (det) inference pipeline across three axes derived
from the real MDS cohort data:

  1. Total follow-up (delta_t):   0.60y  /  3.0y  /  6.63y
  2. Number of timepoints:        2      /  4      /  9
  3. Initial VAF at t=0:          0.01   /  0.05   /  0.15

For each axis, the other two variables are held at their midpoint values.
Within each scenario, s_true and h_true are varied on a small grid.

Timepoints are randomly sampled (without replacement) from a uniform
distribution over [0, delta_t * (n_tps - 1)], then sorted — reflecting
irregular real sampling rather than fixed intervals.

Outputs
-------
  sensitivity_results.csv         — per-cell records
  sensitivity_heatmaps.png        — one heatmap block per scenario
  sensitivity_vaf_trajectories.png — VAF trajectories per scenario
"""

import sys
import itertools
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

sys.path.append("..")
sys.path.append("../src")

# =============================================================================
# Import inference pipeline
# =============================================================================

from KI_clonal_inference_6 import infer_sh_jointly_from_dynamics

# =============================================================================
# True-value grid (shared across all scenarios)
# =============================================================================

S_TRUE_VALUES = np.round([0.1, 0.3, 0.6, 1.0], 2)
H_TRUE_VALUES = np.round([0.0, 0.3, 0.6, 1.0], 2)

# =============================================================================
# Scenario definitions
# Real-data ranges:
#   delta_t      : 0.60 – 6.63y   (MDS1135H55 – MDS918V64)
#   n_timepoints : 2 – 9
#   initial_vaf  : estimated 0.01 – 0.15
#
# Midpoints used when a variable is held fixed:
#   delta_t_mid  = 3.0y
#   n_tps_mid    = 4
#   vaf0_mid     = 0.05
#
# delta_t is the gap between consecutive timepoints.
# Total follow-up span = delta_t * (n_tps - 1).
# =============================================================================

DELTA_T_MID  = 3.0
N_TPS_MID    = 4
VAF0_MID     = 0.05

SCENARIOS = [
    # --- Axis 1: inter-timepoint gap ---
    dict(label="delta_t=0.60y (worst)",  delta_t=0.60,  n_tps=N_TPS_MID,  vaf0=VAF0_MID,  axis="delta_t"),
    dict(label="delta_t=3.0y  (mid)",   delta_t=3.0,   n_tps=N_TPS_MID,  vaf0=VAF0_MID,  axis="delta_t"),
    dict(label="delta_t=6.63y (best)",  delta_t=6.63,  n_tps=N_TPS_MID,  vaf0=VAF0_MID,  axis="delta_t"),

    # --- Axis 2: number of timepoints ---
    dict(label="n_tps=2 (worst)",       delta_t=DELTA_T_MID,  n_tps=2,  vaf0=VAF0_MID,   axis="n_tps"),
    dict(label="n_tps=4 (mid)",         delta_t=DELTA_T_MID,  n_tps=4,  vaf0=VAF0_MID,   axis="n_tps"),
    dict(label="n_tps=9 (best)",        delta_t=DELTA_T_MID,  n_tps=9,  vaf0=VAF0_MID,   axis="n_tps"),

    # --- Axis 3: initial VAF ---
    dict(label="vaf0=0.01 (worst)",     delta_t=DELTA_T_MID,  n_tps=N_TPS_MID,  vaf0=0.01,  axis="vaf0"),
    dict(label="vaf0=0.05 (mid)",       delta_t=DELTA_T_MID,  n_tps=N_TPS_MID,  vaf0=0.05,  axis="vaf0"),
    dict(label="vaf0=0.15 (best)",      delta_t=DELTA_T_MID,  n_tps=N_TPS_MID,  vaf0=0.15,  axis="vaf0"),
]

# =============================================================================
# Fixed inference / simulation settings
# =============================================================================

N_W          = 1e5
DEPTH        = 500       # representative of real MDS sequencing depth
SEED         = 42
S_RESOLUTION = 30
H_RESOLUTION = 20

OUTPUT_CSV      = "sensitivity_results.csv"
OUTPUT_FIG      = "sensitivity_heatmaps.png"
OUTPUT_VAF_FIG  = "sensitivity_vaf_trajectories.png"

# =============================================================================
# Synthetic data generation
# =============================================================================

def make_timepoints(total_span, n_tps, rng):
    """Sample n_tps timepoints uniformly from [0, total_span], sorted.

    total_span = delta_t * (n_tps - 1), so delta_t is the mean gap.

    Special case: n_tps=2 always uses [0, total_span] exactly so that the
    total follow-up is fully used even with only 2 measurements.
    """
    if n_tps == 1:
        return np.array([0.0])
    if n_tps == 2:
        return np.array([0.0, total_span])
    # Random interior points, always anchored at 0
    interior = rng.uniform(0.0, total_span, size=n_tps - 1)
    return np.sort(np.concatenate([[0.0], interior]))


def simulate_vaf(time_points, s, h, x0_tot, N_w=N_W):
    t     = np.asarray(time_points, dtype=float)
    x_tot = x0_tot * np.exp(s * (t - t[0]))
    x_het = (1 - h) * x_tot
    x_hom = h * x_tot
    vaf   = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    return np.clip(vaf, 1e-8, (1.0 + h) / 2.0 - 1e-8)


def x0_tot_from_vaf0(vaf0, h, N_w=N_W):
    # vaf0 = (1+h)*x_tot0 / (2*(N_w + x_tot0))
    # solving for x_tot0:
    x_tot0 = 2 * vaf0 * N_w / ((1 + h) - 2 * vaf0)
    return x_tot0


def generate_participant(s_true, h_true, delta_t, n_tps, vaf0, depth, seed):
    rng = np.random.default_rng(seed)

    total_span  = delta_t * (n_tps - 1)
    time_points = make_timepoints(total_span, n_tps, rng)
    x0_tot      = x0_tot_from_vaf0(vaf0, h_true)
    vaf_true    = simulate_vaf(time_points, s_true, h_true, x0_tot)

    DP = np.full((1, len(time_points)), depth, dtype=float)
    AO = rng.binomial(int(depth), vaf_true).reshape(1, -1).astype(float)

    return AO, DP, time_points, vaf_true

# =============================================================================
# Run inference on one cell
# =============================================================================

def _empty_record(scenario_label, axis, s_true, h_true, msg):
    return dict(
        scenario=scenario_label, axis=axis,
        s_true=s_true, h_true=h_true,
        s_inf=np.nan,  h_inf=np.nan,
        s_err=np.nan,  h_err=np.nan,
        s_lo=np.nan,   s_hi=np.nan,
        h_lo=np.nan,   h_hi=np.nan,
        s_in_ci=np.nan, h_in_ci=np.nan,
        elapsed_s=np.nan,
        status=f"error: {msg}",
    )


def run_cell(scenario, s_true, h_true, cell_seed):
    label    = scenario["label"]
    axis     = scenario["axis"]
    delta_t  = scenario["delta_t"]
    n_tps    = scenario["n_tps"]
    vaf0     = scenario["vaf0"]

    try:
        AO, DP, time_points, vaf_true = generate_participant(
            s_true, h_true, delta_t, n_tps, vaf0, DEPTH, cell_seed
        )

        # det pipeline expects (n_tps, n_muts) layout
        AO_T = AO.T
        DP_T = DP.T
        cs   = [[0]]

        t0 = time.perf_counter()
        results = infer_sh_jointly_from_dynamics(
            cs, AO_T, DP_T, time_points,
            s_resolution=S_RESOLUTION,
            h_resolution=H_RESOLUTION,
        )
        elapsed = time.perf_counter() - t0

        r     = results[0]
        s_map = r["s_map"]
        h_map = r["h_map"]
        s_ci  = r["s_ci"]
        h_ci  = r["h_ci"]

        return dict(
            scenario=label, axis=axis,
            s_true=s_true,  h_true=h_true,
            s_inf=s_map,    h_inf=h_map,
            s_err=s_map - s_true,
            h_err=h_map - h_true,
            s_lo=s_ci[0],  s_hi=s_ci[1],
            h_lo=h_ci[0],  h_hi=h_ci[1],
            s_in_ci=int(s_ci[0] <= s_true <= s_ci[1]),
            h_in_ci=int(h_ci[0] <= h_true <= h_ci[1]),
            elapsed_s=elapsed,
            status="ok",
        )

    except Exception as e:
        traceback.print_exc()
        return _empty_record(label, axis, s_true, h_true, str(e))


# =============================================================================
# Main grid runner
# =============================================================================

def run_benchmark():
    records   = []
    true_grid = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))
    n_total   = len(SCENARIOS) * len(true_grid)

    print(f"\n{len(SCENARIOS)} scenarios × {len(true_grid)} (s,h) cells = {n_total} runs\n")

    run_idx = 0
    for scen in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  {scen['label']}")
        total_span = scen['delta_t'] * (scen['n_tps'] - 1)
        print(f"  delta_t={scen['delta_t']}y  n_tps={scen['n_tps']}  "
              f"total_span={total_span:.2f}y  vaf0={scen['vaf0']}")
        print(f"{'='*60}")

        for s_true, h_true in true_grid:
            run_idx += 1
            cell_seed = SEED + run_idx
            rec = run_cell(scen, s_true, h_true, cell_seed)

            tag = "✓" if rec["status"] == "ok" else "✗"
            if rec["status"] == "ok":
                print(
                    f"  [{run_idx:3d}/{n_total}] {tag}"
                    f"  s={s_true:.2f} h={h_true:.2f}"
                    f"  → s_inf={rec['s_inf']:.3f} ({rec['s_err']:+.3f})"
                    f"  h_inf={rec['h_inf']:.3f} ({rec['h_err']:+.3f})"
                    f"  s_CI={'✓' if rec['s_in_ci'] else '✗'}"
                    f"  h_CI={'✓' if rec['h_in_ci'] else '✗'}"
                    f"  {rec['elapsed_s']:.1f}s"
                )
            else:
                print(f"  [{run_idx:3d}/{n_total}] {tag}  s={s_true:.2f} h={h_true:.2f}  {rec['status']}")

            records.append(rec)

    return pd.DataFrame(records)


# =============================================================================
# Summary statistics
# =============================================================================

def print_summary(df):
    print("\n" + "=" * 70)
    print("SENSITIVITY SUMMARY")
    print("=" * 70)

    for axis in ["delta_t", "n_tps", "vaf0"]:
        print(f"\n  Axis: {axis}")
        print(f"  {'Scenario':<30}  {'s MAE':>6}  {'h MAE':>6}  "
              f"{'s bias':>7}  {'h bias':>7}  "
              f"{'s CI%':>6}  {'h CI%':>6}  {'n':>4}")

        axis_scens = [s for s in SCENARIOS if s["axis"] == axis]
        for scen in axis_scens:
            ok = df[(df["scenario"] == scen["label"]) & (df["status"] == "ok")]
            if ok.empty:
                print(f"  {scen['label']:<30}  (no successful runs)")
                continue

            s_cov = ok["s_in_ci"].mean() * 100
            h_cov = pd.to_numeric(ok["h_in_ci"], errors="coerce").mean() * 100

            print(
                f"  {scen['label']:<30}"
                f"  {ok['s_err'].abs().mean():>6.3f}"
                f"  {ok['h_err'].abs().mean():>6.3f}"
                f"  {ok['s_err'].mean():>+7.3f}"
                f"  {ok['h_err'].mean():>+7.3f}"
                f"  {s_cov:>6.1f}"
                f"  {h_cov:>6.1f}"
                f"  {len(ok):>4}"
            )

    print("=" * 70)


# =============================================================================
# Plotting — heatmaps
# =============================================================================

def pivot(df, value_col):
    return df.pivot(index="s_true", columns="h_true", values=value_col)


def draw_heatmap(ax, mat, title, cmap, vmin=None, vmax=None,
                 centered=False, fmt="{:.2f}", annotate=True):
    kwargs = dict(cmap=cmap, aspect="auto")
    if centered:
        v = max(abs(vmin), abs(vmax))
        kwargs["norm"] = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    im = ax.imshow(
        mat.values, origin="lower",
        extent=[
            mat.columns.min() - 0.075, mat.columns.max() + 0.075,
            mat.index.min()   - 0.075, mat.index.max()   + 0.075,
        ],
        **kwargs,
    )
    if annotate:
        for s in mat.index:
            for h in mat.columns:
                val = mat.loc[s, h]
                if np.isnan(val):
                    continue
                ax.text(h, s, fmt.format(val),
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val) > 0.4 else "black")

    ax.set_xlabel("True  h", fontsize=7)
    ax.set_ylabel("True  s", fontsize=7)
    ax.set_title(title, fontsize=7.5, fontweight="bold")
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


def plot_sensitivity(df, output_path):
    """
    Layout: one row of 4 panels per scenario, grouped by axis.

    Panels per scenario:
      [s error]  [h error]  [s in CI]  [h in CI]
    """
    n_scen = len(SCENARIOS)
    fig    = plt.figure(figsize=(22, 4.5 * n_scen))
    gs     = gridspec.GridSpec(n_scen, 4, figure=fig,
                               hspace=0.55, wspace=0.35)

    axis_colours = {"delta_t": "#2271B2", "n_tps": "#3DAA6A", "vaf0": "#D44D3A"}

    for row, scen in enumerate(SCENARIOS):
        label = scen["label"]
        axis  = scen["axis"]
        sub   = df[(df["scenario"] == label) & (df["status"] == "ok")].copy()

        if sub.empty:
            for col in range(4):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
                ax.set_title(label, fontsize=7)
            continue

        axes_list = [fig.add_subplot(gs[row, c]) for c in range(4)]

        draw_heatmap(
            axes_list[0], pivot(sub, "s_err"),
            f"{label}\ns error  (inf − true)",
            "RdBu_r", -0.5, 0.5, centered=True,
        )
        draw_heatmap(
            axes_list[1], pivot(sub, "h_err"),
            "h error  (inf − true)",
            "RdBu_r", -0.5, 0.5, centered=True,
        )
        draw_heatmap(
            axes_list[2], pivot(sub, "s_in_ci"),
            "s in 90% CI  (1=yes)",
            "RdYlGn", 0, 1, fmt="{:.0f}",
        )
        draw_heatmap(
            axes_list[3], pivot(sub, "h_in_ci"),
            "h in 90% CI  (1=yes)",
            "RdYlGn", 0, 1, fmt="{:.0f}",
        )

        # Colour-coded spine to indicate which axis this row belongs to
        for ax in axes_list:
            for spine in ax.spines.values():
                spine.set_edgecolor(axis_colours[axis])
                spine.set_linewidth(1.8)

    # Axis legend
    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3, label=k)
        for k, c in axis_colours.items()
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right", fontsize=8,
        title="Varied axis", title_fontsize=8,
        framealpha=0.8,
    )

    fig.suptitle(
        "Sensitivity benchmark — det pipeline\n"
        "Rows grouped by axis; other variables held at midpoint "
        f"(delta_t={DELTA_T_MID}y, n_tps={N_TPS_MID}, vaf0={VAF0_MID})\n"
        "delta_t = inter-timepoint gap; total span = delta_t × (n_tps − 1)",
        fontsize=12, fontweight="bold", y=1.005,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.show()


# =============================================================================
# Plotting — VAF trajectories for OAT benchmark
# =============================================================================

def plot_vaf_trajectories(df, output_path):
    """
    One figure per axis (delta_t, n_tps, vaf0).
    Within each figure: one column per scenario, one row per (s_true, h_true)
    combination drawn from a representative subset of the true-value grid.

    Each panel shows:
      - True VAF trajectory (solid blue)
      - Inferred MAP trajectory (dashed red)
      - Observed noisy data points (blue dots with Wilson CI error bars)
      - VAF ceiling for true h (grey dotted)

    Uses the same seed as run_benchmark so trajectories match the inference.
    """
    # Representative subset: 3 s × 3 h to keep the figure readable
    s_subset = [0.1, 0.3, 1.0]
    h_subset = [0.0, 0.3, 1.0]
    true_grid_full = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))

    axis_colours = {"delta_t": "#2271B2", "n_tps": "#3DAA6A", "vaf0": "#D44D3A"}

    for axis in ["delta_t", "n_tps", "vaf0"]:
        axis_scens = [s for s in SCENARIOS if s["axis"] == axis]
        n_scens    = len(axis_scens)
        n_rows     = len(s_subset) * len(h_subset)   # 9 panels per scenario column

        fig, axes = plt.subplots(
            n_rows, n_scens,
            figsize=(4.5 * n_scens, 3.0 * n_rows),
            squeeze=False,
        )

        for col, scen in enumerate(axis_scens):
            label   = scen["label"]
            delta_t = scen["delta_t"]
            n_tps   = scen["n_tps"]
            vaf0    = scen["vaf0"]
            total_span = delta_t * (n_tps - 1)

            # Column header
            axes[0, col].set_title(
                f"{label}\nspan={total_span:.1f}y  n_tps={n_tps}  vaf0={vaf0}",
                fontsize=8, fontweight="bold",
                color=axis_colours[axis],
            )

            panel_idx = 0
            for s_true in s_subset:
                for h_true in h_subset:
                    ax = axes[panel_idx, col]

                    # Recover the same seed used in run_benchmark
                    grid_pos  = true_grid_full.index((s_true, h_true))
                    scen_pos  = SCENARIOS.index(scen)
                    run_idx   = scen_pos * len(true_grid_full) + grid_pos + 1
                    cell_seed = SEED + run_idx

                    # Regenerate the same synthetic data
                    AO, DP, time_points, vaf_true = generate_participant(
                        s_true, h_true, delta_t, n_tps, vaf0, DEPTH, cell_seed
                    )
                    obs_vaf = (AO / np.maximum(DP, 1.0))[0]

                    # True trajectory on a dense grid
                    t_dense  = np.linspace(time_points[0], time_points[-1], 300)
                    x0_tot   = x0_tot_from_vaf0(vaf0, h_true)
                    vaf_dense = simulate_vaf(t_dense, s_true, h_true, x0_tot)
                    ax.plot(t_dense, vaf_dense, color="steelblue", lw=1.8,
                            label="True")

                    # VAF ceiling
                    v_ceil = (1.0 + h_true) / 2.0
                    ax.axhline(v_ceil, color="grey", lw=0.7, ls=":",
                               alpha=0.6)

                    # Inferred MAP trajectory (if inference succeeded)
                    ok_row = df[
                        (df["scenario"] == label) &
                        (df["s_true"] == s_true) &
                        (df["h_true"] == h_true) &
                        (df["status"] == "ok")
                    ]
                    if not ok_row.empty:
                        s_inf = ok_row["s_inf"].iloc[0]
                        h_inf = ok_row["h_inf"].iloc[0]
                        x0_inf = x0_tot_from_vaf0(vaf0, h_inf)
                        vaf_inf = simulate_vaf(t_dense, s_inf, h_inf, x0_inf)
                        ax.plot(t_dense, vaf_inf, color="tomato", lw=1.5,
                                ls="--", label=f"MAP s={s_inf:.2f} h={h_inf:.2f}")

                    # Observed points with Wilson CI error bars
                    n_reads = DP[0]
                    k_reads = AO[0]
                    p_hat   = obs_vaf
                    z       = 1.96
                    denom_w = 1.0 + z**2 / n_reads
                    centre  = (p_hat + z**2 / (2 * n_reads)) / denom_w
                    half    = (z * np.sqrt(p_hat * (1 - p_hat) / n_reads
                               + z**2 / (4 * n_reads**2))) / denom_w
                    lo = np.maximum(centre - half, 1e-6)
                    hi = centre + half

                    ax.errorbar(
                        time_points, obs_vaf,
                        yerr=[obs_vaf - lo, hi - obs_vaf],
                        fmt="o", color="steelblue", ms=4, lw=1.0,
                        capsize=2, zorder=5, alpha=0.85,
                    )

                    ax.set_ylim(0, min(v_ceil * 1.15, 1.0))
                    ax.tick_params(labelsize=6)
                    ax.set_ylabel(f"s={s_true} h={h_true}\nVAF", fontsize=6)
                    if panel_idx == n_rows - 1:
                        ax.set_xlabel("Time (years)", fontsize=7)

                    if panel_idx == 0 and col == 0:
                        ax.legend(fontsize=5, loc="upper left")

                    panel_idx += 1

        fig.suptitle(
            f"VAF trajectories — axis: {axis}\n"
            "Blue solid = true  |  Red dashed = MAP  |  Grey dotted = VAF ceiling\n"
            "delta_t = inter-timepoint gap; total span = delta_t × (n_tps − 1)",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out = output_path.replace(".png", f"_{axis}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"VAF trajectory figure saved: {out}")
        plt.show()


# =============================================================================
# Identifiability diagram
# =============================================================================

IDENT_S_VALUES       = np.round(np.linspace(0.1, 1.0, 6), 2)
IDENT_H_VALUES       = np.round(np.linspace(0.0, 1.0, 6), 2)
IDENT_VAF0_VALUES    = np.round([0.01, 0.03, 0.05, 0.10, 0.15], 2)
IDENT_DELTA_T_VALUES = np.round([0.60, 1.5, 3.0, 5.0, 6.63], 2)

RHO_THRESHOLD = 0.5

OUTPUT_IDENT_FIG = "identifiability_diagram.png"
OUTPUT_IDENT_CSV = "identifiability_results.csv"


def compute_rho(s, h, vaf0, delta_t, N_w=N_W):
    """Analytically compute saturation ratio rho = v_T / v_max(h).

    Uses t_T = delta_t * (N_TPS_MID - 1) as the final timepoint,
    consistent with the fixed-midpoint convention.
    """
    x0_tot  = x0_tot_from_vaf0(vaf0, h, N_w)
    t_end   = delta_t * (N_TPS_MID - 1)
    tps     = np.array([0.0, t_end])
    vaf     = simulate_vaf(tps, s, h, x0_tot, N_w)
    v_T     = vaf[-1]
    v_max = (1.0 + h) / 2.0
    return float(v_T / max(v_max, 1e-8))


def run_identifiability_grid():
    """Compute rho for all (s, h, vaf0, delta_t) combinations."""
    records = []
    grid = list(itertools.product(
        IDENT_S_VALUES, IDENT_H_VALUES,
        IDENT_VAF0_VALUES, IDENT_DELTA_T_VALUES,
    ))
    print(f"\nIdentifiability grid: {len(grid)} cells (analytical, no inference)\n")

    for s, h, vaf0, delta_t in grid:
        rho = compute_rho(s, h, vaf0, delta_t)
        records.append(dict(
            s=s, h=h, vaf0=vaf0, delta_t=delta_t,
            rho=rho,
            identifiable=int(rho >= RHO_THRESHOLD),
        ))

    return pd.DataFrame(records)


def plot_identifiability(ident_df, bench_df, output_path):
    n_vaf0    = len(IDENT_VAF0_VALUES)
    n_delta_t = len(IDENT_DELTA_T_VALUES)

    # Part A — rho heatmaps
    fig_a, axes_a = plt.subplots(
        n_vaf0, n_delta_t,
        figsize=(3.5 * n_delta_t, 3.2 * n_vaf0),
        squeeze=False,
    )

    for row, vaf0 in enumerate(IDENT_VAF0_VALUES):
        for col, delta_t in enumerate(IDENT_DELTA_T_VALUES):
            ax  = axes_a[row, col]
            sub = ident_df[(ident_df["vaf0"] == vaf0) & (ident_df["delta_t"] == delta_t)]
            mat = sub.pivot(index="s", columns="h", values="rho")

            im = ax.imshow(
                mat.values, origin="lower", aspect="auto",
                vmin=0.0, vmax=1.0, cmap="viridis",
                extent=[
                    mat.columns.min() - 0.08, mat.columns.max() + 0.08,
                    mat.index.min()   - 0.08, mat.index.max()   + 0.08,
                ],
            )

            try:
                ax.contour(
                    mat.columns, mat.index, mat.values,
                    levels=[RHO_THRESHOLD],
                    colors="white", linewidths=1.5, linestyles="--",
                )
            except Exception:
                pass

            for s_v in mat.index:
                for h_v in mat.columns:
                    rho_val = mat.loc[s_v, h_v]
                    ax.text(h_v, s_v, f"{rho_val:.2f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if rho_val < 0.6 else "black")

            ax.set_xlabel("True  h", fontsize=7)
            ax.set_ylabel("True  s", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f"vaf0={vaf0:.2f}  Δt={delta_t:.1f}y",
                         fontsize=7.5, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8, label="ρ", pad=0.02)

    fig_a.suptitle(
        f"h identifiability: saturation ratio  ρ = v_T / v_max(h)\n"
        f"White dashed contour = identifiability threshold  ρ = {RHO_THRESHOLD}\n"
        f"(n_tps fixed at {N_TPS_MID}; v_T evaluated at t = delta_t × (n_tps − 1))",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path_a = output_path.replace(".png", "_rho_heatmaps.png")
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    print(f"Identifiability heatmaps saved: {path_a}")
    plt.show()

    # Part B — empirical h error vs rho
    ok = bench_df[bench_df["status"] == "ok"].copy()
    if ok.empty:
        print("No benchmark data for Part B — skipping.")
        return

    scen_lookup = {s["label"]: s for s in SCENARIOS}

    rho_vals   = []
    h_err_vals = []
    axis_vals  = []

    for _, row in ok.iterrows():
        scen = scen_lookup.get(row["scenario"])
        if scen is None:
            continue
        rho = compute_rho(
            s=row["s_true"], h=row["h_true"],
            vaf0=scen["vaf0"], delta_t=scen["delta_t"],
        )
        rho_vals.append(rho)
        h_err_vals.append(abs(row["h_err"]))
        axis_vals.append(scen["axis"])

    rho_arr   = np.array(rho_vals)
    h_err_arr = np.array(h_err_vals)
    axis_arr  = np.array(axis_vals)

    axis_colours = {"delta_t": "#2271B2", "n_tps": "#3DAA6A", "vaf0": "#D44D3A"}

    fig_b, ax_b = plt.subplots(figsize=(7, 5))

    for axis_name, colour in axis_colours.items():
        mask = axis_arr == axis_name
        ax_b.scatter(
            rho_arr[mask], h_err_arr[mask],
            color=colour, alpha=0.65, s=35, label=axis_name, zorder=3,
        )

    sort_idx = np.argsort(rho_arr)
    rho_s    = rho_arr[sort_idx]
    h_err_s  = h_err_arr[sort_idx]
    window   = max(5, len(rho_s) // 10)
    if len(rho_s) >= window:
        running_med = np.array([
            np.median(h_err_s[max(0, i - window // 2): i + window // 2 + 1])
            for i in range(len(rho_s))
        ])
        ax_b.plot(rho_s, running_med, color="black", lw=2,
                  label="Running median", zorder=4)

    ax_b.axvline(RHO_THRESHOLD, color="grey", lw=1.5, ls="--",
                 label=f"ρ threshold = {RHO_THRESHOLD}")
    ax_b.axvspan(0, RHO_THRESHOLD, alpha=0.06, color="red",
                 label="Unidentifiable region")

    ax_b.set_xlabel("Saturation ratio  ρ = v_T / v_max(h)", fontsize=10)
    ax_b.set_ylabel("|h error|  (|inferred − true|)", fontsize=10)
    ax_b.set_title(
        "Empirical h recovery error vs saturation ratio\n"
        "(each point = one OAT benchmark cell; colour = varied axis)",
        fontsize=10, fontweight="bold",
    )
    ax_b.legend(fontsize=8, framealpha=0.8)
    ax_b.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path_b = output_path.replace(".png", "_rho_vs_herr.png")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"rho vs h-error figure saved: {path_b}")
    plt.show()


# =============================================================================
# Fisher information
# =============================================================================

import jax
import jax.numpy as jnp

OUTPUT_FISHER_FIG = "fisher_information.png"
OUTPUT_FISHER_CSV = "fisher_results.csv"


def _vaf_at_t_jax(h, s, x0_tot, t, N_w=N_W):
    x_tot = x0_tot * jnp.exp(s * t)
    vaf   = x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))
    return jnp.clip(vaf, 1e-8, (1.0 + h) / 2.0 - 1e-8)


_dvaf_dh_single = jax.jit(jax.grad(_vaf_at_t_jax, argnums=0))


def compute_fisher_h(s, h, vaf0, delta_t, n_tps, depth, N_w=N_W):
    """Compute Fisher information I(h) and Cramér-Rao bound sqrt(1/I(h)).

    FIX: total span = delta_t * (n_tps - 1), consistent with generate_participant.
    """
    # FIX: was delta_t * n_tps — now correctly delta_t * (n_tps - 1)
    tps    = np.linspace(0.0, delta_t * (n_tps - 1), n_tps)
    x0_tot = float(x0_tot_from_vaf0(vaf0, h))  # was x0_het — wrong name, right value if fixed

    fisher = 0.0
    for t in tps:
        h_j   = jnp.array(h)
        s_j   = jnp.array(s)
        x0_j  = jnp.array(x0_tot)
        t_j   = jnp.array(t)

        v_t   = float(_vaf_at_t_jax(h_j, s_j, x0_j, t_j))
        dv_dh = float(_dvaf_dh_single(h_j, s_j, x0_j, t_j))

        denom = v_t * (1.0 - v_t)
        if denom > 1e-10:
            fisher += depth * (dv_dh ** 2) / denom

    cr_bound = float(1.0 / np.sqrt(max(fisher, 1e-12)))
    return fisher, cr_bound


def run_fisher_grid(n_tps=N_TPS_MID, depth=DEPTH):
    """Compute Fisher information over the full (s, h, vaf0, delta_t) grid."""
    grid = list(itertools.product(
        IDENT_S_VALUES, IDENT_H_VALUES,
        IDENT_VAF0_VALUES, IDENT_DELTA_T_VALUES,
    ))
    print(f"\nFisher information grid: {len(grid)} cells  "
          f"(n_tps={n_tps}, depth={depth})\n")

    records = []
    for i, (s, h, vaf0, delta_t) in enumerate(grid):
        fisher, cr_bound = compute_fisher_h(s, h, vaf0, delta_t, n_tps, depth)
        records.append(dict(
            s=s, h=h, vaf0=vaf0, delta_t=delta_t,
            fisher=fisher,
            cr_bound=cr_bound,
            identifiable=int(cr_bound < 0.15),
        ))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(grid)}")

    return pd.DataFrame(records)


def plot_fisher(fisher_df, ident_df, output_path):
    n_vaf0    = len(IDENT_VAF0_VALUES)
    n_delta_t = len(IDENT_DELTA_T_VALUES)

    CR_CONTOUR = 0.15

    # Figure A — CR bound heatmaps
    fig_a, axes_a = plt.subplots(
        n_vaf0, n_delta_t,
        figsize=(3.5 * n_delta_t, 3.2 * n_vaf0),
        squeeze=False,
    )

    for row, vaf0 in enumerate(IDENT_VAF0_VALUES):
        for col, delta_t in enumerate(IDENT_DELTA_T_VALUES):
            ax  = axes_a[row, col]
            sub = fisher_df[
                (fisher_df["vaf0"] == vaf0) &
                (fisher_df["delta_t"] == delta_t)
            ]
            mat = sub.pivot(index="s", columns="h", values="cr_bound")

            im = ax.imshow(
                mat.values, origin="lower", aspect="auto",
                vmin=0.0, vmax=0.5, cmap="RdYlGn_r",
                extent=[
                    mat.columns.min() - 0.08, mat.columns.max() + 0.08,
                    mat.index.min()   - 0.08, mat.index.max()   + 0.08,
                ],
            )

            try:
                ax.contour(
                    mat.columns, mat.index, mat.values,
                    levels=[CR_CONTOUR],
                    colors="black", linewidths=1.5, linestyles="--",
                )
            except Exception:
                pass

            for s_v in mat.index:
                for h_v in mat.columns:
                    val = mat.loc[s_v, h_v]
                    ax.text(h_v, s_v, f"{val:.2f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if val > 0.35 else "black")

            ax.set_xlabel("True  h", fontsize=7)
            ax.set_ylabel("True  s", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(f"vaf0={vaf0:.2f}  Δt={delta_t:.1f}y",
                         fontsize=7.5, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.8,
                         label="CR bound  √(1/I(h))", pad=0.02)

    fig_a.suptitle(
        "Cramér-Rao bound on h  [ = √(1/I(h)) ]\n"
        "Black dashed contour = CR bound 0.15  (identifiability threshold)\n"
        "Green = identifiable  |  Red = unidentifiable",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path_a = output_path.replace(".png", "_cr_heatmaps.png")
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    print(f"CR bound heatmaps saved: {path_a}")
    plt.show()

    # Figure B — Fisher vs rho scatter
    merged = fisher_df.merge(
        ident_df[["s", "h", "vaf0", "delta_t", "rho"]],
        on=["s", "h", "vaf0", "delta_t"], how="inner",
    )

    fig_b, axes_b = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes_b[0].scatter(
        merged["rho"], merged["cr_bound"],
        c=merged["h"], cmap="plasma", s=25, alpha=0.7,
    )
    axes_b[0].axhline(CR_CONTOUR, color="black", lw=1.5, ls="--",
                      label=f"CR threshold = {CR_CONTOUR}")
    axes_b[0].axvline(RHO_THRESHOLD, color="grey", lw=1.5, ls="--",
                      label=f"ρ threshold = {RHO_THRESHOLD}")
    axes_b[0].set_xlabel("Saturation ratio  ρ", fontsize=10)
    axes_b[0].set_ylabel("CR bound  √(1/I(h))", fontsize=10)
    axes_b[0].set_title("ρ vs CR bound\n(colour = true h)",
                         fontsize=10, fontweight="bold")
    axes_b[0].legend(fontsize=8)
    axes_b[0].spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc, ax=axes_b[0], label="True h")

    sc2 = axes_b[1].scatter(
        merged["rho"],
        np.log10(np.maximum(merged["fisher"], 1e-12)),
        c=merged["vaf0"], cmap="viridis", s=25, alpha=0.7,
    )
    axes_b[1].set_xlabel("Saturation ratio  ρ", fontsize=10)
    axes_b[1].set_ylabel("log₁₀  I(h)", fontsize=10)
    axes_b[1].set_title("ρ vs Fisher information\n(colour = vaf0)",
                         fontsize=10, fontweight="bold")
    axes_b[1].spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc2, ax=axes_b[1], label="vaf0")

    fig_b.suptitle(
        "Relationship between saturation ratio ρ and Fisher information I(h)\n"
        "If tight: ρ is a reliable proxy for identifiability. "
        "If loose: ρ is misleading.",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    path_b = output_path.replace(".png", "_fisher_vs_rho.png")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"Fisher vs rho figure saved: {path_b}")
    plt.show()

    # Figure C — Agreement / disagreement between rho and CR bound
    fig_c, ax_c = plt.subplots(figsize=(7, 6))

    colours = []
    for _, r in merged.iterrows():
        rho_ok = r["rho"]      >= RHO_THRESHOLD
        cr_ok  = r["cr_bound"] <= CR_CONTOUR
        if rho_ok and cr_ok:
            colours.append("#3DAA6A")
        elif not rho_ok and not cr_ok:
            colours.append("#D44D3A")
        elif not rho_ok and cr_ok:
            colours.append("#F0A500")
        else:
            colours.append("#8E5FB9")

    ax_c.scatter(merged["rho"], merged["cr_bound"],
                 c=colours, s=30, alpha=0.75, zorder=3)
    ax_c.axhline(CR_CONTOUR,    color="black", lw=1.5, ls="--")
    ax_c.axvline(RHO_THRESHOLD, color="grey",  lw=1.5, ls="--")
    ax_c.set_xlabel("Saturation ratio  ρ", fontsize=10)
    ax_c.set_ylabel("CR bound  √(1/I(h))", fontsize=10)
    ax_c.set_title(
        "Agreement between ρ and CR bound\n"
        "Green = both identifiable  |  Red = both unidentifiable\n"
        "Amber = ρ too conservative  |  Purple = ρ too optimistic",
        fontsize=9, fontweight="bold",
    )
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.text(RHO_THRESHOLD * 0.5, CR_CONTOUR * 1.1,
              "ρ: ✗  CR: ✓\n(ρ too conservative)",
              fontsize=7, ha="center", color="#F0A500")
    ax_c.text(RHO_THRESHOLD * 1.5, CR_CONTOUR * 1.1,
              "ρ: ✓  CR: ✗\n(ρ too optimistic)",
              fontsize=7, ha="center", color="#8E5FB9")

    plt.tight_layout()
    path_c = output_path.replace(".png", "_agreement.png")
    fig_c.savefig(path_c, dpi=150, bbox_inches="tight")
    print(f"Agreement figure saved: {path_c}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # --- OAT sensitivity benchmark ---
    df = run_benchmark()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print_summary(df)
    plot_sensitivity(df, OUTPUT_FIG)
    plot_vaf_trajectories(df, OUTPUT_VAF_FIG)

    # --- Full-factorial identifiability diagram (rho) ---
    ident_df = run_identifiability_grid()
    ident_df.to_csv(OUTPUT_IDENT_CSV, index=False)
    print(f"\nIdentifiability results saved to {OUTPUT_IDENT_CSV}")
    plot_identifiability(ident_df, df, OUTPUT_IDENT_FIG)

    # --- Fisher information identifiability ---
    fisher_df = run_fisher_grid()
    fisher_df.to_csv(OUTPUT_FISHER_CSV, index=False)
    print(f"\nFisher results saved to {OUTPUT_FISHER_CSV}")
    plot_fisher(fisher_df, ident_df, OUTPUT_FISHER_FIG)