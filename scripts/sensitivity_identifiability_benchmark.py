"""
sensitivity_identifiability_benchmark.py
----------------------------------------

Benchmark h-identifiability in the deterministic inference pipeline.

This script simulates one-mutation participants under controlled settings and
tests how well the deterministic posterior inference recovers:

    - selection coefficient s
    - homozygous fraction / zygosity parameter h

The benchmark varies:

    1. Inter-timepoint gap delta_t
    2. Number of timepoints
    3. Initial VAF

For each scenario, s_true and h_true are varied on a small grid.

In addition to MAP error and CI coverage, this version computes explicit
identifiability metrics for h:

    - h_ci_width:
        width of the 90% credible interval

    - h_mass_015:
        posterior mass within true h ± 0.15

    - h_overlap_010:
        overlap coefficient between inferred h posterior and a reference
        posterior centred at true h

    - h_entropy:
        normalised posterior entropy, where 1 means almost flat/unidentified

    - rho:
        saturation ratio v_T / v_max(h)

    - Fisher information and Cramér-Rao bound for h

Outputs
-------
    sensitivity_results.csv
    identifiability_results.csv
    fisher_results.csv

    sensitivity_heatmaps.png
    vaf_trajectories_delta_t.png
    vaf_trajectories_n_tps.png
    vaf_trajectories_vaf0.png
    h_identifiability_summary.png
    posterior_overlap_examples.png
    identifiability_rho_heatmaps.png
    identifiability_rho_vs_herr.png
    fisher_cr_heatmaps.png
    fisher_vs_rho.png
    fisher_rho_agreement.png
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

import jax
import jax.numpy as jnp


# =============================================================================
# Import inference pipeline
# =============================================================================

sys.path.append("..")
sys.path.append("../src")

from KI_clonal_inference_6 import infer_sh_jointly_from_dynamics


# =============================================================================
# Configuration
# =============================================================================

N_W = 1e5
DEPTH = 500
SEED = 42

S_RESOLUTION = 60
H_RESOLUTION = 80

# True-value grid.
S_TRUE_VALUES = np.round([0.1, 0.3, 0.6, 1.0], 2)
H_TRUE_VALUES = np.round([0.0, 0.3, 0.6, 1.0], 2)

# Scenario midpoints.
DELTA_T_MID = 3.0
N_TPS_MID = 4
VAF0_MID = 0.05

# Scenario definitions.
SCENARIOS = [
    # Axis 1: inter-timepoint gap.
    dict(
        label="delta_t=0.60y (worst)",
        delta_t=0.60,
        n_tps=N_TPS_MID,
        vaf0=VAF0_MID,
        axis="delta_t",
    ),
    dict(
        label="delta_t=3.0y (mid)",
        delta_t=3.0,
        n_tps=N_TPS_MID,
        vaf0=VAF0_MID,
        axis="delta_t",
    ),
    dict(
        label="delta_t=6.63y (best)",
        delta_t=6.63,
        n_tps=N_TPS_MID,
        vaf0=VAF0_MID,
        axis="delta_t",
    ),

    # Axis 2: number of timepoints.
    dict(
        label="n_tps=2 (worst)",
        delta_t=DELTA_T_MID,
        n_tps=2,
        vaf0=VAF0_MID,
        axis="n_tps",
    ),
    dict(
        label="n_tps=4 (mid)",
        delta_t=DELTA_T_MID,
        n_tps=4,
        vaf0=VAF0_MID,
        axis="n_tps",
    ),
    dict(
        label="n_tps=9 (best)",
        delta_t=DELTA_T_MID,
        n_tps=9,
        vaf0=VAF0_MID,
        axis="n_tps",
    ),

    # Axis 3: initial VAF.
    dict(
        label="vaf0=0.01 (worst)",
        delta_t=DELTA_T_MID,
        n_tps=N_TPS_MID,
        vaf0=0.01,
        axis="vaf0",
    ),
    dict(
        label="vaf0=0.05 (mid)",
        delta_t=DELTA_T_MID,
        n_tps=N_TPS_MID,
        vaf0=0.05,
        axis="vaf0",
    ),
    dict(
        label="vaf0=0.15 (best)",
        delta_t=DELTA_T_MID,
        n_tps=N_TPS_MID,
        vaf0=0.15,
        axis="vaf0",
    ),
]

# Identifiability-grid settings.
IDENT_S_VALUES = np.round(np.linspace(0.1, 1.0, 6), 2)
IDENT_H_VALUES = np.round(np.linspace(0.0, 1.0, 6), 2)
IDENT_VAF0_VALUES = np.round([0.01, 0.03, 0.05, 0.10, 0.15], 2)
IDENT_DELTA_T_VALUES = np.round([0.60, 1.5, 3.0, 5.0, 6.63], 2)

RHO_THRESHOLD = 0.5
CR_BOUND_THRESHOLD = 0.15

# Posterior-overlap settings.
REFERENCE_H_SIGMA = 0.10
TRUE_H_WINDOW = 0.15

# Output files.
OUTPUT_CSV = "sensitivity_results.csv"
OUTPUT_IDENT_CSV = "identifiability_results.csv"
OUTPUT_FISHER_CSV = "fisher_results.csv"

OUTPUT_HEATMAPS = "sensitivity_heatmaps.png"
OUTPUT_VAF_TRAJECTORIES_BASE = "vaf_trajectories.png"
OUTPUT_H_IDENT_SUMMARY = "h_identifiability_summary.png"
OUTPUT_POSTERIOR_EXAMPLES = "posterior_overlap_examples.png"
OUTPUT_IDENT_FIG = "identifiability_diagram.png"
OUTPUT_FISHER_FIG = "fisher_information.png"


# =============================================================================
# Core VAF model
# =============================================================================

def x0_tot_from_vaf0(vaf0, h, N_w=N_W):
    """
    Convert initial VAF to initial total mutant-cell count.

    Model:
        VAF = x_tot * (1+h) / (2 * (N_w + x_tot))

    Inversion:
        x_tot0 = 2 * vaf0 * N_w / ((1+h) - 2*vaf0)
    """
    denom = (1.0 + h) - 2.0 * vaf0
    denom = max(float(denom), 1e-8)
    return 2.0 * vaf0 * N_w / denom


def simulate_vaf(time_points, s, h, x0_tot, N_w=N_W):
    """
    Simulate deterministic VAF trajectory.

    h = fraction of mutant population that is homozygous.
    """
    t = np.asarray(time_points, dtype=float)
    x_tot = x0_tot * np.exp(s * (t - t[0]))

    vaf = x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))
    v_max = (1.0 + h) / 2.0

    return np.clip(vaf, 1e-8, v_max - 1e-8)


def make_timepoints(total_span, n_tps, rng):
    """
    Generate irregular timepoints.

    Always includes 0.

    For n_tps=2, use [0, total_span] so the full follow-up is represented.
    For n_tps>2, sample n_tps-1 additional timepoints uniformly over the span.
    """
    if n_tps <= 1:
        return np.array([0.0])

    if n_tps == 2:
        return np.array([0.0, total_span], dtype=float)

    interior = rng.uniform(0.0, total_span, size=n_tps - 1)
    return np.sort(np.concatenate([[0.0], interior]))


def generate_participant(s_true, h_true, delta_t, n_tps, vaf0, depth, seed):
    """
    Generate one synthetic one-mutation participant.

    Returns
    -------
    AO : shape (1, n_tps)
    DP : shape (1, n_tps)
    time_points : shape (n_tps,)
    vaf_true : shape (n_tps,)
    """
    rng = np.random.default_rng(seed)

    total_span = delta_t * (n_tps - 1)
    time_points = make_timepoints(total_span, n_tps, rng)

    x0_tot = x0_tot_from_vaf0(vaf0, h_true)
    vaf_true = simulate_vaf(time_points, s_true, h_true, x0_tot)

    DP = np.full((1, len(time_points)), depth, dtype=float)
    AO = rng.binomial(int(depth), vaf_true).reshape(1, -1).astype(float)

    return AO, DP, time_points, vaf_true


# =============================================================================
# Posterior identifiability helpers
# =============================================================================

def normalise_density_on_grid(y, x):
    """
    Normalise y as a density over grid x so that integral y dx = 1.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0.0)

    area = np.trapz(y, x=x)

    if area <= 0 or not np.isfinite(area):
        flat = np.ones_like(y, dtype=float)
        return flat / np.trapz(flat, x=x)

    return y / area


def truncated_gaussian_reference(h_range, h_true, sigma=REFERENCE_H_SIGMA):
    """
    Reference density centred at true h.

    This is not another inference result; it is an interpretable target
    distribution used for posterior-overlap scoring.
    """
    h_range = np.asarray(h_range, dtype=float)
    ref = np.exp(-0.5 * ((h_range - h_true) / sigma) ** 2)
    return normalise_density_on_grid(ref, h_range)


def posterior_overlap(p, q, x):
    """
    Overlap coefficient between two densities.

    OVL = integral min(p, q) dx

    Returns value in [0, 1].
    """
    p = normalise_density_on_grid(p, x)
    q = normalise_density_on_grid(q, x)

    ovl = np.trapz(np.minimum(p, q), x=x)
    return float(np.clip(ovl, 0.0, 1.0))


def posterior_mass_within(p, x, centre, width=TRUE_H_WINDOW):
    """
    Posterior mass inside centre ± width.
    """
    p = normalise_density_on_grid(p, x)
    x = np.asarray(x, dtype=float)

    mask = (x >= centre - width) & (x <= centre + width)

    if not np.any(mask):
        return 0.0

    return float(np.trapz(p[mask], x=x[mask]))


def posterior_entropy_discrete(p):
    """
    Normalised entropy of a discrete posterior vector.

    Returns near 1 for flat/unidentified posterior.
    Returns near 0 for highly concentrated posterior.
    """
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)

    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        return np.nan

    p = p / total
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)))
    entropy_norm = entropy / np.log(len(p))

    return float(entropy_norm)


def compute_posterior_diagnostics(result, h_true):
    """
    Compute h-identifiability diagnostics from one inference result.
    """
    h_range = np.asarray(result["h_range"], dtype=float)
    h_post = np.asarray(result["h_posterior"], dtype=float)
    h_ci = result["h_ci"]

    h_post_density = normalise_density_on_grid(h_post, h_range)
    h_ref_density = truncated_gaussian_reference(
        h_range,
        h_true,
        sigma=REFERENCE_H_SIGMA,
    )

    h_overlap = posterior_overlap(
        h_post_density,
        h_ref_density,
        h_range,
    )

    h_mass = posterior_mass_within(
        h_post_density,
        h_range,
        h_true,
        width=TRUE_H_WINDOW,
    )

    h_entropy = posterior_entropy_discrete(h_post)
    h_ci_width = float(h_ci[1] - h_ci[0])

    return dict(
        h_ci_width=h_ci_width,
        h_overlap_010=h_overlap,
        h_mass_015=h_mass,
        h_entropy=h_entropy,
    )


# =============================================================================
# Saturation ratio rho
# =============================================================================

def compute_rho(s, h, vaf0, delta_t, n_tps=N_TPS_MID, N_w=N_W):
    """
    Compute saturation ratio:

        rho = v_T / v_max(h)

    where:
        v_max(h) = (1+h)/2

    Larger rho means trajectory approaches the h-specific VAF ceiling,
    making h easier to identify from VAF data.
    """
    total_span = delta_t * (n_tps - 1)
    x0_tot = x0_tot_from_vaf0(vaf0, h, N_w=N_w)

    tps = np.array([0.0, total_span], dtype=float)
    vaf = simulate_vaf(tps, s, h, x0_tot, N_w=N_w)

    v_T = vaf[-1]
    v_max = (1.0 + h) / 2.0

    return float(v_T / max(v_max, 1e-8))


# =============================================================================
# Fisher information for h
# =============================================================================

def _vaf_at_t_jax(h, s, x0_tot, t, N_w=N_W):
    x_tot = x0_tot * jnp.exp(s * t)
    vaf = x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))
    return jnp.clip(vaf, 1e-8, (1.0 + h) / 2.0 - 1e-8)


_dvaf_dh_single = jax.jit(jax.grad(_vaf_at_t_jax, argnums=0))


def compute_fisher_h(s, h, vaf0, delta_t, n_tps, depth, N_w=N_W):
    """
    Compute Fisher information for h under binomial sampling.

        I(h) = sum_t depth * (dv_t/dh)^2 / (v_t * (1-v_t))

    Important:
        This treats x0_tot as fixed at its true h-derived value.
        If x0 is treated as a nuisance parameter, h identifiability is weaker.
    """
    total_span = delta_t * (n_tps - 1)
    tps = np.linspace(0.0, total_span, n_tps)

    x0_tot = float(x0_tot_from_vaf0(vaf0, h, N_w=N_w))

    fisher = 0.0

    for t in tps:
        h_j = jnp.array(h)
        s_j = jnp.array(s)
        x0_j = jnp.array(x0_tot)
        t_j = jnp.array(t)

        v_t = float(_vaf_at_t_jax(h_j, s_j, x0_j, t_j, N_w))
        dv_dh = float(_dvaf_dh_single(h_j, s_j, x0_j, t_j, N_w))

        denom = v_t * (1.0 - v_t)

        if denom > 1e-10:
            fisher += depth * (dv_dh ** 2) / denom

    cr_bound = float(1.0 / np.sqrt(max(fisher, 1e-12)))

    return fisher, cr_bound


# =============================================================================
# Run one benchmark cell
# =============================================================================

def _empty_record(scenario_label, axis, s_true, h_true, msg):
    return dict(
        scenario=scenario_label,
        axis=axis,
        s_true=s_true,
        h_true=h_true,
        s_inf=np.nan,
        h_inf=np.nan,
        s_err=np.nan,
        h_err=np.nan,
        s_lo=np.nan,
        s_hi=np.nan,
        h_lo=np.nan,
        h_hi=np.nan,
        s_in_ci=np.nan,
        h_in_ci=np.nan,
        h_ci_width=np.nan,
        h_overlap_010=np.nan,
        h_mass_015=np.nan,
        h_entropy=np.nan,
        rho=np.nan,
        fisher_h=np.nan,
        cr_bound_h=np.nan,
        elapsed_s=np.nan,
        status=f"error: {msg}",
    )


def run_cell(scenario, s_true, h_true, cell_seed):
    """
    Simulate and infer one cell of the benchmark.
    """
    label = scenario["label"]
    axis = scenario["axis"]
    delta_t = scenario["delta_t"]
    n_tps = scenario["n_tps"]
    vaf0 = scenario["vaf0"]

    try:
        AO, DP, time_points, vaf_true = generate_participant(
            s_true=s_true,
            h_true=h_true,
            delta_t=delta_t,
            n_tps=n_tps,
            vaf0=vaf0,
            depth=DEPTH,
            seed=cell_seed,
        )

        # infer_sh_jointly_from_dynamics expects AO/DP as (n_tps, n_mutations).
        AO_T = AO.T
        DP_T = DP.T
        cs = [[0]]

        t0 = time.perf_counter()

        results = infer_sh_jointly_from_dynamics(
            cs,
            AO_T,
            DP_T,
            time_points,
            s_resolution=S_RESOLUTION,
            h_resolution=H_RESOLUTION,
        )

        elapsed = time.perf_counter() - t0

        r = results[0]

        s_map = float(r["s_map"])
        h_map = float(r["h_map"])
        s_ci = r["s_ci"]
        h_ci = r["h_ci"]

        diagnostics = compute_posterior_diagnostics(r, h_true)

        rho = compute_rho(
            s=s_true,
            h=h_true,
            vaf0=vaf0,
            delta_t=delta_t,
            n_tps=n_tps,
        )

        fisher_h, cr_bound_h = compute_fisher_h(
            s=s_true,
            h=h_true,
            vaf0=vaf0,
            delta_t=delta_t,
            n_tps=n_tps,
            depth=DEPTH,
        )

        return dict(
            scenario=label,
            axis=axis,
            s_true=s_true,
            h_true=h_true,
            s_inf=s_map,
            h_inf=h_map,
            s_err=s_map - s_true,
            h_err=h_map - h_true,
            s_lo=s_ci[0],
            s_hi=s_ci[1],
            h_lo=h_ci[0],
            h_hi=h_ci[1],
            s_in_ci=int(s_ci[0] <= s_true <= s_ci[1]),
            h_in_ci=int(h_ci[0] <= h_true <= h_ci[1]),
            h_ci_width=diagnostics["h_ci_width"],
            h_overlap_010=diagnostics["h_overlap_010"],
            h_mass_015=diagnostics["h_mass_015"],
            h_entropy=diagnostics["h_entropy"],
            rho=rho,
            fisher_h=fisher_h,
            cr_bound_h=cr_bound_h,
            elapsed_s=elapsed,
            status="ok",
        )

    except Exception as exc:
        traceback.print_exc()
        return _empty_record(label, axis, s_true, h_true, str(exc))


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark():
    records = []
    true_grid = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))
    n_total = len(SCENARIOS) * len(true_grid)

    print()
    print(f"{len(SCENARIOS)} scenarios × {len(true_grid)} cells = {n_total} runs")
    print()

    run_idx = 0

    for scen in SCENARIOS:
        print()
        print("=" * 70)
        print(f"  {scen['label']}")
        total_span = scen["delta_t"] * (scen["n_tps"] - 1)
        print(
            f"  delta_t={scen['delta_t']}y  "
            f"n_tps={scen['n_tps']}  "
            f"total_span={total_span:.2f}y  "
            f"vaf0={scen['vaf0']}"
        )
        print("=" * 70)

        for s_true, h_true in true_grid:
            run_idx += 1
            cell_seed = SEED + run_idx

            rec = run_cell(scen, s_true, h_true, cell_seed)
            records.append(rec)

            if rec["status"] == "ok":
                print(
                    f"  [{run_idx:3d}/{n_total}] ✓"
                    f"  s={s_true:.2f} h={h_true:.2f}"
                    f"  → s_inf={rec['s_inf']:.3f} ({rec['s_err']:+.3f})"
                    f"  h_inf={rec['h_inf']:.3f} ({rec['h_err']:+.3f})"
                    f"  OVL={rec['h_overlap_010']:.2f}"
                    f"  mass±0.15={rec['h_mass_015']:.2f}"
                    f"  H={rec['h_entropy']:.2f}"
                    f"  ρ={rec['rho']:.2f}"
                    f"  {rec['elapsed_s']:.1f}s"
                )
            else:
                print(
                    f"  [{run_idx:3d}/{n_total}] ✗"
                    f"  s={s_true:.2f} h={h_true:.2f}"
                    f"  {rec['status']}"
                )

    return pd.DataFrame(records)


# =============================================================================
# Summary
# =============================================================================

def print_summary(df):
    print()
    print("=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)

    for axis in ["delta_t", "n_tps", "vaf0"]:
        print()
        print(f"Axis: {axis}")
        print(
            f"{'Scenario':<30}"
            f" {'s MAE':>7}"
            f" {'h MAE':>7}"
            f" {'h OVL':>7}"
            f" {'h mass':>7}"
            f" {'h CIw':>7}"
            f" {'h H':>7}"
            f" {'rho':>7}"
            f" {'CR':>7}"
            f" {'n':>4}"
        )

        for scen in [s for s in SCENARIOS if s["axis"] == axis]:
            ok = df[
                (df["scenario"] == scen["label"]) &
                (df["status"] == "ok")
            ]

            if ok.empty:
                print(f"{scen['label']:<30}  no successful runs")
                continue

            print(
                f"{scen['label']:<30}"
                f" {ok['s_err'].abs().mean():>7.3f}"
                f" {ok['h_err'].abs().mean():>7.3f}"
                f" {ok['h_overlap_010'].mean():>7.3f}"
                f" {ok['h_mass_015'].mean():>7.3f}"
                f" {ok['h_ci_width'].mean():>7.3f}"
                f" {ok['h_entropy'].mean():>7.3f}"
                f" {ok['rho'].mean():>7.3f}"
                f" {ok['cr_bound_h'].mean():>7.3f}"
                f" {len(ok):>4}"
            )

    print("=" * 80)


# =============================================================================
# Heatmap plotting helpers
# =============================================================================

def pivot(df, value_col):
    return df.pivot(index="s_true", columns="h_true", values=value_col)


def draw_heatmap(
    ax,
    mat,
    title,
    cmap,
    vmin=None,
    vmax=None,
    centered=False,
    fmt="{:.2f}",
    annotate=True,
):
    kwargs = dict(cmap=cmap, aspect="auto")

    if centered:
        v = max(abs(vmin), abs(vmax))
        kwargs["norm"] = TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    im = ax.imshow(
        mat.values,
        origin="lower",
        extent=[
            mat.columns.min() - 0.075,
            mat.columns.max() + 0.075,
            mat.index.min() - 0.075,
            mat.index.max() + 0.075,
        ],
        **kwargs,
    )

    if annotate:
        for s in mat.index:
            for h in mat.columns:
                val = mat.loc[s, h]
                if np.isnan(val):
                    continue

                color = "white" if abs(val) > 0.45 else "black"

                ax.text(
                    h,
                    s,
                    fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=color,
                )

    ax.set_xlabel("True h", fontsize=7)
    ax.set_ylabel("True s", fontsize=7)
    ax.set_title(title, fontsize=7.5, fontweight="bold")
    ax.tick_params(labelsize=6)

    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)


# =============================================================================
# Plot sensitivity heatmaps
# =============================================================================

def plot_sensitivity(df, output_path=OUTPUT_HEATMAPS):
    """
    One row per scenario.

    Columns:
        1. s error
        2. h error
        3. h posterior overlap
        4. h posterior mass near true h
        5. h posterior entropy
        6. h CI width
    """
    n_scen = len(SCENARIOS)

    fig = plt.figure(figsize=(27, 4.4 * n_scen))
    gs = gridspec.GridSpec(
        n_scen,
        6,
        figure=fig,
        hspace=0.58,
        wspace=0.35,
    )

    axis_colours = {
        "delta_t": "#2271B2",
        "n_tps": "#3DAA6A",
        "vaf0": "#D44D3A",
    }

    for row, scen in enumerate(SCENARIOS):
        label = scen["label"]
        axis = scen["axis"]

        sub = df[
            (df["scenario"] == label) &
            (df["status"] == "ok")
        ].copy()

        axes_list = [fig.add_subplot(gs[row, c]) for c in range(6)]

        if sub.empty:
            for ax in axes_list:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            continue

        draw_heatmap(
            axes_list[0],
            pivot(sub, "s_err"),
            f"{label}\ns error",
            "RdBu_r",
            -0.5,
            0.5,
            centered=True,
        )

        draw_heatmap(
            axes_list[1],
            pivot(sub, "h_err"),
            "h error",
            "RdBu_r",
            -0.5,
            0.5,
            centered=True,
        )

        draw_heatmap(
            axes_list[2],
            pivot(sub, "h_overlap_010"),
            "h posterior overlap\nwith true-centred ref.",
            "viridis",
            0,
            1,
            fmt="{:.2f}",
        )

        draw_heatmap(
            axes_list[3],
            pivot(sub, "h_mass_015"),
            "posterior mass\nwithin true h ± 0.15",
            "viridis",
            0,
            1,
            fmt="{:.2f}",
        )

        draw_heatmap(
            axes_list[4],
            pivot(sub, "h_entropy"),
            "h posterior entropy\n1 = flat",
            "magma",
            0,
            1,
            fmt="{:.2f}",
        )

        draw_heatmap(
            axes_list[5],
            pivot(sub, "h_ci_width"),
            "h 90% CI width",
            "RdYlGn_r",
            0,
            1,
            fmt="{:.2f}",
        )

        for ax in axes_list:
            for spine in ax.spines.values():
                spine.set_edgecolor(axis_colours[axis])
                spine.set_linewidth(1.8)

    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3, label=k)
        for k, c in axis_colours.items()
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=8,
        title="Varied axis",
        title_fontsize=8,
        framealpha=0.8,
    )

    fig.suptitle(
        "Sensitivity benchmark for h identifiability\n"
        f"Reference overlap uses Gaussian target centred at true h, sigma={REFERENCE_H_SIGMA}; "
        f"mass window = ±{TRUE_H_WINDOW}",
        fontsize=13,
        fontweight="bold",
        y=1.005,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# =============================================================================
# Summary plot for h identifiability
# =============================================================================

def plot_h_identifiability_summary(df, output_path=OUTPUT_H_IDENT_SUMMARY):
    ok = df[df["status"] == "ok"].copy()

    metrics = [
        ("h_overlap_010", "Mean h posterior overlap", "higher is better"),
        ("h_mass_015", "Mean posterior mass near true h", "higher is better"),
        ("h_entropy", "Mean h posterior entropy", "lower is better"),
        ("h_ci_width", "Mean h 90% CI width", "lower is better"),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        3,
        figsize=(14, 12),
        sharey=False,
    )

    axis_colours = {
        "delta_t": "#2271B2",
        "n_tps": "#3DAA6A",
        "vaf0": "#D44D3A",
    }

    for row, (metric, ylabel, note) in enumerate(metrics):
        for col, axis_name in enumerate(["delta_t", "n_tps", "vaf0"]):
            ax = axes[row, col]

            labels = []
            means = []
            sems = []

            for scen in [s for s in SCENARIOS if s["axis"] == axis_name]:
                vals = ok[ok["scenario"] == scen["label"]][metric].dropna()

                labels.append(
                    scen["label"]
                    .replace("delta_t=", "")
                    .replace("n_tps=", "")
                    .replace("vaf0=", "")
                    .replace(" (worst)", "")
                    .replace(" (mid)", "")
                    .replace(" (best)", "")
                )

                means.append(vals.mean())
                sems.append(vals.sem() if len(vals) > 1 else 0.0)

            ax.bar(
                range(len(labels)),
                means,
                yerr=sems,
                capsize=4,
                color=axis_colours[axis_name],
                alpha=0.85,
            )

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(axis_name, fontsize=9, fontweight="bold")
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=9)
            ax.text(
                0.02,
                0.95,
                note,
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )
            ax.spines[["top", "right"]].set_visible(False)

            if metric in ["h_overlap_010", "h_mass_015", "h_entropy", "h_ci_width"]:
                ax.set_ylim(0, 1)

    fig.suptitle(
        "Summary of h identifiability across benchmark axes",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# =============================================================================
# Posterior overlap example plots
# =============================================================================

def rerun_single_result(scen, s_true, h_true):
    """
    Recreate a single inference result so we can plot its posterior.
    """
    true_grid_full = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))

    scen_pos = SCENARIOS.index(scen)
    grid_pos = true_grid_full.index((s_true, h_true))
    run_idx = scen_pos * len(true_grid_full) + grid_pos + 1
    cell_seed = SEED + run_idx

    AO, DP, time_points, vaf_true = generate_participant(
        s_true=s_true,
        h_true=h_true,
        delta_t=scen["delta_t"],
        n_tps=scen["n_tps"],
        vaf0=scen["vaf0"],
        depth=DEPTH,
        seed=cell_seed,
    )

    results = infer_sh_jointly_from_dynamics(
        [[0]],
        AO.T,
        DP.T,
        time_points,
        s_resolution=S_RESOLUTION,
        h_resolution=H_RESOLUTION,
    )

    return results[0], AO, DP, time_points, vaf_true


def plot_posterior_overlap_examples(
    df,
    output_path=OUTPUT_POSTERIOR_EXAMPLES,
):
    """
    Plot inferred h posterior and true-centred reference posterior.

    This directly visualises the overlap idea:
        overlap = area under min(inferred posterior, reference posterior)
    """
    examples = [
        ("Poor / short follow-up", SCENARIOS[0], 0.3, 0.6),
        ("Mid", SCENARIOS[1], 0.3, 0.6),
        ("Good / long follow-up", SCENARIOS[2], 0.3, 0.6),
        ("Low starting VAF", SCENARIOS[6], 0.3, 0.6),
        ("Mid starting VAF", SCENARIOS[7], 0.3, 0.6),
        ("High starting VAF", SCENARIOS[8], 0.3, 0.6),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7),
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    for ax, (title, scen, s_true, h_true) in zip(axes, examples):
        result, AO, DP, time_points, vaf_true = rerun_single_result(
            scen,
            s_true,
            h_true,
        )

        h_range = np.asarray(result["h_range"], dtype=float)
        h_post = normalise_density_on_grid(result["h_posterior"], h_range)
        h_ref = truncated_gaussian_reference(
            h_range,
            h_true,
            sigma=REFERENCE_H_SIGMA,
        )

        overlap_density = np.minimum(h_post, h_ref)
        ovl = posterior_overlap(h_post, h_ref, h_range)

        ax.plot(h_range, h_post, color="#2271B2", lw=2, label="Inferred posterior")
        ax.plot(h_range, h_ref, color="#D44D3A", lw=2, ls="--", label="True-centred reference")
        ax.fill_between(
            h_range,
            0,
            overlap_density,
            color="#3DAA6A",
            alpha=0.35,
            label="Overlap area",
        )

        ax.axvline(h_true, color="black", lw=1.2, ls=":", label="True h")
        ax.axvline(result["h_map"], color="#2271B2", lw=1.2, ls="-.", label="MAP h")

        ax.set_title(
            f"{title}\n{scen['label']} | OVL={ovl:.2f}",
            fontsize=9,
            fontweight="bold",
        )

        ax.set_xlabel("h")
        ax.set_ylabel("Density")
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
    )

    fig.suptitle(
        "Posterior-overlap interpretation for h identifiability\n"
        "Green area = integral of min(inferred posterior, true-centred reference)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# =============================================================================
# VAF trajectory plots
# =============================================================================

def plot_vaf_trajectories(df, output_base=OUTPUT_VAF_TRAJECTORIES_BASE):
    """
    Plot true and inferred VAF trajectories for representative cells.
    """
    s_subset = [0.1, 0.3, 1.0]
    h_subset = [0.0, 0.3, 1.0]

    true_grid_full = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))

    axis_colours = {
        "delta_t": "#2271B2",
        "n_tps": "#3DAA6A",
        "vaf0": "#D44D3A",
    }

    for axis_name in ["delta_t", "n_tps", "vaf0"]:
        axis_scens = [s for s in SCENARIOS if s["axis"] == axis_name]
        n_scens = len(axis_scens)
        n_rows = len(s_subset) * len(h_subset)

        fig, axes = plt.subplots(
            n_rows,
            n_scens,
            figsize=(4.5 * n_scens, 3.0 * n_rows),
            squeeze=False,
        )

        for col, scen in enumerate(axis_scens):
            label = scen["label"]
            delta_t = scen["delta_t"]
            n_tps = scen["n_tps"]
            vaf0 = scen["vaf0"]

            total_span = delta_t * (n_tps - 1)

            axes[0, col].set_title(
                f"{label}\nspan={total_span:.1f}y | n={n_tps} | vaf0={vaf0}",
                fontsize=8,
                fontweight="bold",
                color=axis_colours[axis_name],
            )

            panel_idx = 0

            for s_true in s_subset:
                for h_true in h_subset:
                    ax = axes[panel_idx, col]

                    grid_pos = true_grid_full.index((s_true, h_true))
                    scen_pos = SCENARIOS.index(scen)
                    run_idx = scen_pos * len(true_grid_full) + grid_pos + 1
                    cell_seed = SEED + run_idx

                    AO, DP, time_points, vaf_true = generate_participant(
                        s_true=s_true,
                        h_true=h_true,
                        delta_t=delta_t,
                        n_tps=n_tps,
                        vaf0=vaf0,
                        depth=DEPTH,
                        seed=cell_seed,
                    )

                    obs_vaf = (AO / np.maximum(DP, 1.0))[0]

                    t_dense = np.linspace(time_points[0], time_points[-1], 300)

                    x0_true = x0_tot_from_vaf0(vaf0, h_true)
                    vaf_dense_true = simulate_vaf(
                        t_dense,
                        s_true,
                        h_true,
                        x0_true,
                    )

                    ax.plot(
                        t_dense,
                        vaf_dense_true,
                        color="steelblue",
                        lw=1.8,
                        label="True",
                    )

                    ok_row = df[
                        (df["scenario"] == label) &
                        (df["s_true"] == s_true) &
                        (df["h_true"] == h_true) &
                        (df["status"] == "ok")
                    ]

                    if not ok_row.empty:
                        s_inf = float(ok_row["s_inf"].iloc[0])
                        h_inf = float(ok_row["h_inf"].iloc[0])

                        x0_inf = x0_tot_from_vaf0(vaf0, h_inf)
                        vaf_dense_inf = simulate_vaf(
                            t_dense,
                            s_inf,
                            h_inf,
                            x0_inf,
                        )

                        ax.plot(
                            t_dense,
                            vaf_dense_inf,
                            color="tomato",
                            lw=1.5,
                            ls="--",
                            label=f"MAP s={s_inf:.2f}, h={h_inf:.2f}",
                        )

                    v_ceil = (1.0 + h_true) / 2.0

                    ax.axhline(
                        v_ceil,
                        color="grey",
                        lw=0.7,
                        ls=":",
                        alpha=0.6,
                    )

                    # Wilson binomial CI.
                    n_reads = DP[0]
                    k_reads = AO[0]
                    p_hat = obs_vaf
                    z = 1.96

                    denom_w = 1.0 + z**2 / n_reads
                    centre = (p_hat + z**2 / (2 * n_reads)) / denom_w
                    half = (
                        z *
                        np.sqrt(
                            p_hat * (1.0 - p_hat) / n_reads +
                            z**2 / (4 * n_reads**2)
                        )
                    ) / denom_w

                    lo = np.maximum(centre - half, 1e-6)
                    hi = centre + half

                    ax.errorbar(
                        time_points,
                        obs_vaf,
                        yerr=[obs_vaf - lo, hi - obs_vaf],
                        fmt="o",
                        color="steelblue",
                        ms=4,
                        lw=1.0,
                        capsize=2,
                        zorder=5,
                        alpha=0.85,
                    )

                    ax.set_ylim(0, min(v_ceil * 1.15, 1.0))
                    ax.tick_params(labelsize=6)
                    ax.set_ylabel(f"s={s_true}, h={h_true}\nVAF", fontsize=6)

                    if panel_idx == n_rows - 1:
                        ax.set_xlabel("Time (years)", fontsize=7)

                    if panel_idx == 0 and col == 0:
                        ax.legend(fontsize=5, loc="upper left")

                    panel_idx += 1

        fig.suptitle(
            f"VAF trajectories — varied axis: {axis_name}\n"
            "Blue = true | red dashed = inferred MAP | grey dotted = VAF ceiling",
            fontsize=10,
            fontweight="bold",
        )

        plt.tight_layout()

        out = output_base.replace(".png", f"_{axis_name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.show()


# =============================================================================
# Analytical identifiability grid: rho
# =============================================================================

def run_identifiability_grid():
    records = []

    grid = list(
        itertools.product(
            IDENT_S_VALUES,
            IDENT_H_VALUES,
            IDENT_VAF0_VALUES,
            IDENT_DELTA_T_VALUES,
        )
    )

    print()
    print(f"Identifiability rho grid: {len(grid)} cells")
    print()

    for s, h, vaf0, delta_t in grid:
        rho = compute_rho(
            s=s,
            h=h,
            vaf0=vaf0,
            delta_t=delta_t,
            n_tps=N_TPS_MID,
        )

        records.append(
            dict(
                s=s,
                h=h,
                vaf0=vaf0,
                delta_t=delta_t,
                rho=rho,
                identifiable=int(rho >= RHO_THRESHOLD),
            )
        )

    return pd.DataFrame(records)


def plot_identifiability(ident_df, bench_df, output_path=OUTPUT_IDENT_FIG):
    """
    Plot rho heatmaps and empirical h error vs rho.
    """
    n_vaf0 = len(IDENT_VAF0_VALUES)
    n_delta_t = len(IDENT_DELTA_T_VALUES)

    fig_a, axes_a = plt.subplots(
        n_vaf0,
        n_delta_t,
        figsize=(3.5 * n_delta_t, 3.2 * n_vaf0),
        squeeze=False,
    )

    for row_idx, vaf0 in enumerate(IDENT_VAF0_VALUES):
        for col_idx, delta_t in enumerate(IDENT_DELTA_T_VALUES):
            ax = axes_a[row_idx, col_idx]

            sub = ident_df[
                (ident_df["vaf0"] == vaf0) &
                (ident_df["delta_t"] == delta_t)
            ]

            mat = sub.pivot(index="s", columns="h", values="rho")

            im = ax.imshow(
                mat.values,
                origin="lower",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                extent=[
                    mat.columns.min() - 0.08,
                    mat.columns.max() + 0.08,
                    mat.index.min() - 0.08,
                    mat.index.max() + 0.08,
                ],
            )

            try:
                ax.contour(
                    mat.columns,
                    mat.index,
                    mat.values,
                    levels=[RHO_THRESHOLD],
                    colors="white",
                    linewidths=1.5,
                    linestyles="--",
                )
            except Exception:
                pass

            for s_v in mat.index:
                for h_v in mat.columns:
                    val = mat.loc[s_v, h_v]
                    ax.text(
                        h_v,
                        s_v,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if val < 0.6 else "black",
                    )

            ax.set_xlabel("True h", fontsize=7)
            ax.set_ylabel("True s", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(
                f"vaf0={vaf0:.2f}, Δt={delta_t:.1f}y",
                fontsize=7.5,
                fontweight="bold",
            )

            plt.colorbar(im, ax=ax, shrink=0.8, label="ρ", pad=0.02)

    fig_a.suptitle(
        f"Saturation ratio ρ = v_T / v_max(h)\n"
        f"White dashed contour: ρ = {RHO_THRESHOLD}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    path_a = output_path.replace(".png", "_rho_heatmaps.png")
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_a}")
    plt.show()

    # Empirical h diagnostics vs rho.
    ok = bench_df[bench_df["status"] == "ok"].copy()

    if ok.empty:
        print("No successful benchmark rows for rho comparison.")
        return

    fig_b, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ycols = [
        ("h_err_abs", "|h error|", False),
        ("h_overlap_010", "h posterior overlap", True),
        ("h_entropy", "h posterior entropy", False),
    ]

    ok["h_err_abs"] = ok["h_err"].abs()

    axis_colours = {
        "delta_t": "#2271B2",
        "n_tps": "#3DAA6A",
        "vaf0": "#D44D3A",
    }

    for ax, (ycol, ylabel, higher_better) in zip(axes, ycols):
        for axis_name, colour in axis_colours.items():
            sub = ok[ok["axis"] == axis_name]
            ax.scatter(
                sub["rho"],
                sub[ycol],
                color=colour,
                alpha=0.7,
                s=35,
                label=axis_name,
            )

        ax.axvline(
            RHO_THRESHOLD,
            color="grey",
            lw=1.5,
            ls="--",
            label=f"ρ={RHO_THRESHOLD}",
        )

        ax.set_xlabel("ρ = v_T / v_max(h)")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

        if ycol in ["h_overlap_010", "h_entropy"]:
            ax.set_ylim(0, 1)

    axes[0].legend(fontsize=8, framealpha=0.8)

    fig_b.suptitle(
        "Empirical h recovery and posterior identifiability vs saturation ratio",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    path_b = output_path.replace(".png", "_rho_vs_hmetrics.png")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_b}")
    plt.show()


# =============================================================================
# Fisher grids and plots
# =============================================================================

def run_fisher_grid(n_tps=N_TPS_MID, depth=DEPTH):
    grid = list(
        itertools.product(
            IDENT_S_VALUES,
            IDENT_H_VALUES,
            IDENT_VAF0_VALUES,
            IDENT_DELTA_T_VALUES,
        )
    )

    print()
    print(f"Fisher grid: {len(grid)} cells")
    print()

    records = []

    for i, (s, h, vaf0, delta_t) in enumerate(grid):
        fisher, cr_bound = compute_fisher_h(
            s=s,
            h=h,
            vaf0=vaf0,
            delta_t=delta_t,
            n_tps=n_tps,
            depth=depth,
        )

        records.append(
            dict(
                s=s,
                h=h,
                vaf0=vaf0,
                delta_t=delta_t,
                fisher=fisher,
                cr_bound=cr_bound,
                identifiable=int(cr_bound <= CR_BOUND_THRESHOLD),
            )
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(grid)}")

    return pd.DataFrame(records)


def plot_fisher(fisher_df, ident_df, output_path=OUTPUT_FISHER_FIG):
    n_vaf0 = len(IDENT_VAF0_VALUES)
    n_delta_t = len(IDENT_DELTA_T_VALUES)

    fig_a, axes_a = plt.subplots(
        n_vaf0,
        n_delta_t,
        figsize=(3.5 * n_delta_t, 3.2 * n_vaf0),
        squeeze=False,
    )

    for row_idx, vaf0 in enumerate(IDENT_VAF0_VALUES):
        for col_idx, delta_t in enumerate(IDENT_DELTA_T_VALUES):
            ax = axes_a[row_idx, col_idx]

            sub = fisher_df[
                (fisher_df["vaf0"] == vaf0) &
                (fisher_df["delta_t"] == delta_t)
            ]

            mat = sub.pivot(index="s", columns="h", values="cr_bound")

            im = ax.imshow(
                mat.values,
                origin="lower",
                aspect="auto",
                vmin=0.0,
                vmax=0.5,
                cmap="RdYlGn_r",
                extent=[
                    mat.columns.min() - 0.08,
                    mat.columns.max() + 0.08,
                    mat.index.min() - 0.08,
                    mat.index.max() + 0.08,
                ],
            )

            try:
                ax.contour(
                    mat.columns,
                    mat.index,
                    mat.values,
                    levels=[CR_BOUND_THRESHOLD],
                    colors="black",
                    linewidths=1.5,
                    linestyles="--",
                )
            except Exception:
                pass

            for s_v in mat.index:
                for h_v in mat.columns:
                    val = mat.loc[s_v, h_v]
                    ax.text(
                        h_v,
                        s_v,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if val > 0.35 else "black",
                    )

            ax.set_xlabel("True h", fontsize=7)
            ax.set_ylabel("True s", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_title(
                f"vaf0={vaf0:.2f}, Δt={delta_t:.1f}y",
                fontsize=7.5,
                fontweight="bold",
            )

            plt.colorbar(
                im,
                ax=ax,
                shrink=0.8,
                label="CR bound",
                pad=0.02,
            )

    fig_a.suptitle(
        "Cramér-Rao bound for h\n"
        f"Black dashed contour: CR bound = {CR_BOUND_THRESHOLD}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    path_a = output_path.replace(".png", "_cr_heatmaps.png")
    fig_a.savefig(path_a, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_a}")
    plt.show()

    merged = fisher_df.merge(
        ident_df[["s", "h", "vaf0", "delta_t", "rho"]],
        on=["s", "h", "vaf0", "delta_t"],
        how="inner",
    )

    fig_b, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(
        merged["rho"],
        merged["cr_bound"],
        c=merged["h"],
        cmap="plasma",
        s=25,
        alpha=0.75,
    )

    axes[0].axhline(
        CR_BOUND_THRESHOLD,
        color="black",
        lw=1.5,
        ls="--",
        label=f"CR={CR_BOUND_THRESHOLD}",
    )
    axes[0].axvline(
        RHO_THRESHOLD,
        color="grey",
        lw=1.5,
        ls="--",
        label=f"ρ={RHO_THRESHOLD}",
    )
    axes[0].set_xlabel("ρ")
    axes[0].set_ylabel("CR bound")
    axes[0].set_title("ρ vs CR bound")
    axes[0].legend(fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc, ax=axes[0], label="True h")

    sc2 = axes[1].scatter(
        merged["rho"],
        np.log10(np.maximum(merged["fisher"], 1e-12)),
        c=merged["vaf0"],
        cmap="viridis",
        s=25,
        alpha=0.75,
    )

    axes[1].axvline(
        RHO_THRESHOLD,
        color="grey",
        lw=1.5,
        ls="--",
    )
    axes[1].set_xlabel("ρ")
    axes[1].set_ylabel("log10 Fisher information")
    axes[1].set_title("ρ vs Fisher information")
    axes[1].spines[["top", "right"]].set_visible(False)
    plt.colorbar(sc2, ax=axes[1], label="Initial VAF")

    fig_b.suptitle(
        "Relationship between saturation ratio and Fisher information",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    path_b = output_path.replace(".png", "_fisher_vs_rho.png")
    fig_b.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_b}")
    plt.show()

    # Agreement plot.
    fig_c, ax = plt.subplots(figsize=(7, 6))

    colours = []

    for _, r in merged.iterrows():
        rho_ok = r["rho"] >= RHO_THRESHOLD
        cr_ok = r["cr_bound"] <= CR_BOUND_THRESHOLD

        if rho_ok and cr_ok:
            colours.append("#3DAA6A")
        elif not rho_ok and not cr_ok:
            colours.append("#D44D3A")
        elif not rho_ok and cr_ok:
            colours.append("#F0A500")
        else:
            colours.append("#8E5FB9")

    ax.scatter(
        merged["rho"],
        merged["cr_bound"],
        c=colours,
        s=30,
        alpha=0.75,
    )

    ax.axhline(CR_BOUND_THRESHOLD, color="black", lw=1.5, ls="--")
    ax.axvline(RHO_THRESHOLD, color="grey", lw=1.5, ls="--")

    ax.set_xlabel("ρ")
    ax.set_ylabel("CR bound")
    ax.set_title(
        "Agreement between ρ and CR-bound identifiability\n"
        "Green: both identifiable | Red: both unidentifiable | "
        "Amber/Purple: disagreement",
        fontsize=10,
        fontweight="bold",
    )

    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path_c = output_path.replace(".png", "_rho_cr_agreement.png")
    fig_c.savefig(path_c, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_c}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # Run simulation/inference benchmark.
    df = run_benchmark()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    print_summary(df)

    # Main benchmark plots.
    plot_sensitivity(df, OUTPUT_HEATMAPS)
    plot_h_identifiability_summary(df, OUTPUT_H_IDENT_SUMMARY)
    plot_posterior_overlap_examples(df, OUTPUT_POSTERIOR_EXAMPLES)
    plot_vaf_trajectories(df, OUTPUT_VAF_TRAJECTORIES_BASE)

    # Analytical saturation-ratio identifiability.
    ident_df = run_identifiability_grid()
    ident_df.to_csv(OUTPUT_IDENT_CSV, index=False)
    print(f"Saved: {OUTPUT_IDENT_CSV}")

    plot_identifiability(ident_df, df, OUTPUT_IDENT_FIG)

    # Fisher information identifiability.
    fisher_df = run_fisher_grid()
    fisher_df.to_csv(OUTPUT_FISHER_CSV, index=False)
    print(f"Saved: {OUTPUT_FISHER_CSV}")

    plot_fisher(fisher_df, ident_df, OUTPUT_FISHER_FIG)
