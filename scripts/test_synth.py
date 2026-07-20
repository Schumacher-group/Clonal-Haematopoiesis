"""Synthetic recovery benchmark — dual pipeline comparison.

Compares two inference approaches on the same synthetic (s, h) grid:

  det  — infer_sh_jointly_from_dynamics
         Deterministic exponential trajectory + binomial likelihood.
         Cheap. Leading mutation only per clone.

  hmm  — jax_cs_hmm_ll_vec_mixed
         Full birth-death HMM marginalised over a (s, h) grid.
         Expensive. Uses all mutations.

Both pipelines share:
  - identical synthetic data per grid cell (same seed)
  - same s_resolution / h_resolution grids
  - MAP extracted from the joint 2D posterior via argmax + unravel_index

Outputs
-------
  synthetic_recovery_results.csv        — per-cell records for both pipelines
  synthetic_recovery_heatmaps.png       — side-by-side heatmap comparison
  synthetic_recovery_trajectories.png   — VAF trajectory grids, one per pipeline

Notes
-----
- h is STATIC through time
- sequencing noise is binomial
- timepoints are adaptive: spaced to end near the VAF ceiling
"""

import sys
import itertools
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.append("..")
sys.path.append("../src")

# =============================================================================
# Configuration
# =============================================================================

S_TRUE_VALUES = np.round(np.arange(0.4, 0.85, 0.2), 2)
H_TRUE_VALUES = np.round(np.arange(0.0, 1.05, 0.3), 2)

N_W       = 1e5
DEPTH     = 2000
SEED      = 42

N_TIMEPOINTS  = 4
SPACING_YEARS = 2.5
INITIAL_VAF   = 0.25

# Shared grid resolution for both pipelines
S_RESOLUTION = 30
H_RESOLUTION = 20

OUTPUT_CSV = "synthetic_recovery_results.csv"
OUTPUT_FIG = "synthetic_recovery_heatmaps.png"

# Pipelines to run — set either to False to skip
RUN_DET = True
RUN_HMM = True

# HMM-specific: lower resolution for speed; raise for publication runs
HMM_RESOLUTION = 400   # Monte Carlo samples inside compute_global_variables_mixed

# =============================================================================
# Minimal AnnData-compatible container
# =============================================================================

class SimpleVar:
    def __init__(self, time_points):
        self.time_points = pd.Series(time_points)
        self.columns = ["time_points"]


class SimpleObs:
    def __init__(self, n):
        self._df = pd.DataFrame(index=[f"mut_{i}" for i in range(n)])
        self.index   = self._df.index
        self.columns = self._df.columns

    @property
    def iloc(self):
        return self._df.iloc

    def __getitem__(self, key):        return self._df[key]
    def __setitem__(self, key, value): self._df[key] = value; self.columns = self._df.columns
    def __contains__(self, key):       return key in self._df.columns
    def to_string(self):               return self._df.to_string()


class SyntheticParticipant:
    def __init__(self, AO, DP, time_points):
        assert AO.shape == DP.shape
        n_mutations, n_timepoints = AO.shape
        self.layers = {"AO": AO, "DP": DP}
        self.var    = SimpleVar(time_points)
        self.obs    = SimpleObs(n_mutations)
        self.uns    = {}
        self.n_obs  = n_mutations
        self.n_vars = n_timepoints
        self.shape  = (n_mutations, n_timepoints)

    def __getitem__(self, idx):
        part = SyntheticParticipant(
            self.layers["AO"][idx:idx+1],
            self.layers["DP"][idx:idx+1],
            np.asarray(self.var.time_points),
        )
        part.obs._df  = self.obs._df.iloc[idx:idx+1].copy()
        part.obs.index = part.obs._df.index
        return part

    @property
    def X(self):
        return self.layers["AO"] / np.maximum(self.layers["DP"], 1.0)


# =============================================================================
# Synthetic forward model
# =============================================================================

def simulate_clone(time_points, s, xhet0, h, N_w=N_W):
    """Forward model: mixed-zygosity exponential clone growth."""
    t     = np.asarray(time_points, dtype=float)
    x_het = xhet0 * np.exp(s * (t - t[0]))
    x_hom = h * x_het
    x_tot = x_het + x_hom
    vaf   = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    return x_het, x_hom, vaf


def make_time_points(n_points=N_TIMEPOINTS, spacing_years=SPACING_YEARS, t_ceiling=None):
    """Adaptive timepoints ending at t_ceiling, or fixed spacing from 0."""
    if t_ceiling is None:
        return np.arange(n_points, dtype=float) * spacing_years
    t_end   = t_ceiling
    t_start = t_end - (n_points - 1) * spacing_years
    return np.linspace(t_start, t_end, n_points)


def compute_t_ceiling(s, h, xhet0, N_w=N_W, fraction=0.99):
    """Time at which x_tot reaches fraction * N_w / (1-fraction)."""
    x_tot0        = xhet0 * (1 + h)
    x_tot_target  = fraction * N_w / (1 - fraction)
    t_ceiling     = np.log(x_tot_target / x_tot0) / s
    t_ceiling     = max(t_ceiling, (N_TIMEPOINTS - 1) * SPACING_YEARS)
    return t_ceiling


def generate_synthetic_participant(s_true, h_true, depth, seed, N_w=N_W):
    rng    = np.random.default_rng(seed)
    xhet0  = max(2.0 * N_w * INITIAL_VAF / (1.0 + h_true), 10.0)
    t_ceil = compute_t_ceiling(s_true, h_true, xhet0, N_w)
    tps    = make_time_points(t_ceiling=t_ceil)

    _, _, vaf_true = simulate_clone(tps, s_true, xhet0, h_true, N_w)
    vaf_ceiling    = (1.0 + 2.0 * h_true) / (2.0 * (1.0 + h_true))
    vaf_true       = np.clip(vaf_true, 1e-6, vaf_ceiling - 1e-6)

    DP = np.full((1, len(tps)), depth, dtype=float)
    AO = rng.binomial(depth, vaf_true).reshape(1, -1).astype(float)

    return SyntheticParticipant(AO, DP, tps)


# =============================================================================
# Pipeline wrappers — uniform interface
#
# Both return: s_map, h_map, s_ci=(lo, hi), h_ci=(lo, hi), elapsed_s
# =============================================================================

def run_det(AO, DP, time_points, cs, s_resolution, h_resolution):
    """Wrapper around infer_sh_jointly_from_dynamics."""
    from KI_clonal_inference_6 import infer_sh_jointly_from_dynamics

    t0 = time.perf_counter()
    results = infer_sh_jointly_from_dynamics(
        cs, AO, DP, time_points,
        s_resolution=s_resolution,
        h_resolution=h_resolution,
    )
    elapsed = time.perf_counter() - t0

    r = results[0]
    return r["s_map"], r["h_map"], r["s_ci"], r["h_ci"], elapsed


def run_hmm(AO, DP, time_points, cs, s_resolution, h_resolution, seed_offset=0):
    """Wrapper around jax_cs_hmm_ll_vec_mixed with MAP extraction."""
    import jax.numpy as jnp
    import jax.random as jrnd
    import numpy as np
    from KI_clonal_inference_5 import (
        compute_deterministic_size_mixed,
        jax_cs_hmm_ll_vec_mixed,
    )

    EPS = 1e-8

    t0    = time.perf_counter()
    key   = jrnd.PRNGKey(758493 + seed_offset)
    AO_j  = jnp.array(AO)
    DP_j  = jnp.array(DP)
    tps_j = jnp.array(time_points)

    s_vec = jnp.linspace(0.01, 1.0, s_resolution)
    h_vec = jnp.linspace(0.0,  1.0, h_resolution)

    det_size, total_cells, max_total, _ = compute_deterministic_size_mixed(
        cs, AO_j, DP_j, AO_j.shape[1]
    )

    # output shape: (n_s, n_h, n_clones)
    output = jax_cs_hmm_ll_vec_mixed(
        s_vec, h_vec, AO_j, DP_j, tps_j,
        cs, det_size, total_cells, max_total,
        key, resolution=HMM_RESOLUTION,
    )
    elapsed = time.perf_counter() - t0

    # Clone 0 posterior (single-mutation benchmark)
    grid = np.array(output[:, :, 0], copy=False)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.maximum(grid, 0.0)

    total = grid.sum()
    if total == 0:
        return np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan), elapsed

    joint_posterior = grid / total

    # MAP from 2D joint
    s_arr = np.asarray(s_vec)
    h_arr = np.asarray(h_vec)
    mi = np.argmax(joint_posterior)
    si, hi = np.unravel_index(mi, joint_posterior.shape)
    s_map = float(s_arr[si])
    h_map = float(h_arr[hi])

    # Marginal CIs
    s_post = joint_posterior.sum(axis=1)
    h_post = joint_posterior.sum(axis=0)
    s_post = s_post / max(s_post.sum(), EPS)
    h_post = h_post / max(h_post.sum(), EPS)

    sc = np.cumsum(s_post)
    hc = np.cumsum(h_post)

    s_ci = (
        float(s_arr[min(np.searchsorted(sc, 0.05), len(s_arr) - 1)]),
        float(s_arr[min(np.searchsorted(sc, 0.95), len(s_arr) - 1)]),
    )
    h_ci = (
        float(h_arr[min(np.searchsorted(hc, 0.05), len(h_arr) - 1)]),
        float(h_arr[min(np.searchsorted(hc, 0.95), len(h_arr) - 1)]),
    )

    return s_map, h_map, s_ci, h_ci, elapsed


# =============================================================================
# Grid runner
# =============================================================================

def _empty_record(s_true, h_true, pipeline, msg):
    return dict(
        pipeline=pipeline,
        s_true=s_true, h_true=h_true,
        s_inf=np.nan, h_inf=np.nan,
        s_err=np.nan, h_err=np.nan,
        s_lo=np.nan,  s_hi=np.nan,
        h_lo=np.nan,  h_hi=np.nan,
        s_in_ci=np.nan, h_in_ci=np.nan,
        elapsed_s=np.nan,
        status=f"error: {msg}",
    )


def run_grid():
    grid    = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))
    n_total = len(grid)
    records = []

    print(f"\nRunning {n_total} grid cells × "
          f"{int(RUN_DET) + int(RUN_HMM)} pipeline(s)\n")

    for idx, (s_true, h_true) in enumerate(grid):

        print(f"[{idx+1:3d}/{n_total}]  s={s_true:.2f}  h={h_true:.2f}")

        part = generate_synthetic_participant(
            s_true=s_true, h_true=h_true, depth=DEPTH, seed=SEED + idx
        )

        obs_vaf = (part.layers["AO"] / np.maximum(part.layers["DP"], 1.0))[0]
        obs_tps = np.asarray(part.var.time_points)
        vaf_ceil = (1.0 + 2.0 * h_true) / (2.0 * (1.0 + h_true))
        vaf_strs = "  ".join(f"t={t:.1f}y:{v:.3f}" for t, v in zip(obs_tps, obs_vaf))
        print(f"         VAFs: {vaf_strs}  [ceil={vaf_ceil:.3f}]")

        AO         = part.layers["AO"].T          # (n_tps, n_muts)
        DP         = part.layers["DP"].T
        time_points = np.asarray(part.var.time_points)
        cs         = [[0]]

        # ------------------------------------------------------------------ #
        # Pipeline A: deterministic trajectory                                #
        # ------------------------------------------------------------------ #
        if RUN_DET:
            try:
                s_map, h_map, s_ci, h_ci, elapsed = run_det(
                    AO, DP, time_points, cs, S_RESOLUTION, H_RESOLUTION
                )
                se = s_map - s_true
                he = h_map - h_true
                print(
                    f"  [det]  s={s_map:.3f} ({se:+.3f}) [{s_ci[0]:.2f},{s_ci[1]:.2f}]"
                    f"  h={h_map:.3f} ({he:+.3f}) [{h_ci[0]:.2f},{h_ci[1]:.2f}]"
                    f"  {elapsed:.1f}s"
                )
                records.append(dict(
                    pipeline="det",
                    s_true=s_true, h_true=h_true,
                    s_inf=s_map,   h_inf=h_map,
                    s_err=se,      h_err=he,
                    s_lo=s_ci[0],  s_hi=s_ci[1],
                    h_lo=h_ci[0],  h_hi=h_ci[1],
                    s_in_ci=int(s_ci[0] <= s_true <= s_ci[1]),
                    h_in_ci=int(h_ci[0] <= h_true <= h_ci[1]),
                    elapsed_s=elapsed,
                    status="ok",
                ))
            except Exception as e:
                traceback.print_exc()
                records.append(_empty_record(s_true, h_true, "det", e))

        # ------------------------------------------------------------------ #
        # Pipeline B: full HMM                                                #
        # ------------------------------------------------------------------ #
        if RUN_HMM:
            try:
                s_map, h_map, s_ci, h_ci, elapsed = run_hmm(
                    AO, DP, time_points, cs, S_RESOLUTION, H_RESOLUTION,
                    seed_offset=idx,
                )
                se = s_map - s_true
                he = h_map - h_true
                print(
                    f"  [hmm]  s={s_map:.3f} ({se:+.3f}) [{s_ci[0]:.2f},{s_ci[1]:.2f}]"
                    f"  h={h_map:.3f} ({he:+.3f}) [{h_ci[0]:.2f},{h_ci[1]:.2f}]"
                    f"  {elapsed:.1f}s"
                )
                records.append(dict(
                    pipeline="hmm",
                    s_true=s_true, h_true=h_true,
                    s_inf=s_map,   h_inf=h_map,
                    s_err=se,      h_err=he,
                    s_lo=s_ci[0],  s_hi=s_ci[1],
                    h_lo=h_ci[0],  h_hi=h_ci[1],
                    s_in_ci=int(s_ci[0] <= s_true <= s_ci[1]),
                    h_in_ci=int(h_ci[0] <= h_true <= h_ci[1]),
                    elapsed_s=elapsed,
                    status="ok",
                ))
            except Exception as e:
                traceback.print_exc()
                records.append(_empty_record(s_true, h_true, "hmm", e))

        print()

    return pd.DataFrame(records)


# =============================================================================
# Summary statistics
# =============================================================================

def print_summary(df):
    print("\n" + "=" * 70)
    print("RECOVERY SUMMARY")
    print("=" * 70)

    for pipe in df["pipeline"].unique():
        ok = df[(df["pipeline"] == pipe) & (df["status"] == "ok")]
        h_cov = pd.to_numeric(ok["h_in_ci"], errors="coerce").mean()
        t_mean = ok["elapsed_s"].mean()

        print(f"\n  Pipeline: {pipe.upper()}")
        print(f"    Completed:         {len(ok)} / {len(df[df['pipeline']==pipe])}")
        print(f"    s  MAE:            {ok['s_err'].abs().mean():.3f}")
        print(f"    h  MAE:            {ok['h_err'].abs().mean():.3f}")
        print(f"    s  bias:           {ok['s_err'].mean():+.3f}")
        print(f"    h  bias:           {ok['h_err'].mean():+.3f}")
        print(f"    s  90% CI coverage:{ok['s_in_ci'].mean()*100:6.1f}%")
        print(f"    h  90% CI coverage:{h_cov*100:6.1f}%")
        print(f"    Mean runtime/cell: {t_mean:.2f}s")

    # Head-to-head: cells where both pipelines succeeded
    if df["pipeline"].nunique() == 2:
        det = df[df["pipeline"] == "det"].set_index(["s_true", "h_true"])
        hmm = df[df["pipeline"] == "hmm"].set_index(["s_true", "h_true"])
        both = det.join(hmm, lsuffix="_det", rsuffix="_hmm", how="inner")
        both = both[(both["status_det"] == "ok") & (both["status_hmm"] == "ok")]
        if len(both):
            ds_diff = (both["s_inf_det"] - both["s_inf_hmm"]).abs()
            dh_diff = (both["h_inf_det"] - both["h_inf_hmm"]).abs()
            print(f"\n  Head-to-head MAP agreement ({len(both)} cells):")
            print(f"    |s_det - s_hmm| mean: {ds_diff.mean():.3f}  max: {ds_diff.max():.3f}")
            print(f"    |h_det - h_hmm| mean: {dh_diff.mean():.3f}  max: {dh_diff.max():.3f}")

    print("=" * 70)


# =============================================================================
# Plotting utilities
# =============================================================================

def pivot(df, value_col):
    return df.pivot(index="s_true", columns="h_true", values=value_col)


def draw_heatmap(ax, mat, title, cmap, vmin=None, vmax=None,
                 centered=False, fmt="{:.2f}"):
    kwargs = dict(cmap=cmap, aspect="auto")
    if centered:
        kwargs["norm"] = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    im = ax.imshow(
        mat.values, origin="lower",
        extent=[
            mat.columns.min() - 0.05, mat.columns.max() + 0.05,
            mat.index.min()   - 0.05, mat.index.max()   + 0.05,
        ],
        **kwargs,
    )
    for s in mat.index:
        for h in mat.columns:
            val = mat.loc[s, h]
            if np.isnan(val):
                continue
            ax.text(h, s, fmt.format(val),
                    ha="center", va="center", fontsize=7)
    ax.set_xlabel("True h")
    ax.set_ylabel("True s")
    ax.set_title(title, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)


# =============================================================================
# Figure 1 — side-by-side heatmaps per pipeline
# =============================================================================

def plot_heatmaps(df, output_path):
    pipelines = [p for p in ["det", "hmm"] if p in df["pipeline"].values]
    n_pipes   = len(pipelines)

    # Guard: nothing to plot if all cells failed
    for pipe in pipelines:
        sub = df[(df["pipeline"] == pipe) & (df["status"] == "ok")]
        if sub.empty:
            print(f"[{pipe}] No successful runs — skipping heatmaps.")
            return

    # ... rest unchanged
    """
    Layout: one block of 2 rows × 4 cols per pipeline.

    Row 1: inferred s | inferred h | s CI width | h CI width
    Row 2: s error    | h error    | s in CI    | h in CI
    """
    pipelines = [p for p in ["det", "hmm"] if p in df["pipeline"].values]
    n_pipes   = len(pipelines)

    fig, axes = plt.subplots(
        2 * n_pipes, 4,
        figsize=(26, 10 * n_pipes),
        squeeze=False,
    )

    labels = {"det": "Det (trajectory)", "hmm": "HMM"}

    for block, pipe in enumerate(pipelines):
        sub = df[(df["pipeline"] == pipe) & (df["status"] == "ok")].copy()
        sub["s_ci_width"] = sub["s_hi"] - sub["s_lo"]
        sub["h_ci_width"] = sub["h_hi"] - sub["h_lo"]

        row0 = block * 2
        row1 = row0 + 1
        tag  = labels[pipe]

        ok = sub
        s_mae = ok["s_err"].abs().mean()
        h_mae = ok["h_err"].abs().mean()
        s_cov = ok["s_in_ci"].mean() * 100
        h_cov = pd.to_numeric(ok["h_in_ci"], errors="coerce").mean() * 100

        draw_heatmap(axes[row0, 0], pivot(sub, "s_inf"),
                     f"[{tag}] Inferred s", "viridis", 0, 1)
        draw_heatmap(axes[row0, 1], pivot(sub, "h_inf"),
                     f"[{tag}] Inferred h", "plasma",  0, 1)
        draw_heatmap(axes[row0, 2], pivot(sub, "s_ci_width"),
                     f"[{tag}] s 90% CI width", "YlOrRd", 0, 0.5)
        draw_heatmap(axes[row0, 3], pivot(sub, "h_ci_width"),
                     f"[{tag}] h 90% CI width", "YlOrRd", 0, 0.5)

        draw_heatmap(axes[row1, 0], pivot(sub, "s_err"),
                     f"[{tag}] s error", "RdBu_r", -0.5, 0.5, centered=True)
        draw_heatmap(axes[row1, 1], pivot(sub, "h_err"),
                     f"[{tag}] h error", "RdBu_r", -0.5, 0.5, centered=True)
        draw_heatmap(axes[row1, 2], pivot(sub, "s_in_ci"),
                     f"[{tag}] s in 90% CI", "RdYlGn", 0, 1, fmt="{:.0f}")
        draw_heatmap(axes[row1, 3], pivot(sub, "h_in_ci"),
                     f"[{tag}] h in 90% CI", "RdYlGn", 0, 1, fmt="{:.0f}")

        # Per-pipeline footer
        fig.text(
            0.5, 1.0 - (block + 1) / n_pipes + 0.01,
            f"{tag} — s: MAE={s_mae:.3f}, 90%CI={s_cov:.0f}%  |  "
            f"h: MAE={h_mae:.3f}, 90%CI={h_cov:.0f}%",
            ha="center", fontsize=10, style="italic",
        )

    fig.suptitle(
        "Synthetic recovery benchmark — pipeline comparison",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap figure saved: {output_path}")
    plt.show()


# =============================================================================
# Figure 2 — MAP agreement between pipelines (difference heatmaps)
# =============================================================================

def plot_comparison_heatmaps(df, output_path):
    """
    Only produced when both pipelines ran successfully.
    Shows: s_det - s_hmm, h_det - h_hmm, and |error| improvement.
    """
    if df["pipeline"].nunique() < 2:
        print("Skipping comparison heatmaps — only one pipeline ran.")
        return

    det = df[df["pipeline"] == "det"].set_index(["s_true", "h_true"])
    hmm = df[df["pipeline"] == "hmm"].set_index(["s_true", "h_true"])
    both = det.join(hmm, lsuffix="_det", rsuffix="_hmm", how="inner")
    both = both[(both["status_det"] == "ok") & (both["status_hmm"] == "ok")].reset_index()

    if both.empty:
        print("No overlapping successful cells — skipping comparison figure.")
        return

    both["ds"]          = both["s_inf_det"] - both["s_inf_hmm"]
    both["dh"]          = both["h_inf_det"] - both["h_inf_hmm"]
    both["s_abserr_det"] = both["s_err_det"].abs()
    both["s_abserr_hmm"] = both["s_err_hmm"].abs()
    both["h_abserr_det"] = both["h_err_det"].abs()
    both["h_abserr_hmm"] = both["h_err_hmm"].abs()
    # positive = det wins (smaller error), negative = hmm wins
    both["s_improvement"] = both["s_abserr_hmm"] - both["s_abserr_det"]
    both["h_improvement"] = both["h_abserr_hmm"] - both["h_abserr_det"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Pipeline comparison: det vs HMM\n"
        "(improvement > 0 → det wins; < 0 → HMM wins)",
        fontsize=13, fontweight="bold",
    )

    draw_heatmap(axes[0, 0], pivot(both, "ds"),
                 "s_det − s_hmm  (MAP agreement)",
                 "RdBu_r", -0.3, 0.3, centered=True)
    draw_heatmap(axes[0, 1], pivot(both, "s_abserr_det"),
                 "|s error| det", "YlOrRd", 0, 0.5)
    draw_heatmap(axes[0, 2], pivot(both, "s_abserr_hmm"),
                 "|s error| hmm", "YlOrRd", 0, 0.5)

    draw_heatmap(axes[1, 0], pivot(both, "dh"),
                 "h_det − h_hmm  (MAP agreement)",
                 "RdBu_r", -0.3, 0.3, centered=True)
    draw_heatmap(axes[1, 1], pivot(both, "h_abserr_det"),
                 "|h error| det", "YlOrRd", 0, 0.5)
    draw_heatmap(axes[1, 2], pivot(both, "h_abserr_hmm"),
                 "|h error| hmm", "YlOrRd", 0, 0.5)

    plt.tight_layout()
    cmp_path = output_path.replace(".png", "_comparison.png")
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved: {cmp_path}")
    plt.show()


# =============================================================================
# Figure 3 — VAF trajectories (one panel per grid cell, one figure per pipeline)
# =============================================================================

def plot_vaf_trajectories(df, output_path):
    """
    One trajectory figure per pipeline.  Each panel shows:
      - True VAF trajectory          (blue solid)
      - Inferred MAP trajectory      (red dashed)
      - Observed noisy sample points (blue dots)
      - VAF ceiling (1+h)/2          (grey dotted)
    """
    s_vals    = sorted(df["s_true"].unique())
    h_vals    = sorted(df["h_true"].unique())
    n_rows, n_cols = len(s_vals), len(h_vals)
    grid_cells = list(itertools.product(s_vals, h_vals))

    pipelines = [p for p in ["det", "hmm"] if p in df["pipeline"].values]
    labels    = {"det": "Det (trajectory)", "hmm": "HMM"}
    colours   = {"det": "tomato",           "hmm": "darkorange"}

    for pipe in pipelines:
        sub = df[df["pipeline"] == pipe].set_index(["s_true", "h_true"])

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.5 * n_cols, 2.2 * n_rows),
            sharex=False, sharey=True,
        )

        for row_idx, s_true in enumerate(s_vals):
            for col_idx, h_true in enumerate(h_vals):
                ax = axes[row_idx, col_idx]

                try:
                    r = sub.loc[(s_true, h_true)]
                except KeyError:
                    ax.set_visible(False)
                    continue

                if r["status"] != "ok":
                    ax.text(0.5, 0.5, "FAILED", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7, color="red")
                    ax.set_visible(True)
                    continue

                # Regenerate participant to get actual timepoints used
                grid_idx    = grid_cells.index((s_true, h_true))
                part        = generate_synthetic_participant(
                    s_true, h_true, DEPTH, seed=SEED + grid_idx)
                tps         = np.asarray(part.var.time_points)
                obs_vaf     = (part.layers["AO"] / np.maximum(part.layers["DP"], 1.0))[0]

                xhet0 = max(2.0 * N_W * INITIAL_VAF / (1.0 + h_true), 10.0)
                _, _, vaf_true = simulate_clone(tps, s_true, xhet0, h_true)
                ax.plot(tps, vaf_true, color="steelblue", lw=1.5, label="True")

                s_inf = r["s_inf"]
                h_inf = r["h_inf"]
                if not (np.isnan(s_inf) or np.isnan(h_inf)):
                    xhet0_inf   = max(2.0 * N_W * INITIAL_VAF / (1.0 + h_inf), 10.0)
                    _, _, vaf_inf = simulate_clone(tps, s_inf, xhet0_inf, h_inf)
                    ax.plot(tps, vaf_inf,
                            color=colours[pipe], lw=1.5, ls="--",
                            label=f"{labels[pipe]}")

                ax.scatter(tps, obs_vaf, color="steelblue", s=14, zorder=5, alpha=0.8)

                ceiling = (1.0 + 2.0 * h_true) / (2.0 * (1.0 + h_true))
                ax.axhline(ceiling, color="grey", lw=0.7, ls=":")

                ax.set_ylim(0, 1)
                ax.set_xlim(tps[0] - 0.5, tps[-1] + 0.5)
                ax.tick_params(labelsize=5)

                if row_idx == n_rows - 1:
                    ax.set_xlabel(f"h={h_true:.1f}", fontsize=6)
                if col_idx == 0:
                    ax.set_ylabel(f"s={s_true:.1f}", fontsize=6)

                if not (np.isnan(s_inf) or np.isnan(h_inf)):
                    ax.text(0.97, 0.05,
                            f"s={s_inf:.2f}\nh={h_inf:.2f}",
                            transform=ax.transAxes, fontsize=5,
                            ha="right", va="bottom", color=colours[pipe])

        axes[0, 0].legend(fontsize=5, loc="upper left")
        fig.suptitle(
            f"[{labels[pipe]}] VAF trajectories: true (blue) vs inferred ({colours[pipe]} dashed)\n"
            "dots = observed  |  grey dotted = VAF ceiling",
            fontsize=9,
        )
        plt.tight_layout()
        traj_path = output_path.replace(".png", f"_trajectories_{pipe}.png")
        fig.savefig(traj_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory figure saved: {traj_path}")
        plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    df = run_grid()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    print_summary(df)
    plot_heatmaps(df, OUTPUT_FIG)
    plot_comparison_heatmaps(df, OUTPUT_FIG)
    plot_vaf_trajectories(df, OUTPUT_FIG)