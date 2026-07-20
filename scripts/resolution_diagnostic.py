"""
resolution_diagnostic.py
------------------------
Tests whether poor h recovery is a grid-resolution artefact or a genuine
likelihood flatness problem.

Fixed conditions (chosen from OAT benchmark — moderate Fisher CR bound,
low vaf0 regime):
    vaf0    = 0.03
    delta_t = 3.0 y
    h_true  = 0.5   (midpoint, away from boundaries)
    depth   = 500

Two s values tested:
    s = 0.2  (slow growth — clone barely moves in VAF over 3y)
    s = 0.6  (moderate growth — clone grows meaningfully)

For each (s_true, H_RESOLUTION) combination:
    - Generate synthetic data (fixed seed)
    - Run infer_sh_jointly_from_dynamics
    - Store full 2D joint posterior, marginal h posterior, MAP, CI

Outputs
-------
    resolution_diagnostic_posteriors.png  — 2D joint + marginal h per combo
    resolution_diagnostic_summary.png     — MAP and CI width vs resolution
"""

import sys
import itertools
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append("..")
sys.path.append("../src")

from KI_clonal_inference_6 import infer_sh_jointly_from_dynamics

# =============================================================================
# Configuration
# =============================================================================

VAF0    = 0.03
DELTA_T = 3.0
H_TRUE  = 0.5
DEPTH   = 500
N_W     = 1e5
SEED    = 42
N_TPS   = 4        # evenly spaced for cleanliness
S_RESOLUTION = 30  # fixed — we are varying H_RESOLUTION only

S_VALUES      = [0.2, 0.6]
H_RESOLUTIONS = [10, 20, 40, 80, 150]

OUTPUT_POSTERIORS = "resolution_diagnostic_posteriors.png"
OUTPUT_SUMMARY    = "resolution_diagnostic_summary.png"

# =============================================================================
# Data generation (same helpers as sensitivity_benchmark.py)
# =============================================================================

def x0_het_from_vaf0(vaf0, h, N_w=N_W):
    denom  = (1.0 + 2.0 * h) - 2.0 * vaf0
    denom  = max(denom, 1e-8)
    x_tot0 = 2.0 * vaf0 * N_w / denom
    return x_tot0 / (1.0 + h)


def simulate_vaf(time_points, s, h, x0_het, N_w=N_W):
    t     = np.asarray(time_points, dtype=float)
    x_het = x0_het * np.exp(s * (t - t[0]))
    x_hom = h * x_het
    x_tot = x_het + x_hom
    vaf   = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    v_max = (1.0 + 2.0 * h) / (2.0 * (1.0 + h))
    return np.clip(vaf, 1e-8, v_max - 1e-8)


def generate_data(s_true, h_true, vaf0, delta_t, n_tps, depth, seed):
    rng        = np.random.default_rng(seed)
    time_points = np.linspace(0.0, delta_t * (n_tps - 1), n_tps)
    x0_het     = x0_het_from_vaf0(vaf0, h_true)
    vaf_true   = simulate_vaf(time_points, s_true, h_true, x0_het)

    DP = np.full((1, n_tps), depth, dtype=float)
    AO = rng.binomial(int(depth), vaf_true).reshape(1, -1).astype(float)

    return AO, DP, time_points, vaf_true


# =============================================================================
# Run inference at one (s_true, h_resolution) combination
# =============================================================================

def run_one(s_true, h_res, AO, DP, time_points):
    """Run det inference and return full posterior details."""
    t0 = time.perf_counter()
    results = infer_sh_jointly_from_dynamics(
        cs=[[0]],
        AO=AO.T,
        DP=DP.T,
        time_points=time_points,
        s_resolution=S_RESOLUTION,
        h_resolution=h_res,
    )
    elapsed = time.perf_counter() - t0

    r = results[0]
    return dict(
        s_true       = s_true,
        h_res        = h_res,
        s_map        = r["s_map"],
        h_map        = r["h_map"],
        s_ci         = r["s_ci"],
        h_ci         = r["h_ci"],
        h_ci_width   = r["h_ci"][1] - r["h_ci"][0],
        s_posterior  = r["s_posterior"],
        h_posterior  = r["h_posterior"],
        joint_posterior = r["joint_posterior"],
        s_range      = r["s_range"],
        h_range      = r["h_range"],
        elapsed_s    = elapsed,
    )


# =============================================================================
# Main
# =============================================================================

def run_all():
    results = {}

    for s_true in S_VALUES:
        print(f"\n{'='*55}")
        print(f"  s_true = {s_true}  (vaf0={VAF0}, delta_t={DELTA_T}y, h_true={H_TRUE})")
        print(f"{'='*55}")

        AO, DP, time_points, vaf_true = generate_data(
            s_true, H_TRUE, VAF0, DELTA_T, N_TPS, DEPTH, SEED
        )

        obs_vaf = (AO / np.maximum(DP, 1.0))[0]
        v_max   = (1.0 + 2.0 * H_TRUE) / (2.0 * (1.0 + H_TRUE))
        print(f"  True VAF trajectory: {np.round(vaf_true, 4)}")
        print(f"  Observed VAF:        {np.round(obs_vaf, 4)}")
        print(f"  VAF ceiling:         {v_max:.4f}")
        print(f"  rho (v_T/v_max):     {vaf_true[-1]/v_max:.3f}")
        print()

        results[s_true] = []
        for h_res in H_RESOLUTIONS:
            r = run_one(s_true, h_res, AO, DP, time_points)
            results[s_true].append(r)
            print(
                f"  H_RES={h_res:3d}"
                f"  s_map={r['s_map']:.3f}"
                f"  h_map={r['h_map']:.3f}"
                f"  h_CI=[{r['h_ci'][0]:.2f},{r['h_ci'][1]:.2f}]"
                f"  width={r['h_ci_width']:.3f}"
                f"  {r['elapsed_s']:.1f}s"
            )

    return results


# =============================================================================
# Figure 1 — full posterior panels
# =============================================================================

def plot_posteriors(results):
    """
    Layout: rows = s_true values, cols = H_RESOLUTION values.
    Each panel: 2D joint posterior (s x h) with marginal h below.
    True values marked with crosshairs.
    """
    n_s   = len(S_VALUES)
    n_res = len(H_RESOLUTIONS)

    fig = plt.figure(figsize=(3.5 * n_res, 5.5 * n_s))
    outer_gs = gridspec.GridSpec(n_s, n_res, figure=fig,
                                 hspace=0.5, wspace=0.35)

    for row, s_true in enumerate(S_VALUES):
        for col, h_res in enumerate(H_RESOLUTIONS):
            r = results[s_true][col]

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=outer_gs[row, col],
                height_ratios=[3, 1], hspace=0.08,
            )
            ax_joint  = fig.add_subplot(inner_gs[0])
            ax_margin = fig.add_subplot(inner_gs[1])

            # 2D joint posterior
            jp = r["joint_posterior"]
            jp_norm = jp / max(jp.max(), 1e-12)

            im = ax_joint.imshow(
                jp_norm, origin="lower", aspect="auto",
                cmap="viridis", vmin=0, vmax=1,
                extent=[
                    r["h_range"].min(), r["h_range"].max(),
                    r["s_range"].min(), r["s_range"].max(),
                ],
            )

            # True value crosshairs
            ax_joint.axvline(H_TRUE,  color="red",   lw=1.2, ls="--", alpha=0.8)
            ax_joint.axhline(s_true,  color="red",   lw=1.2, ls="--", alpha=0.8)
            # MAP
            ax_joint.axvline(r["h_map"], color="white", lw=1.0, ls=":", alpha=0.9)
            ax_joint.axhline(r["s_map"], color="white", lw=1.0, ls=":", alpha=0.9)

            ax_joint.set_ylabel("s", fontsize=7)
            ax_joint.tick_params(labelsize=6)
            ax_joint.set_xticklabels([])

            title = (
                f"s={s_true}  H_RES={h_res}\n"
                f"MAP: s={r['s_map']:.2f} h={r['h_map']:.2f}"
            )
            ax_joint.set_title(title, fontsize=7, fontweight="bold")

            # Marginal h posterior
            h_post = r["h_posterior"]
            h_post_norm = h_post / max(h_post.max(), 1e-12)
            ax_margin.plot(r["h_range"], h_post_norm,
                           color="#2271B2", lw=1.5)
            ax_margin.fill_between(r["h_range"], h_post_norm,
                                   alpha=0.2, color="#2271B2")
            ax_margin.axvline(H_TRUE,    color="red",   lw=1.2, ls="--", alpha=0.8)
            ax_margin.axvline(r["h_map"], color="white" if jp_norm.max() > 0.5
                              else "black", lw=1.0, ls=":", alpha=0.9)

            # Shade 90% CI
            h_lo, h_hi = r["h_ci"]
            mask = (r["h_range"] >= h_lo) & (r["h_range"] <= h_hi)
            ax_margin.fill_between(r["h_range"][mask], h_post_norm[mask],
                                   alpha=0.35, color="orange",
                                   label=f"90% CI [{h_lo:.2f},{h_hi:.2f}]")

            ax_margin.set_xlabel("h", fontsize=7)
            ax_margin.set_ylabel("p(h)", fontsize=6)
            ax_margin.set_xlim(0, 1)
            ax_margin.set_ylim(0, 1.1)
            ax_margin.tick_params(labelsize=5)
            ax_margin.legend(fontsize=5, loc="upper left",
                             handlelength=1, framealpha=0.6)

    fig.suptitle(
        f"h posterior geometry vs H_RESOLUTION\n"
        f"vaf0={VAF0}  Δt={DELTA_T}y  h_true={H_TRUE}  depth={DEPTH}\n"
        f"Red dashed = true value  |  White dotted = MAP",
        fontsize=11, fontweight="bold",
    )

    fig.savefig(OUTPUT_POSTERIORS, dpi=150, bbox_inches="tight")
    print(f"\nPosterior figure saved: {OUTPUT_POSTERIORS}")
    plt.show()


# =============================================================================
# Figure 2 — MAP and CI width vs resolution
# =============================================================================

def plot_summary(results):
    """
    Two panels per s value:
      Left:  h_map vs H_RESOLUTION (with true h marked)
      Right: h CI width vs H_RESOLUTION
    """
    n_s  = len(S_VALUES)
    fig, axes = plt.subplots(n_s, 2, figsize=(9, 4 * n_s), squeeze=False)

    colours = {"h_map": "#2271B2", "ci_width": "#D44D3A"}

    for row, s_true in enumerate(S_VALUES):
        res_list   = H_RESOLUTIONS
        h_maps     = [r["h_map"]      for r in results[s_true]]
        h_ci_widths = [r["h_ci_width"] for r in results[s_true]]
        h_ci_lo    = [r["h_ci"][0]    for r in results[s_true]]
        h_ci_hi    = [r["h_ci"][1]    for r in results[s_true]]

        # Panel A — h MAP vs resolution
        ax = axes[row, 0]
        ax.plot(res_list, h_maps, "o-", color=colours["h_map"],
                lw=2, ms=6, label="h MAP")
        ax.fill_between(res_list, h_ci_lo, h_ci_hi,
                        alpha=0.2, color=colours["h_map"], label="90% CI")
        ax.axhline(H_TRUE, color="red", lw=1.5, ls="--", label=f"h true = {H_TRUE}")
        ax.set_xlabel("H_RESOLUTION", fontsize=9)
        ax.set_ylabel("Inferred h", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"s={s_true}: h MAP vs resolution", fontsize=9,
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # Panel B — CI width vs resolution
        ax2 = axes[row, 1]
        ax2.plot(res_list, h_ci_widths, "s-", color=colours["ci_width"],
                 lw=2, ms=6)
        ax2.axhline(0.15, color="grey", lw=1.2, ls=":",
                    label="CR bound threshold (0.15)")
        ax2.set_xlabel("H_RESOLUTION", fontsize=9)
        ax2.set_ylabel("h 90% CI width", fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.set_title(f"s={s_true}: CI width vs resolution", fontsize=9,
                      fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Resolution sensitivity: h MAP and CI width\n"
        f"vaf0={VAF0}  Δt={DELTA_T}y  h_true={H_TRUE}  depth={DEPTH}\n"
        "If MAP stabilises and CI narrows with resolution → grid artefact\n"
        "If flat regardless → genuine likelihood flatness",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTPUT_SUMMARY, dpi=150, bbox_inches="tight")
    print(f"Summary figure saved: {OUTPUT_SUMMARY}")
    plt.show()


# =============================================================================
# Figure 3 — VAF trajectories at multiple h values
# =============================================================================

def plot_vaf_trajectories(results):
    """
    For each s_true, show:
      - Observed noisy VAF data points
      - True VAF trajectory (h=H_TRUE)
      - Predicted trajectories at a range of h values [0, 0.25, 0.5, 0.75, 1.0]
        all using the same s_map and x0 anchored to observed vaf0

    This directly shows whether different h values produce visually
    distinguishable trajectories at vaf0=0.03 — if they don't, the
    likelihood flatness is explained by the data, not the inference.
    """
    H_PLOT_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
    colours       = plt.cm.plasma(np.linspace(0.1, 0.9, len(H_PLOT_VALUES)))
    t_dense       = np.linspace(0.0, DELTA_T, 200)

    fig, axes = plt.subplots(1, len(S_VALUES),
                             figsize=(6 * len(S_VALUES), 5),
                             squeeze=False)

    for col, s_true in enumerate(S_VALUES):
        ax = axes[0, col]

        # Regenerate the same data used in inference
        AO, DP, time_points, vaf_true = generate_data(
            s_true, H_TRUE, VAF0, DELTA_T, N_TPS, DEPTH, SEED
        )
        obs_vaf = (AO / np.maximum(DP, 1.0))[0]

        # Use MAP s from highest-resolution run
        r_best = results[s_true][-1]
        s_map  = r_best["s_map"]

        # Plot predicted trajectories at each h, anchored to observed vaf0
        for h_val, colour in zip(H_PLOT_VALUES, colours):
            x0_het   = x0_het_from_vaf0(VAF0, h_val)
            vaf_pred = simulate_vaf(t_dense, s_map, h_val, x0_het)
            v_max    = (1.0 + 2.0 * h_val) / (2.0 * (1.0 + h_val))
            ax.plot(t_dense, vaf_pred, color=colour, lw=1.8,
                    label=f"h={h_val:.2f}  (ceil={v_max:.3f})",
                    alpha=0.85)
            # Mark ceiling
            ax.axhline(v_max, color=colour, lw=0.6, ls=":", alpha=0.4)

        # True trajectory
        ax.plot(time_points, vaf_true, color="black", lw=2.0,
                ls="--", label=f"True (h={H_TRUE}, s={s_true})", zorder=5)

        # Observed data points with binomial error bars
        n_reads = DP[0]
        k_reads = AO[0]
        p_hat   = obs_vaf
        z       = 1.96
        denom_w = 1.0 + z**2 / n_reads
        centre  = (p_hat + z**2 / (2 * n_reads)) / denom_w
        half    = z * np.sqrt(p_hat * (1 - p_hat) / n_reads
                              + z**2 / (4 * n_reads**2)) / denom_w
        lo = np.maximum(centre - half, 1e-6)
        hi = centre + half

        ax.errorbar(time_points, obs_vaf,
                    yerr=[obs_vaf - lo, hi - obs_vaf],
                    fmt="o", color="black", ms=6, lw=1.2, capsize=3,
                    zorder=6, label="Observed VAF")

        ax.set_xlabel("Time (years)", fontsize=10)
        ax.set_ylabel("VAF", fontsize=10)
        ax.set_title(
            f"s_true={s_true}  (s_map={s_map:.2f})\n"
            f"vaf0={VAF0}  Δt={DELTA_T}y  h_true={H_TRUE}  depth={DEPTH}",
            fontsize=9, fontweight="bold",
        )
        ax.legend(fontsize=7.5, framealpha=0.8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate with rho at true h
        v_max_true = (1.0 + 2.0 * H_TRUE) / (2.0 * (1.0 + H_TRUE))
        rho        = vaf_true[-1] / v_max_true
        ax.text(0.97, 0.05,
                f"ρ = {rho:.3f}  (v_T/v_max)",
                transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig.suptitle(
        "VAF trajectories at different h values\n"
        "If curves overlap → h is unidentifiable from trajectory shape alone",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = "resolution_diagnostic_vaf.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"VAF trajectory figure saved: {out}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    results = run_all()
    plot_vaf_trajectories(results)
    plot_posteriors(results)
    plot_summary(results)