"""
plot_clonal_fits.py
-------------------
For each fitted participant, produces a three-panel figure:

  Left   : VAF trajectories with projected fit lines, coloured by clone
  Centre : Marginal posterior over fitness s, coloured by clone
  Right  : Marginal posterior over zygosity h, coloured by clone

Usage
-----
    python plot_clonal_fits.py \
        --input  ../exports/MDS/MDS_cohort_fitted.pk \
        --outdir ../exports/MDS/figures/

One PDF per participant is written to outdir.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette — one colour per clone index, up to 8 clones
# ---------------------------------------------------------------------------

CLONE_COLOURS = [
    "#2271B2",   # blue
    "#F0A500",   # amber
    "#3DAA6A",   # green
    "#D44D3A",   # red
    "#8E5FB9",   # purple
    "#E07B54",   # terracotta
    "#4ABFBF",   # teal
    "#A0522D",   # sienna
]


def _clone_colour(clone_index: int) -> str:
    return CLONE_COLOURS[int(clone_index) % len(CLONE_COLOURS)]


# ---------------------------------------------------------------------------
# Single-participant plot
# ---------------------------------------------------------------------------

def plot_participant(part, participant_id: str, ax_vaf, ax_s, ax_h):
    """Fill three pre-created axes for one participant."""

    if "optimal_model" not in part.uns:
        ax_vaf.set_visible(False)
        ax_s.set_visible(False)
        ax_h.set_visible(False)
        ax_vaf.set_title(f"{participant_id}\n(no model)", fontsize=8)
        return

    model       = part.uns["optimal_model"]
    cs          = model["clonal_structure"]
    joint_inf   = model["joint_inference"]
    time_points = np.asarray(part.var.time_points, dtype=float)

    AO  = part.layers["AO"].astype(float)   # (n_mutations, n_timepoints)
    DP  = part.layers["DP"].astype(float)
    vaf = AO / np.maximum(DP, 1.0)

    mut_names    = list(part.obs.index)
    clonal_index = np.asarray(part.obs["clonal_index"], dtype=int)

    # ------------------------------------------------------------------ #
    # Panel 1 — VAF trajectories
    # ------------------------------------------------------------------ #

    ax_vaf.set_yscale("log")

    for mut_idx, mut_name in enumerate(mut_names):
        ci    = clonal_index[mut_idx]
        color = _clone_colour(ci)

        obs_vaf = vaf[mut_idx]
        obs_dp  = DP[mut_idx]
        valid   = obs_dp > 0

        # Error bars: Wilson binomial 95 % CI
        n_reads = obs_dp[valid]
        k_reads = AO[mut_idx][valid]
        p_hat   = obs_vaf[valid]
        z       = 1.96
        denom   = 1.0 + z**2 / n_reads
        centre  = (p_hat + z**2 / (2 * n_reads)) / denom
        half    = z * np.sqrt(p_hat * (1 - p_hat) / n_reads + z**2 / (4 * n_reads**2)) / denom
        lo      = np.maximum(centre - half, 1e-5)
        hi      = centre + half

        ax_vaf.errorbar(
            time_points[valid],
            obs_vaf[valid],
            yerr=[obs_vaf[valid] - lo, hi - obs_vaf[valid]],
            fmt="o",
            color=color,
            markersize=4,
            linewidth=0.8,
            capsize=2,
            zorder=3,
            label=f"Clone {ci}: {mut_name}" if mut_idx == cs[ci][0] else f"  {mut_name}",
        )

    # Projected fit lines — one per clone, using leading mutation trajectory
    for clone_idx, clone_mutations in enumerate(cs):
        result = joint_inf[clone_idx]
        color  = _clone_colour(clone_idx)

        t_fit   = result["time_points_valid"]
        vaf_fit = result["projected_vaf_valid"]

        if len(t_fit) >= 2:
            # Extend line slightly beyond last observed point
            t_ext   = np.linspace(t_fit[0], t_fit[-1], 200)
            s_map   = result["s_map"]
            h_map   = result["h_map"]
            x0      = result["initial_mutant_cells"]
            N_w     = 1e5
            x_ext   = x0 * np.exp(s_map * (t_ext - t_fit[0]))
            vaf_ext = x_ext * (1.0 + 2.0 * h_map) / (2.0 * (N_w + x_ext))
            ax_vaf.plot(t_ext, vaf_ext, color=color, linewidth=1.5,
                        alpha=0.85, zorder=2)

    ax_vaf.set_xlabel("Time (years)", fontsize=8)
    ax_vaf.set_ylabel("VAF", fontsize=8)
    ax_vaf.set_title(participant_id, fontsize=9, fontweight="bold")
    ax_vaf.tick_params(labelsize=7)
    ax_vaf.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Legend — only one entry per clone (leading mutation name)
    handles, labels = ax_vaf.get_legend_handles_labels()
    # Keep only leading-mutation entries (those starting with "Clone")
    filtered = [(h, l) for h, l in zip(handles, labels) if l.startswith("Clone")]
    if filtered:
        hs, ls = zip(*filtered)
        ax_vaf.legend(hs, ls, fontsize=6, framealpha=0.7,
                      loc="upper left", handlelength=1.2)

    # ------------------------------------------------------------------ #
    # Panel 2 — Fitness posterior
    # ------------------------------------------------------------------ #

    for clone_idx, result in enumerate(joint_inf):
        color      = _clone_colour(clone_idx)
        s_range    = result["s_range"]
        s_post     = result["s_posterior"]
        s_post_norm = s_post / np.maximum(s_post.max(), 1e-12)

        ax_s.plot(s_range, s_post_norm, color=color, linewidth=1.5)
        ax_s.fill_between(s_range, s_post_norm, alpha=0.15, color=color)

        # MAP marker
        s_map = result["s_map"]
        ax_s.axvline(s_map, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    ax_s.set_xlabel("Fitness  $s$", fontsize=8)
    ax_s.set_ylabel("Normalised probability", fontsize=8)
    ax_s.set_xlim(0, None)
    ax_s.set_ylim(0, 1.05)
    ax_s.tick_params(labelsize=7)
    ax_s.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------ #
    # Panel 3 — Zygosity posterior
    # ------------------------------------------------------------------ #

    for clone_idx, result in enumerate(joint_inf):
        color      = _clone_colour(clone_idx)
        h_range    = result["h_range"]
        h_post     = result["h_posterior"]
        h_post_norm = h_post / np.maximum(h_post.max(), 1e-12)

        ax_h.plot(h_range, h_post_norm, color=color, linewidth=1.5)
        ax_h.fill_between(h_range, h_post_norm, alpha=0.15, color=color)

        h_map = result["h_map"]
        ax_h.axvline(h_map, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    ax_h.set_xlabel("Zygosity  $h$", fontsize=8)
    ax_h.set_ylabel("Normalised probability", fontsize=8)
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1.05)
    ax_h.tick_params(labelsize=7)
    ax_h.spines[["top", "right"]].set_visible(False)

    for ax in (ax_vaf, ax_s, ax_h):
        ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Cohort loop
# ---------------------------------------------------------------------------

def plot_cohort(cohort: list, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    for part, pid in cohort:
        fig, axes = plt.subplots(
            1, 3,
            figsize=(11, 3.5),
            gridspec_kw={"width_ratios": [2, 1, 1], "wspace": 0.35},
        )

        plot_participant(part, pid, *axes)

        fig.tight_layout()
        out_path = outdir / f"{pid}_fit.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_cohort(pk_path: str):
    with open(pk_path, "rb") as f:
        data = pickle.load(f)

    # Accept either a list of (part, id) tuples or a dict {id: part}
    if isinstance(data, dict):
        return [(v, k) for k, v in data.items()]
    elif isinstance(data, list):
        # Try to extract participant id from part.obs attributes
        cohort = []
        for item in data:
            if isinstance(item, tuple):
                cohort.append(item)
            else:
                # Single AnnData-like object — try common id fields
                pid = (
                    getattr(item, "uns", {}).get("participant_id")
                    or getattr(item, "uns", {}).get("id")
                    or str(id(item))
                )
                cohort.append((item, pid))
        return cohort
    else:
        raise ValueError(f"Unexpected pickle format: {type(data)}")


def main():
    parser = argparse.ArgumentParser(description="Plot clonal VAF fits and posteriors.")
    parser.add_argument("--input",  required=True, help="Path to fitted cohort pickle")
    parser.add_argument("--outdir", required=True, help="Directory for output PDFs")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    cohort = load_cohort(args.input)
    print(f"  {len(cohort)} participants loaded")

    plot_cohort(cohort, Path(args.outdir))
    print("Done.")


if __name__ == "__main__":
    main()