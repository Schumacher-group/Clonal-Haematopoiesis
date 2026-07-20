#!/usr/bin/env python3
"""Plot beta posteriors from het_inference.py and per-participant VAF distributions.

This script does two things:
1. Plot the beta distributions implied by a configurable set of AO/DP pairs,
   matching the sampling parameterization used in src/het_inference.py:
   Beta(AO + 1, DP - AO + 1).
2. For every participant in a processed cohort pickle, plot the VAF posterior
   distribution for every mutation at every timepoint.

Expected cohort format:
- list of AnnData objects
- part.layers['AO'] with shape (n_mutations, n_timepoints)
- part.layers['DP'] with shape (n_mutations, n_timepoints)
- part.var['time_points']
- part.uns['participant_id']
"""

from __future__ import annotations

import argparse
import math
import pickle as pk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import beta


DEFAULT_INPUT = Path("../exports/MDS/MDS_cohort_processed.pk")
DEFAULT_OUTPUT_DIR = Path("../exports/MDS/beta_vaf_distributions")
DEFAULT_REFERENCE_PAIRS = (
    (1, 10),
    (2, 10),
    (5, 20),
    (10, 50),
    (25, 100),
    (50, 200),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot beta/VAF distributions from AO/DP values and participant data."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT,
        help="Processed cohort pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--x-grid-size",
        type=int,
        default=1000,
        help="Number of points used to evaluate each beta density.",
    )
    parser.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Optional cap on number of participants to plot.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        nargs="*",
        default=None,
        help="Optional AO/DP pairs like 1/10 5/20 25/100.",
    )
    return parser.parse_args()


def parse_pairs(pair_args: list[str] | None) -> list[tuple[int, int]]:
    if not pair_args:
        return list(DEFAULT_REFERENCE_PAIRS)

    pairs: list[tuple[int, int]] = []
    for item in pair_args:
        try:
            ao_str, dp_str = item.split("/")
            ao = int(ao_str)
            dp = int(dp_str)
        except ValueError as exc:
            raise ValueError(f"Invalid pair '{item}'. Use AO/DP format, e.g. 5/20.") from exc

        if ao < 0 or dp <= 0 or ao > dp:
            raise ValueError(f"Invalid AO/DP pair '{item}'. Require 0 <= AO <= DP and DP > 0.")
        pairs.append((ao, dp))
    return pairs


def mutation_label(part, mutation_index: int) -> str:
    obs = part.obs.iloc[mutation_index]
    for column in ("p_key", "key", "PreferredSymbol"):
        if column in part.obs.columns:
            return str(obs[column])
    return str(part.obs.index[mutation_index])


def beta_density_from_counts(ao: float | int, dp: float | int, x_grid: np.ndarray) -> np.ndarray:
    """Return the beta posterior density used in het_inference.py.

    het_inference.py samples hidden VAFs with:
    Beta(AO + 1, DP - AO + 1)
    """
    ao_int = int(ao)
    dp_int = int(dp)
    alpha = ao_int + 1
    beta_param = dp_int - ao_int + 1
    return beta.pdf(x_grid, alpha, beta_param)


def load_cohort(input_file: Path):
    with input_file.open("rb") as handle:
        return pk.load(handle)


def plot_reference_beta_distributions(pairs: list[tuple[int, int]], output_dir: Path, x_grid_size: int) -> Path:
    output_path = output_dir / "reference_beta_distributions.png"
    x_grid = np.linspace(1e-6, 1.0 - 1e-6, x_grid_size)

    plt.figure(figsize=(10, 6))
    for ao, dp in pairs:
        density = beta_density_from_counts(ao, dp, x_grid)
        mean_vaf = (ao + 1) / (dp + 2)
        plt.plot(x_grid, density, label=f"AO/DP={ao}/{dp} (mean={mean_vaf:.3f})")

    plt.xlabel("VAF")
    plt.ylabel("Density")
    plt.title("Beta posterior distributions used in het_inference.py")
    plt.legend(loc="best", fontsize="small")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_participant_vaf_distributions(part, output_dir: Path, x_grid_size: int) -> Path:
    participant_id = str(part.uns.get("participant_id", "unknown_participant"))
    output_path = output_dir / f"{participant_id}_vaf_distributions.png"

    ao = np.asarray(part.layers["AO"], dtype=float)
    dp = np.asarray(part.layers["DP"], dtype=float)
    time_points = np.asarray(part.var["time_points"], dtype=float)
    n_mutations, n_timepoints = ao.shape

    x_grid = np.linspace(1e-6, 1.0 - 1e-6, x_grid_size)
    ncols = min(3, n_timepoints)
    nrows = math.ceil(n_timepoints / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    palette = sns.color_palette("tab10", n_colors=max(n_mutations, 1))

    for time_index, tp in enumerate(time_points):
        ax = axes_flat[time_index]
        for mutation_index in range(n_mutations):
            dp_value = dp[mutation_index, time_index]
            ao_value = ao[mutation_index, time_index]
            if dp_value <= 0 or ao_value < 0 or ao_value > dp_value:
                continue

            density = beta_density_from_counts(ao_value, dp_value, x_grid)
            observed_vaf = ao_value / dp_value if dp_value > 0 else np.nan
            label = f"{mutation_label(part, mutation_index)} ({int(ao_value)}/{int(dp_value)}, obs={observed_vaf:.3f})"
            ax.plot(x_grid, density, color=palette[mutation_index % len(palette)], linewidth=2, label=label)

        ax.set_title(f"Timepoint {time_index + 1} (time={tp:g})")
        ax.set_xlabel("VAF")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="x-small")

    for empty_ax in axes_flat[n_timepoints:]:
        empty_ax.axis("off")

    fig.suptitle(f"VAF posterior distributions: {participant_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_dir = output_dir / "participants"
    participant_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(args.pairs)
    cohort = load_cohort(args.input_file)
    if args.max_participants is not None:
        cohort = cohort[: args.max_participants]

    reference_plot = plot_reference_beta_distributions(pairs, output_dir, args.x_grid_size)
    print(f"Saved reference beta plot -> {reference_plot}")

    for index, part in enumerate(cohort, start=1):
        participant_id = str(part.uns.get("participant_id", f"participant_{index}"))
        print(f"[{index}/{len(cohort)}] Plotting {participant_id}")
        plot_path = plot_participant_vaf_distributions(part, participant_dir, args.x_grid_size)
        print(f"  Saved -> {plot_path}")


if __name__ == "__main__":
    main()
