"""2-mutation model selection benchmark.

Tests whether the HMM correctly selects between two clonal structures:
  - Model A: both mutations in the SAME clone  [[0, 1]]
  - Model B: mutations in DIFFERENT clones     [[0], [1]]

Ground truth is always Model A (co-clonal). We measure how often each
pipeline picks the correct structure.

Grid: s x h combinations, each replicated N_REPS times with different seeds.

Outputs
-------
  model_selection_results.csv
  model_selection_summary.png
"""

import sys
import itertools
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
sys.path.append("../src")

# =============================================================================
# Configuration
# =============================================================================

S_TRUE_VALUES  = np.round(np.arange(0.3, 0.85, 0.2), 2)   # 0.3, 0.5, 0.7
H_TRUE_VALUES  = np.round(np.arange(0.0, 1.05, 0.5), 2)   # 0.0, 0.5, 1.0

N_REPS         = 3        # replicates per grid cell (different noise seeds)
N_W            = 1e5
DEPTH          = 2000
BASE_SEED      = 100

N_TIMEPOINTS   = 4
SPACING_YEARS  = 2.5
INITIAL_VAF    = 0.25

S_RESOLUTION   = 30
H_RESOLUTION   = 20
HMM_RESOLUTION = 400

# Second mutation has slightly different s to make the problem non-trivial
S_OFFSET       = 0.0   # same clone, so same s; VAF noise alone distinguishes

OUTPUT_CSV = "model_selection_results.csv"
OUTPUT_FIG = "model_selection_summary.png"

# =============================================================================
# Forward model (reuse from test_synth)
# =============================================================================

def simulate_clone(time_points, s, xhet0, h, N_w=N_W):
    t     = np.asarray(time_points, dtype=float)
    x_het = xhet0 * np.exp(s * (t - t[0]))
    x_hom = h * x_het
    x_tot = x_het + x_hom
    vaf   = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    return vaf


def compute_t_ceiling(s, h, xhet0, N_w=N_W, fraction=0.99):
    x_tot0       = xhet0 * (1 + h)
    x_tot_target = fraction * N_w / (1 - fraction)
    t_ceiling    = np.log(x_tot_target / x_tot0) / s
    t_ceiling    = max(t_ceiling, (N_TIMEPOINTS - 1) * SPACING_YEARS)
    return t_ceiling


def make_time_points(t_ceiling):
    t_end   = t_ceiling
    t_start = t_end - (N_TIMEPOINTS - 1) * SPACING_YEARS
    return np.linspace(t_start, t_end, N_TIMEPOINTS)


def generate_two_mutation_participant(s_true, h_true, depth, seed, N_w=N_W):
    """Generate synthetic data for 2 co-clonal mutations (ground truth: [[0,1]])."""
    rng   = np.random.default_rng(seed)
    xhet0 = max(2.0 * N_w * INITIAL_VAF / (1.0 + h_true), 10.0)
    tps   = make_time_points(compute_t_ceiling(s_true, h_true, xhet0, N_w))

    vaf_ceil = (1.0 + 2.0 * h_true) / (2.0 * (1.0 + h_true))

    # Both mutations share the same clone trajectory; independent binomial noise
    vaf_true = simulate_clone(tps, s_true, xhet0, h_true, N_w)
    vaf_true = np.clip(vaf_true, 1e-6, vaf_ceil - 1e-6)

    DP = np.full((2, len(tps)), depth, dtype=float)
    AO = np.vstack([
        rng.binomial(depth, vaf_true).astype(float),
        rng.binomial(depth, vaf_true).astype(float),
    ])

    return AO, DP, tps


# =============================================================================
# HMM model selection
# =============================================================================

def hmm_model_selection(AO, DP, time_points, s_resolution, h_resolution,
                         hmm_resolution, seed_offset=0):
    """
    Evaluate both clonal structures with the HMM and return the winning model
    along with log-probability ratio (log P(A) - log P(B)).

    Returns: winner ('A'=co-clonal, 'B'=independent), log_ratio, elapsed
    """
    import jax.numpy as jnp
    import jax.random as jrnd
    from KI_clonal_inference_5 import (
        compute_deterministic_size_mixed,
        jax_cs_hmm_ll_vec_mixed,
        compute_model_likelihood_2d,
    )

    t0    = time.perf_counter()
    key   = jrnd.PRNGKey(758493 + seed_offset)
    AO_j  = jnp.array(AO)
    DP_j  = jnp.array(DP)
    tps_j = jnp.array(time_points)

    s_vec = jnp.linspace(0.01, 1.0, s_resolution)
    h_vec = jnp.linspace(0.0,  1.0, h_resolution)
    s_arr = np.asarray(s_vec)
    h_arr = np.asarray(h_vec)

    probs = {}
    for label, cs in [("A", [[0, 1]]), ("B", [[0], [1]])]:
        k1, k2 = jrnd.split(key)
        det_size, total_cells, max_total, _ = compute_deterministic_size_mixed(
            cs, AO_j, DP_j, AO_j.shape[1]
        )
        output = jax_cs_hmm_ll_vec_mixed(
            s_vec, h_vec, AO_j, DP_j, tps_j,
            cs, det_size, total_cells, max_total,
            k1, resolution=hmm_resolution,
        )
        prob = compute_model_likelihood_2d(output, cs, s_arr, h_arr)
        probs[label] = max(prob, 1e-300)

    elapsed = time.perf_counter() - t0

    log_ratio = np.log(probs["A"]) - np.log(probs["B"])
    winner    = "A" if probs["A"] >= probs["B"] else "B"
    return winner, log_ratio, elapsed, probs["A"], probs["B"]


# =============================================================================
# Det model selection via VAF correlation
# =============================================================================

def det_model_selection(AO, DP, time_points):
    """
    Deterministic model selection: compare Pearson correlation of the two
    mutation VAF trajectories. High correlation → co-clonal (A).
    Uses the same threshold as find_valid_clonal_structures (0.5).
    """
    t0 = time.perf_counter()
    vaf = AO / np.maximum(DP, 1.0)   # (2, n_tps)

    r = np.corrcoef(vaf[0], vaf[1])[0, 1]
    # Also check correlation distance relative to time_points
    r0 = np.corrcoef(vaf[0], time_points)[0, 1]
    r1 = np.corrcoef(vaf[1], time_points)[0, 1]
    distance = abs(r0 - r1)

    # mirror the logic in compute_invalid_combinations / find_valid_clonal_structures
    winner = "B" if distance > 0.5 else "A"
    elapsed = time.perf_counter() - t0
    return winner, r, distance, elapsed


# =============================================================================
# Grid runner
# =============================================================================

def run_grid():
    grid    = list(itertools.product(S_TRUE_VALUES, H_TRUE_VALUES))
    records = []
    total   = len(grid) * N_REPS

    print(f"\nRunning {len(grid)} grid cells × {N_REPS} reps = {total} cases\n")

    case_idx = 0
    for s_true, h_true in grid:
        for rep in range(N_REPS):
            seed = BASE_SEED + case_idx
            case_idx += 1

            print(f"[{case_idx:3d}/{total}]  s={s_true:.2f}  h={h_true:.2f}  rep={rep}")

            try:
                AO, DP, tps = generate_two_mutation_participant(s_true, h_true, DEPTH, seed)
                obs_vaf = AO / np.maximum(DP, 1.0)
                vaf_strs = "  ".join(
                    f"t={t:.1f}y:[{v0:.3f},{v1:.3f}]"
                    for t, v0, v1 in zip(tps, obs_vaf[0], obs_vaf[1])
                )
                print(f"         VAFs: {vaf_strs}")
            except Exception as e:
                traceback.print_exc()
                for pipe in ["hmm", "det"]:
                    records.append(dict(pipeline=pipe, s_true=s_true, h_true=h_true,
                                        rep=rep, correct=np.nan, status=f"data error: {e}"))
                continue

            # HMM
            try:
                winner, log_ratio, elapsed, p_a, p_b = hmm_model_selection(
                    AO.T, DP.T, tps, S_RESOLUTION, H_RESOLUTION,
                    HMM_RESOLUTION, seed_offset=case_idx,
                )
                correct = int(winner == "A")
                print(f"  [hmm]  winner={winner} (correct={correct})  "
                      f"log_ratio={log_ratio:+.2f}  P(A)={p_a:.2e}  P(B)={p_b:.2e}  {elapsed:.1f}s")
                records.append(dict(pipeline="hmm", s_true=s_true, h_true=h_true,
                                    rep=rep, correct=correct, log_ratio=log_ratio,
                                    p_a=p_a, p_b=p_b, elapsed_s=elapsed, status="ok"))
            except Exception as e:
                traceback.print_exc()
                records.append(dict(pipeline="hmm", s_true=s_true, h_true=h_true,
                                    rep=rep, correct=np.nan, status=f"error: {e}"))

            # Det (correlation-based)
            try:
                winner, r, dist, elapsed = det_model_selection(AO.T, DP.T, tps)
                correct = int(winner == "A")
                print(f"  [det]  winner={winner} (correct={correct})  "
                      f"r={r:.3f}  dist={dist:.3f}  {elapsed*1000:.1f}ms")
                records.append(dict(pipeline="det", s_true=s_true, h_true=h_true,
                                    rep=rep, correct=correct, pearson_r=r,
                                    vaf_dist=dist, elapsed_s=elapsed, status="ok"))
            except Exception as e:
                traceback.print_exc()
                records.append(dict(pipeline="det", s_true=s_true, h_true=h_true,
                                    rep=rep, correct=np.nan, status=f"error: {e}"))

        print()

    return pd.DataFrame(records)


# =============================================================================
# Summary + plot
# =============================================================================

def print_summary(df):
    print("\n" + "=" * 60)
    print("MODEL SELECTION SUMMARY  (ground truth: co-clonal [[0,1]])")
    print("=" * 60)
    for pipe in ["hmm", "det"]:
        ok = df[(df["pipeline"] == pipe) & (df["status"] == "ok")]
        acc = ok["correct"].mean() * 100
        print(f"\n  {pipe.upper()}")
        print(f"    Accuracy: {acc:.1f}%  ({int(ok['correct'].sum())}/{len(ok)})")
        print(f"    Runtime:  {ok['elapsed_s'].mean()*1000:.1f} ms/case avg")
    print("=" * 60)


def plot_summary(df, output_path):
    pipelines = [p for p in ["det", "hmm"] if p in df["pipeline"].values]
    s_vals = sorted(df["s_true"].unique())
    h_vals = sorted(df["h_true"].unique())

    fig, axes = plt.subplots(1, len(pipelines), figsize=(7 * len(pipelines), 5), squeeze=False)

    labels = {"det": "Det (correlation)", "hmm": "HMM"}

    for col, pipe in enumerate(pipelines):
        ax = axes[0, col]
        ok = df[(df["pipeline"] == pipe) & (df["status"] == "ok")]
        acc_grid = ok.groupby(["s_true", "h_true"])["correct"].mean().reset_index()
        mat = acc_grid.pivot(index="s_true", columns="h_true", values="correct")

        im = ax.imshow(mat.values, origin="lower", vmin=0, vmax=1, cmap="RdYlGn",
                       aspect="auto",
                       extent=[mat.columns.min() - 0.15, mat.columns.max() + 0.15,
                               mat.index.min()   - 0.1,  mat.index.max()   + 0.1])

        for s in mat.index:
            for h in mat.columns:
                v = mat.loc[s, h]
                if not np.isnan(v):
                    ax.text(h, s, f"{v:.2f}", ha="center", va="center", fontsize=9)

        ax.set_xlabel("True h")
        ax.set_ylabel("True s")
        overall = ok["correct"].mean() * 100
        ax.set_title(f"[{labels[pipe]}]  accuracy={overall:.0f}%", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8, label="P(correct)")

    fig.suptitle(
        "Model selection accuracy\n(ground truth: co-clonal; chance = 50%)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    df = run_grid()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print_summary(df)
    plot_summary(df, OUTPUT_FIG)