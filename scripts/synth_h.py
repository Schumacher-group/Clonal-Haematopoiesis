"""
Direct lightweight test script for clone-level LOH h inference.

This bypasses refine_optimal_model_posterior_vec_loh and calls the lower-level
LOH likelihood directly, so we can control h support, resolution, and avoid
zero posterior issues from internal defaults.

Assumes src.KI_clonal_inference_7 exposes:
    compute_deterministic_size_loh
    jax_cs_hmm_ll_vec_loh
    compute_clone_h_posterior

If jax_cs_hmm_ll_vec_loh does not support `resolution`, this script falls back
to calling it without that argument.
"""

import sys
sys.path.append("..")

import os
import inspect
import traceback

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad

import jax.numpy as jnp

from src.KI_clonal_inference_7 import (
    compute_deterministic_size_loh,
    jax_cs_hmm_ll_vec_loh,
    compute_clone_h_posterior,
)


# ==============================================================================
# Configuration
# ==============================================================================

OUTPUT_DIR = "../exports/test_h_inference_direct/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

S_RESOLUTION = 50
H_RESOLUTION = 31
HIDDEN_RESOLUTION = 120

MIN_S = 0.01
MAX_S = 1.0

N_W = 1e5


# ==============================================================================
# Synthetic helpers
# ==============================================================================

def clone_size_to_vaf_mixed_loh(x, h, N_w=N_W):
    return (1.0 + h) * x / (2.0 * (N_w + x))


def vaf_to_clone_size_mixed_loh(vaf, h, N_w=N_W):
    denom = 1.0 + h - 2.0 * vaf

    if denom <= 0:
        raise ValueError(
            f"start_vaf={vaf:.3f} impossible for h={h:.3f}"
        )

    return 2.0 * vaf * N_w / denom


def generate_synthetic_patient(
    true_s,
    true_h,
    patient_id,
    n_timepoints=5,
    n_mutations=3,
    DP_mean=3000,
    seed=42,
    start_vaf=0.1,
    sampling_duration=4.0,
    max_clone_to_wildtype_ratio=5.0,
):
    rng = np.random.default_rng(seed)

    x_start = vaf_to_clone_size_mixed_loh(start_vaf, true_h)
    t_start = np.log(x_start) / true_s

    abs_t = np.linspace(t_start, t_start + sampling_duration, n_timepoints)
    rel_t = abs_t - abs_t[0]

    AO = np.zeros((n_mutations, n_timepoints), dtype=int)
    DP = np.zeros((n_mutations, n_timepoints), dtype=int)

    true_x = np.zeros(n_timepoints)
    true_vaf = np.zeros(n_timepoints)

    print(f"\n{'=' * 80}")
    print(f"GENERATING {patient_id}")
    print(f"{'=' * 80}")
    print(f"true_s={true_s:.3f}, true_h={true_h:.3f}")

    print(f"\n{'rel_t':<8} {'x':<14} {'true_vaf':<10}")
    print("-" * 40)

    for t_idx, t in enumerate(abs_t):
        x_unclipped = np.exp(true_s * t)
        x_cap = N_W * max_clone_to_wildtype_ratio
        x = min(x_unclipped, x_cap)

        v = clone_size_to_vaf_mixed_loh(x, true_h)
        v = float(np.clip(v, 1e-6, 1.0 - 1e-6))

        true_x[t_idx] = x
        true_vaf[t_idx] = v

        print(f"{rel_t[t_idx]:<8.2f} {x:<14.1f} {v:<10.4f}")

        for m in range(n_mutations):
            dp = int(max(1, rng.poisson(DP_mean)))
            ao = int(rng.binomial(dp, v))

            AO[m, t_idx] = ao
            DP[m, t_idx] = dp

    part = ad.AnnData(
        X=AO,
        layers={
            "AO": AO,
            "DP": DP,
        },
    )

    mut_names = [f"{patient_id}_mut_{i + 1}" for i in range(n_mutations)]
    part.obs_names = mut_names
    part.obs["p_key"] = mut_names

    part.var_names = [f"tp_{i}" for i in range(n_timepoints)]
    part.var["time_points"] = rel_t

    part.uns["participant_id"] = patient_id

    obs_vaf = AO / np.maximum(DP, 1)

    print("\nObserved VAF:")
    for m in range(n_mutations):
        print("  " + " → ".join([f"{v:.3f}" for v in obs_vaf[m]]))

    print("\nMax observed VAF per mutation:")
    print(np.array2string(obs_vaf.max(axis=1), precision=3))

    max_true_vaf = float(true_vaf.max())
    theoretical_h_min = max(0.0, 2.0 * max_true_vaf - 1.0)

    print(f"\nmax true VAF={max_true_vaf:.3f}")
    print(f"theoretical h_min={theoretical_h_min:.3f}")
    print(f"h identifiable={max_true_vaf > 0.5}")

    truth = {
        "patient_id": patient_id,
        "true_s": true_s,
        "true_h": true_h,
        "time_points": rel_t,
        "true_x": true_x,
        "true_vaf": true_vaf,
        "max_true_vaf": max_true_vaf,
        "theoretical_h_min": theoretical_h_min,
        "h_identifiable": max_true_vaf > 0.5,
    }

    return part, truth


# ==============================================================================
# Direct inference helpers
# ==============================================================================

def weighted_quantile_from_grid(grid, weights, quantiles=(0.05, 0.95)):
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    if weights.sum() <= 0:
        return np.array([np.nan for _ in quantiles])

    weights = weights / weights.sum()
    order = np.argsort(grid)

    grid_sorted = grid[order]
    weights_sorted = weights[order]
    cdf = np.cumsum(weights_sorted)

    return np.array([
        grid_sorted[np.searchsorted(cdf, q, side="left")]
        for q in quantiles
    ])


def make_direct_h_grid_and_prior(part, cs, h_resolution=31, margin=0.03):
    """
    Build clone-level h grid and prior.

    For one clone, lower bound is max observed VAF:
        h_min = max(0, 2*Vmax - 1 + margin)

    But we keep the global h grid [0,1] and zero prior below h_min.
    """
    AO = np.asarray(part.layers["AO"].T)
    DP = np.asarray(part.layers["DP"].T)
    vaf = AO / np.maximum(DP, 1)

    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    h_priors = []

    for clone in cs:
        clone_vmax = np.max(vaf[:, clone])
        raw_min = max(0.0, 2.0 * clone_vmax - 1.0)

        if raw_min > 0:
            h_min = min(0.98, raw_min + margin)
        else:
            h_min = 0.0

        prior = np.where(np.asarray(h_vec) >= h_min, 1.0, 0.0)

        if prior.sum() == 0:
            prior[-1] = 1.0

        prior = prior / prior.sum()
        h_priors.append(prior)

        print(f"  clone {clone}: observed vmax={clone_vmax:.3f}, h_min={h_min:.3f}")

    h_prior_by_clone = jnp.array(np.stack(h_priors))

    return h_vec, h_prior_by_clone


def call_jax_cs_hmm_ll_vec_loh(
    s_vec,
    AO,
    DP,
    time_points,
    cs,
    deterministic_size,
    total_cells,
    h_vec,
    h_prior_by_clone,
    clonal_map,
):
    sig = inspect.signature(jax_cs_hmm_ll_vec_loh)

    kwargs = {}

    if "resolution" in sig.parameters:
        kwargs["resolution"] = HIDDEN_RESOLUTION

    print("\nCalling jax_cs_hmm_ll_vec_loh with:")
    print(f"  s_resolution={len(s_vec)}")
    print(f"  h_resolution={len(h_vec)}")
    if "resolution" in kwargs:
        print(f"  hidden resolution={kwargs['resolution']}")
    else:
        print("  hidden resolution: function does not expose argument")

    return jax_cs_hmm_ll_vec_loh(
        s_vec,
        AO,
        DP,
        time_points,
        cs,
        deterministic_size,
        total_cells,
        h_vec,
        h_prior_by_clone,
        clonal_map,
        **kwargs,
    )


def direct_infer_one_clone(part):
    """
    Direct clone-level s,h inference for a forced one-clone structure.
    """
    n_mutations = part.n_obs
    cs = [list(range(n_mutations))]

    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var["time_points"].values)

    s_vec = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)

    print("\nForced clonal structure:")
    print(f"  {cs}")

    h_vec, h_prior_by_clone = make_direct_h_grid_and_prior(
        part,
        cs,
        h_resolution=H_RESOLUTION,
        margin=0.03,
    )

    # Deterministic approximation uses lower-bound h for each clone.
    h_for_deterministic = jnp.array([
        h_vec[jnp.argmax(h_prior_by_clone[c] > 0)]
        for c in range(len(cs))
    ])

    print("h_for_deterministic:")
    print(np.asarray(h_for_deterministic))

    (
        deterministic_size,
        total_cells,
        clonal_map,
        leading_mutations,
        clone_sizes,
    ) = compute_deterministic_size_loh(
        cs,
        AO,
        DP,
        AO.shape[1],
        h_for_deterministic,
    )

    output, mutation_likelihood_h = call_jax_cs_hmm_ll_vec_loh(
        s_vec,
        AO,
        DP,
        time_points,
        cs,
        deterministic_size,
        total_cells,
        h_vec,
        h_prior_by_clone,
        clonal_map,
    )

    output_np = np.asarray(output)
    output_np = np.nan_to_num(output_np, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nOutput likelihood diagnostics:")
    print(f"  output shape: {output_np.shape}")
    print(f"  output sum:   {output_np.sum():.3e}")
    print(f"  output max:   {output_np.max():.3e}")
    print(f"  nonzero:      {np.count_nonzero(output_np)} / {output_np.size}")

    if output_np[:, 0].sum() <= 0:
        raise RuntimeError("Zero fitness posterior from direct likelihood")

    # h posterior
    h_post = compute_clone_h_posterior(
        mutation_likelihood_h,
        cs,
        s_vec,
        h_prior_by_clone,
    )

    h_post_np = np.asarray(h_post[0])
    h_post_np = np.nan_to_num(h_post_np, nan=0.0, posinf=0.0, neginf=0.0)

    print("\nh posterior diagnostics:")
    print(f"  sum:      {h_post_np.sum():.3e}")
    print(f"  max:      {h_post_np.max():.3e}")
    print(f"  nonzero:  {np.count_nonzero(h_post_np)} / {h_post_np.size}")

    if h_post_np.sum() <= 0:
        raise RuntimeError("Zero h posterior from direct likelihood")

    s_post = output_np[:, 0]
    s_post = s_post / s_post.sum()

    h_post_np = h_post_np / h_post_np.sum()

    s_range = np.asarray(s_vec)
    h_range = np.asarray(h_vec)

    s_map = float(s_range[np.argmax(s_post)])
    h_map = float(h_range[np.argmax(h_post_np)])

    s_ci = weighted_quantile_from_grid(s_range, s_post)
    h_ci = weighted_quantile_from_grid(h_range, h_post_np)

    return {
        "cs": cs,
        "s_range": s_range,
        "h_range": h_range,
        "s_posterior": s_post,
        "h_posterior": h_post_np,
        "s_map": s_map,
        "h_map": h_map,
        "s_ci": s_ci,
        "h_ci": h_ci,
        "output": output_np,
    }


# ==============================================================================
# Test runner
# ==============================================================================

def run_tests(save_plots=True):
    cases = [
        {
            "patient_id": "SYN_HET_WEAK",
            "true_s": 0.5,
            "true_h": 0.0,
            "seed": 42,
            "sampling_duration": 4.0,
            "start_vaf": 0.10,
        },
        {
            "patient_id": "SYN_HOM_STRONG",
            "true_s": 0.5,
            "true_h": 1.0,
            "seed": 43,
            "sampling_duration": 5.8,
            "start_vaf": 0.10,
        },
        {
            "patient_id": "SYN_MIX_STRONG",
            "true_s": 0.5,
            "true_h": 0.5,
            "seed": 44,
            "sampling_duration": 5.5,
            "start_vaf": 0.10,
        },
    ]

    results = []

    for case in cases:
        try:
            part, truth = generate_synthetic_patient(
                true_s=case["true_s"],
                true_h=case["true_h"],
                patient_id=case["patient_id"],
                seed=case["seed"],
                sampling_duration=case["sampling_duration"],
                start_vaf=case["start_vaf"],
                max_clone_to_wildtype_ratio=5.0,
            )

            inferred = direct_infer_one_clone(part)

            s_err = abs(inferred["s_map"] - truth["true_s"])
            h_err = abs(inferred["h_map"] - truth["true_h"])

            print(f"\n{'=' * 80}")
            print(f"RESULTS: {case['patient_id']}")
            print(f"{'=' * 80}")
            print(f"s true={truth['true_s']:.3f}, MAP={inferred['s_map']:.3f}, err={s_err:.3f}, CI={inferred['s_ci']}")
            print(f"h true={truth['true_h']:.3f}, MAP={inferred['h_map']:.3f}, err={h_err:.3f}, CI={inferred['h_ci']}")
            print(f"max true VAF={truth['max_true_vaf']:.3f}, h_identifiable={truth['h_identifiable']}")

            results.append({
                "success": True,
                "patient_id": case["patient_id"],
                "truth": truth,
                "inferred": inferred,
                "s_error": s_err,
                "h_error": h_err,
            })

        except Exception as e:
            print(f"\nFAILED: {case['patient_id']}")
            print(e)
            traceback.print_exc()

            results.append({
                "success": False,
                "patient_id": case["patient_id"],
                "error": str(e),
            })

    print_summary(results)

    if save_plots:
        plot_results(results)

    return results


def print_summary(results):
    print(f"\n{'#' * 80}")
    print("SUMMARY")
    print(f"{'#' * 80}")

    successful = [r for r in results if r["success"]]

    if not successful:
        print("No successful tests")
        return

    print(
        f"{'patient':<18} "
        f"{'true_s':<8} {'map_s':<8} {'s_err':<8} "
        f"{'true_h':<8} {'map_h':<8} {'h_err':<8} "
        f"{'vafmax':<8}"
    )
    print("-" * 95)

    for r in successful:
        truth = r["truth"]
        inf = r["inferred"]

        print(
            f"{r['patient_id']:<18} "
            f"{truth['true_s']:<8.3f} "
            f"{inf['s_map']:<8.3f} "
            f"{r['s_error']:<8.3f} "
            f"{truth['true_h']:<8.3f} "
            f"{inf['h_map']:<8.3f} "
            f"{r['h_error']:<8.3f} "
            f"{truth['max_true_vaf']:<8.3f}"
        )


def plot_results(results):
    successful = [r for r in results if r["success"]]

    if not successful:
        return

    fig, axes = plt.subplots(
        len(successful),
        3,
        figsize=(15, 4.5 * len(successful)),
    )

    if len(successful) == 1:
        axes = axes[np.newaxis, :]

    for i, r in enumerate(successful):
        truth = r["truth"]
        inf = r["inferred"]

        # True VAF trajectory
        ax = axes[i, 0]
        ax.plot(truth["time_points"], truth["true_vaf"], "ko-", label="true VAF")
        ax.axhline(0.5, color="grey", linestyle=":", label="0.5")
        ax.set_title(f"{r['patient_id']} VAF")
        ax.set_xlabel("time")
        ax.set_ylabel("VAF")
        ax.grid(alpha=0.3)
        ax.legend()

        # s posterior
        ax = axes[i, 1]
        ax.plot(inf["s_range"], inf["s_posterior"])
        ax.axvline(truth["true_s"], color="green", linestyle="--", label="true")
        ax.axvline(inf["s_map"], color="red", linestyle="-", label="MAP")
        ax.axvspan(inf["s_ci"][0], inf["s_ci"][1], color="red", alpha=0.2)
        ax.set_title("s posterior")
        ax.set_xlabel("s")
        ax.grid(alpha=0.3)
        ax.legend()

        # h posterior
        ax = axes[i, 2]
        ax.plot(inf["h_range"], inf["h_posterior"])
        ax.axvline(truth["true_h"], color="green", linestyle="--", label="true")
        ax.axvline(inf["h_map"], color="red", linestyle="-", label="MAP")
        ax.axvline(truth["theoretical_h_min"], color="purple", linestyle=":", label="h_min")
        ax.axvspan(inf["h_ci"][0], inf["h_ci"][1], color="red", alpha=0.2)
        ax.set_title("h posterior")
        ax.set_xlabel("h")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "direct_h_inference_test.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {path}")


if __name__ == "__main__":
    results = run_tests(save_plots=True)
