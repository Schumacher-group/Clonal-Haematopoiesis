"""
Synthetic VAF benchmark — mixed vs het-only vs hom-only pipeline comparison.
Poster version: fitness (main) + h recovery (lower panel), shared x-axis.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S_TRUE           = 0.5
H_VALUES         = np.round(np.arange(0.0, 1.05, 0.1), 2)
N_W              = 1e5
DEPTH            = 2000
N_REPS           = 20
BASE_SEED        = 42
TIME_POINTS      = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
INITIAL_VAF      = 0.05
S_RESOLUTION     = 60
H_RESOLUTION     = 40
EPS              = 1e-8

OUTPUT_CSV = "synthetic_benchmark_results.csv"
OUTPUT_FIG = "synthetic_benchmark_plot.png"

# Colorblind-safe palette (Wong 2011)
C_HET  = "#2171b5"
C_HOM  = "#e6550d"
C_MIX  = "#009696"
C_IDEAL = "#999999"   # grey

# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def simulate_vaf(time_points, s, xhet0, h, N_w=N_W):
    t     = np.asarray(time_points, dtype=float)
    x_het = xhet0 * np.exp(s * (t - t[0]))
    x_hom = h * x_het
    x_tot = x_het + x_hom
    vaf   = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    ceiling = (1.0 + 2.0 * h) / (2.0 * (1.0 + h)) - EPS
    return np.clip(vaf, EPS, ceiling)


def initial_xhet_from_vaf(v0, h, N_w=N_W):
    denom = (1.0 + 2.0 * h) - 2.0 * v0 * (1.0 + h)
    denom = max(denom, EPS)
    return max(2.0 * v0 * N_w / denom, 10.0)


def generate_participant(s_true, h_true, seed):
    rng      = np.random.default_rng(seed)
    xhet0    = initial_xhet_from_vaf(INITIAL_VAF, h_true)
    vaf_true = simulate_vaf(TIME_POINTS, s_true, xhet0, h_true)
    AO       = rng.binomial(DEPTH, vaf_true).astype(float)
    DP       = np.full(len(TIME_POINTS), DEPTH, dtype=float)
    return AO, DP, TIME_POINTS.copy(), vaf_true


# ---------------------------------------------------------------------------
# Log-likelihood
# ---------------------------------------------------------------------------

def log_likelihood_sh(AO, DP, time_points, s, h, N_w=N_W):
    v0       = AO[0] / max(DP[0], 1.0)
    xhet0    = initial_xhet_from_vaf(v0, h, N_w)
    vaf_pred = simulate_vaf(time_points, s, xhet0, h, N_w)
    ll = 0.0
    for i in range(len(AO)):
        ll += binom.logpmf(int(AO[i]), int(DP[i]), vaf_pred[i])
    return ll


# ---------------------------------------------------------------------------
# Inference pipelines
# ---------------------------------------------------------------------------

def infer_mixed(AO, DP, time_points):
    s_range   = np.linspace(0.01, 1.5, S_RESOLUTION)
    h_range   = np.linspace(0.0,  1.0, H_RESOLUTION)
    log_joint = np.full((S_RESOLUTION, H_RESOLUTION), -np.inf)
    for si, s in enumerate(s_range):
        for hi, h in enumerate(h_range):
            log_joint[si, hi] = log_likelihood_sh(AO, DP, time_points, s, h)
    log_joint -= log_joint.max()
    joint = np.exp(np.clip(log_joint, -700, 0))
    joint /= max(joint.sum(), EPS)
    si_map, hi_map = np.unravel_index(np.argmax(joint), joint.shape)
    s_map = float(s_range[si_map])
    h_map = float(h_range[hi_map])
    s_post = joint.sum(axis=1); s_post /= max(s_post.sum(), EPS)
    h_post = joint.sum(axis=0); h_post /= max(h_post.sum(), EPS)
    sc = np.cumsum(s_post); hc = np.cumsum(h_post)
    s_ci = (float(np.interp(0.05, sc, s_range)), float(np.interp(0.95, sc, s_range)))
    h_ci = (float(np.interp(0.05, hc, h_range)), float(np.interp(0.95, hc, h_range)))
    return s_map, h_map, s_ci, h_ci


def infer_fixed_h(AO, DP, time_points, h_fixed):
    s_range = np.linspace(0.01, 1.5, S_RESOLUTION)
    log_lik = np.array([log_likelihood_sh(AO, DP, time_points, s, h_fixed) for s in s_range])
    log_lik -= log_lik.max()
    post = np.exp(np.clip(log_lik, -700, 0))
    post /= max(post.sum(), EPS)
    s_map = float(s_range[np.argmax(post)])
    cum   = np.cumsum(post)
    s_ci  = (float(np.interp(0.05, cum, s_range)), float(np.interp(0.95, cum, s_range)))
    return s_map, s_ci


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark():
    records = []
    n_total = len(H_VALUES) * N_REPS
    done    = 0

    print(f"Running: s_true={S_TRUE}, {len(H_VALUES)} h-values x {N_REPS} reps = {n_total} cells x 3 pipelines\n")

    for h_true in H_VALUES:
        s_mix_list = []; h_mix_list = []
        s_het_list = []; s_hom_list = []

        for rep in range(N_REPS):
            seed = BASE_SEED + int(round(h_true * 100)) * 1000 + rep
            AO, DP, tps, _ = generate_participant(S_TRUE, h_true, seed)

            try:
                sm, hm, _, _ = infer_mixed(AO, DP, tps)
                s_mix_list.append(sm); h_mix_list.append(hm)
            except Exception as e:
                print(f"  [mixed] h={h_true:.1f} rep={rep} FAILED: {e}")
                s_mix_list.append(np.nan); h_mix_list.append(np.nan)

            try:
                sh, _ = infer_fixed_h(AO, DP, tps, 0.0)
                s_het_list.append(sh)
            except Exception as e:
                s_het_list.append(np.nan)

            try:
                sho, _ = infer_fixed_h(AO, DP, tps, 1.0)
                s_hom_list.append(sho)
            except Exception as e:
                s_hom_list.append(np.nan)

            done += 1
            if done % 20 == 0:
                print(f"  {done}/{n_total}", flush=True)

        def sem(a): return np.nanstd(a) / np.sqrt(np.sum(~np.isnan(a)))

        s_mix = np.array(s_mix_list); h_mix = np.array(h_mix_list)
        s_het = np.array(s_het_list); s_hom = np.array(s_hom_list)

        # RMSE per h value
        rmse_mix = np.sqrt(np.nanmean((s_mix - S_TRUE)**2))
        rmse_het = np.sqrt(np.nanmean((s_het - S_TRUE)**2))
        rmse_hom = np.sqrt(np.nanmean((s_hom - S_TRUE)**2))

        records.append(dict(
            h_true     = h_true,
            s_mix_mean = np.nanmean(s_mix), s_mix_sem = sem(s_mix), s_mix_rmse = rmse_mix,
            h_mix_mean = np.nanmean(h_mix), h_mix_sem = sem(h_mix),
            s_het_mean = np.nanmean(s_het), s_het_sem = sem(s_het), s_het_rmse = rmse_het,
            s_hom_mean = np.nanmean(s_hom), s_hom_sem = sem(s_hom), s_hom_rmse = rmse_hom,
        ))
        r = records[-1]
        print(f"  h={h_true:.1f}  s_mix={r['s_mix_mean']:.3f}  h_mix={r['h_mix_mean']:.3f}"
              f"  s_het={r['s_het_mean']:.3f}  s_hom={r['s_hom_mean']:.3f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot — poster layout
# ---------------------------------------------------------------------------

def plot_results(df, output_path):
    h_true = df["h_true"].values

    fig = plt.figure(figsize=(10, 11))
    fig.patch.set_facecolor("white")

    # Top panel tall, bottom panel shorter, shared x-axis
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[2.2, 1],
        hspace=0.08,
        left=0.12, right=0.95,
        top=0.93, bottom=0.09,
    )

    eb_kw = dict(lw=2.5, ms=8, capsize=5, capthick=2, elinewidth=2, zorder=5)

    # =========================================================
    # TOP: fitness inference
    # =========================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#cccccc")
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False, labelsize=13)

    # True fitness line
    ax1.axhline(S_TRUE, color=C_IDEAL, ls="--", lw=2, label="True Fitness", zorder=2)

    # Shaded error regions
    ax1.fill_between(h_true,
                     df["s_het_mean"] - 1.96*df["s_het_sem"],
                     df["s_het_mean"] + 1.96*df["s_het_sem"],
                     color=C_HET, alpha=0.12)
    ax1.fill_between(h_true,
                     df["s_hom_mean"] - 1.96*df["s_hom_sem"],
                     df["s_hom_mean"] + 1.96*df["s_hom_sem"],
                     color=C_HOM, alpha=0.12)
    ax1.fill_between(h_true,
                     df["s_mix_mean"] - 1.96*df["s_mix_sem"],
                     df["s_mix_mean"] + 1.96*df["s_mix_sem"],
                     color=C_MIX, alpha=0.15)

    ax1.errorbar(h_true, df["s_het_mean"], yerr=1.96*df["s_het_sem"],
                 fmt="s-", color=C_HET, label="Het-only model", **eb_kw)
    ax1.errorbar(h_true, df["s_hom_mean"], yerr=1.96*df["s_hom_sem"],
                 fmt="^-", color=C_HOM, label="Hom-only model", **eb_kw)
    ax1.errorbar(h_true, df["s_mix_mean"], yerr=1.96*df["s_mix_sem"],
                 fmt="o-", color=C_MIX, label="Mixed model", **eb_kw)

    ax1.set_ylim(0.2, 0.8)
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylabel("Inferred Fitness (s)", fontsize=15)
    ax1.grid(True, color="#eeeeee", lw=0.8, zorder=0)

    # RMSE summary in corner
    rmse_mix = np.sqrt(np.nanmean((df["s_mix_mean"] - S_TRUE)**2))
    rmse_het = np.sqrt(np.nanmean((df["s_het_mean"] - S_TRUE)**2))
    rmse_hom = np.sqrt(np.nanmean((df["s_hom_mean"] - S_TRUE)**2))
    summary = (f"RMSE\n"
               f"Mixed:    {rmse_mix:.3f}\n"
               f"Het-only: {rmse_het:.3f}\n"
               f"Hom-only: {rmse_hom:.3f}")
    ax1.text(0.02, 0.97, summary, transform=ax1.transAxes,
             fontsize=11, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    leg = ax1.legend(fontsize=12, edgecolor="#cccccc", loc="upper right",
                     framealpha=1)

    ax1.set_title("Mixed model recovers fitness accurately\nregardless of zygosity",
                  fontsize=16, fontweight="bold", pad=10)

    # =========================================================
    # BOTTOM: h recovery
    # =========================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor("white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#cccccc")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=13)

    het_frac_inferred = 1.0 - df["h_mix_mean"].values
    het_frac_sem      = df["h_mix_sem"].values

    ax2.plot(h_true, 1.0 - h_true, "--", color=C_IDEAL, lw=2, label="Perfect recovery")
    ax2.errorbar(h_true, het_frac_inferred, yerr=1.96*het_frac_sem,
                 fmt="o-", color=C_MIX, lw=2.5, ms=7,
                 capsize=5, capthick=2, elinewidth=2, label="Mixed model", zorder=5)

    ax2.set_xlim(-0.03, 1.03)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("True Homozygous Fraction (h)", fontsize=15)
    ax2.set_ylabel(r"Inferred $\frac{x_{Hom}}{x_{het}+x_{Hom}}$", fontsize=13)
    ax2.grid(True, color="#eeeeee", lw=0.8)
    ax2.legend(fontsize=11, edgecolor="#cccccc", framealpha=1)

    # footer
    fig.text(0.5, 0.01,
             f"s_true={S_TRUE}  |  6 timepoints × 3yr gaps  |  depth={DEPTH}×  |  n={N_REPS} replicates",
             ha="center", fontsize=10, color="#888888")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    df = run_benchmark()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved: {OUTPUT_CSV}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    plot_results(df, OUTPUT_FIG)
    print("Done.")