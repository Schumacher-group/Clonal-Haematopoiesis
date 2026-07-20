"""
Single-s synthetic recovery test.

Fixes a single s value and sweeps h across [0, 1] to test whether
the model correctly recovers zygosity from VAF dynamics.
"""

import sys
import itertools
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append("..")

# =============================================================================
# Configuration
# =============================================================================

S_FIXED       = 0.3
H_TRUE_VALUES = np.round(np.arange(0.0, 1.05, 0.1), 2)

N_W           = 1e5
DEPTH         = 500
SEED          = 42

N_TIMEPOINTS  = 4
SPACING_YEARS = 3.0
INITIAL_VAF   = 0.10

S_RESOLUTION  = 30
H_RESOLUTION  = 20
REFINE_S      = 50
REFINE_H      = 30

OUTPUT_CSV    = f"single_s{S_FIXED}_recovery.csv"
OUTPUT_FIG    = f"single_s{S_FIXED}_recovery.png"

# =============================================================================
# AnnData-compatible container (same as benchmark script)
# =============================================================================

class SimpleVar:
    def __init__(self, time_points):
        self.time_points = pd.Series(time_points)
        self.columns = ["time_points"]


class SimpleObs:
    def __init__(self, n):
        self._df = pd.DataFrame(index=[f"mut_{i}" for i in range(n)])
        self.index = self._df.index
        self.columns = self._df.columns

    @property
    def iloc(self):
        return self._df.iloc

    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        self._df[key] = value
        self.columns = self._df.columns

    def __contains__(self, key):
        return key in self._df.columns

    def to_string(self):
        return self._df.to_string()


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
        part.obs._df = self.obs._df.iloc[idx:idx+1].copy()
        part.obs.index = part.obs._df.index
        return part

    @property
    def X(self):
        return self.layers["AO"] / np.maximum(self.layers["DP"], 1.0)


# =============================================================================
# Forward model
# =============================================================================

def make_time_points(n_points=N_TIMEPOINTS, spacing_years=SPACING_YEARS):
    return np.arange(n_points, dtype=float) * spacing_years


def simulate_clone(time_points, s, xhet0, h, N_w=N_W):
    t      = np.asarray(time_points, dtype=float)
    x_het  = xhet0 * np.exp(s * (t - t[0]))
    x_hom  = h * x_het
    x_tot  = x_het + x_hom
    vaf    = (x_het + 2.0 * x_hom) / (2.0 * (N_w + x_tot))
    return x_het, x_hom, vaf


def generate_synthetic_participant(s_true, h_true, depth, seed, N_w=N_W):
    rng        = np.random.default_rng(seed)
    time_points = make_time_points()
    xhet0      = max(2.0 * N_w * INITIAL_VAF / (1.0 + h_true), 10.0)
    _, _, vaf_true = simulate_clone(time_points, s_true, xhet0, h_true, N_w)
    vaf_ceiling = (1.0 + h_true) / 2.0
    vaf_true    = np.clip(vaf_true, 1e-6, vaf_ceiling - 1e-6)
    DP = np.full((1, len(time_points)), depth, dtype=float)
    AO = rng.binomial(depth, vaf_true).reshape(1, -1).astype(float)
    return SyntheticParticipant(AO, DP, time_points)


# =============================================================================
# Run sweep
# =============================================================================

def run_sweep():
    from src.KI_clonal_inference_5 import (
        compute_clonal_models_prob_vec_mixed,
        refine_optimal_model_posterior_vec_mixed,
    )

    records = []
    time_points = make_time_points()

    print(f"\nSingle-s recovery sweep  (s_fixed={S_FIXED})")
    print(f"h values: {H_TRUE_VALUES}")
    print(f"Timepoints: {time_points}\n")

    for idx, h_true in enumerate(H_TRUE_VALUES):

        print(f"[{idx+1:2d}/{len(H_TRUE_VALUES)}]  h_true={h_true:.2f}", end="  ")

        try:
            part = generate_synthetic_participant(
                S_FIXED, h_true, DEPTH, seed=SEED + idx
            )

            # Print the actual observed VAF trajectory so we can sanity check
            obs_vaf = (part.layers["AO"] / np.maximum(part.layers["DP"], 1.0))[0]
            ceiling = (1.0 + h_true) / 2.0
            print(f"VAF: {' -> '.join(f'{v:.3f}' for v in obs_vaf)}  ceiling={ceiling:.2f}", end="  ")

            part = compute_clonal_models_prob_vec_mixed(
                part,
                s_resolution=S_RESOLUTION,
                h_resolution=H_RESOLUTION,
                filter_invalid=False,
                disable_progressbar=True,
            )

            part = refine_optimal_model_posterior_vec_mixed(
                part,
                s_resolution=REFINE_S,
                h_resolution=REFINE_H,
            )

            s_inf = float(part.obs["fitness"].iloc[0])
            h_inf = float(part.obs["zygosity"].iloc[0]) if "zygosity" in part.obs._df.columns else np.nan
            s_lo  = float(part.obs["fitness_5"].iloc[0])
            s_hi  = float(part.obs["fitness_95"].iloc[0])
            h_lo  = float(part.obs["zygosity_5"].iloc[0])  if "zygosity_5"  in part.obs._df.columns else np.nan
            h_hi  = float(part.obs["zygosity_95"].iloc[0]) if "zygosity_95" in part.obs._df.columns else np.nan
            warning = part.uns.get("warning") or ""

            # Joint posterior from refine step
            joint_posterior = part.uns["optimal_model"]["joint_inference"][0]["joint_posterior"]
            s_range         = part.uns["optimal_model"]["joint_inference"][0]["s_range"]
            h_range         = part.uns["optimal_model"]["joint_inference"][0]["h_range"]

            print(f"s_inf={s_inf:.2f}  h_inf={h_inf:.2f}  {warning}")

            records.append(dict(
                h_true=h_true,
                s_true=S_FIXED,
                s_inf=s_inf,
                h_inf=h_inf,
                s_err=s_inf - S_FIXED,
                h_err=h_inf - h_true,
                s_lo=s_lo, s_hi=s_hi,
                h_lo=h_lo, h_hi=h_hi,
                s_in_ci=int(s_lo <= S_FIXED <= s_hi),
                h_in_ci=int(h_lo <= h_true <= h_hi) if not np.isnan(h_lo) else np.nan,
                obs_vaf_final=float(obs_vaf[-1]),
                vaf_ceiling=ceiling,
                warning=warning,
                status="ok",
                joint_posterior=joint_posterior,
                s_range=s_range,
                h_range=h_range,
            ))

        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            records.append(dict(
                h_true=h_true, s_true=S_FIXED,
                s_inf=np.nan, h_inf=np.nan,
                s_err=np.nan, h_err=np.nan,
                s_lo=np.nan, s_hi=np.nan,
                h_lo=np.nan, h_hi=np.nan,
                s_in_ci=np.nan, h_in_ci=np.nan,
                obs_vaf_final=np.nan, vaf_ceiling=np.nan,
                warning="", status=f"error: {e}",
                joint_posterior=None, s_range=None, h_range=None,
            ))

    return records


# =============================================================================
# Plotting
# =============================================================================

def plot_results(records, output_path):

    ok = [r for r in records if r["status"] == "ok"]
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("joint_posterior", "s_range", "h_range")}
                       for r in records])

    n_h    = len(ok)
    time_points = make_time_points()

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Single-s recovery  (s_fixed={S_FIXED},  depth={DEPTH},  "
        f"spacing={SPACING_YEARS}yr,  VAF_init={INITIAL_VAF})",
        fontsize=13, fontweight="bold",
    )

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                              height_ratios=[2.5, 2.5, 3.5])

    # ------------------------------------------------------------------
    # Row 1: inferred s and h vs true h, with CI ribbons
    # ------------------------------------------------------------------
    ax_s = fig.add_subplot(outer[0])
    ax_h = ax_s.twinx()

    h_vals  = df[df.status=="ok"]["h_true"].values
    s_infs  = df[df.status=="ok"]["s_inf"].values
    h_infs  = df[df.status=="ok"]["h_inf"].values
    s_los   = df[df.status=="ok"]["s_lo"].values
    s_his   = df[df.status=="ok"]["s_hi"].values
    h_los   = df[df.status=="ok"]["h_lo"].values
    h_his   = df[df.status=="ok"]["h_hi"].values

    ax_s.fill_between(h_vals, s_los, s_his, alpha=0.15, color="steelblue")
    ax_s.plot(h_vals, s_infs, "o-", color="steelblue", label="s inferred", zorder=3)
    ax_s.axhline(S_FIXED, color="steelblue", ls="--", lw=1.2, alpha=0.6, label="s true")

    ax_h.fill_between(h_vals, h_los, h_his, alpha=0.15, color="tomato")
    ax_h.plot(h_vals, h_infs, "s-", color="tomato", label="h inferred", zorder=3)
    ax_h.plot(h_vals, h_vals, color="tomato", ls="--", lw=1.2, alpha=0.6, label="h true")

    ax_s.set_xlabel("True h")
    ax_s.set_ylabel("Inferred s", color="steelblue")
    ax_h.set_ylabel("Inferred h", color="tomato")
    ax_s.set_ylim(0, 1.1)
    ax_h.set_ylim(0, 1.1)
    ax_s.set_title("Inferred s (blue) and h (red) vs true h, with 90% CI ribbons")

    lines1, labels1 = ax_s.get_legend_handles_labels()
    lines2, labels2 = ax_h.get_legend_handles_labels()
    ax_s.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # ------------------------------------------------------------------
    # Row 2: observed VAF trajectories coloured by h_true
    # ------------------------------------------------------------------
    ax_vaf = fig.add_subplot(outer[1])
    cmap   = plt.cm.plasma
    norm   = plt.Normalize(vmin=0, vmax=1)

    for r in ok:
        part = generate_synthetic_participant(
            S_FIXED, r["h_true"], DEPTH,
            seed=SEED + list(H_TRUE_VALUES).index(r["h_true"])
        )
        obs_vaf = (part.layers["AO"] / np.maximum(part.layers["DP"], 1.0))[0]
        color   = cmap(norm(r["h_true"]))

        # True trajectory
        xhet0 = max(2.0 * N_W * INITIAL_VAF / (1.0 + r["h_true"]), 10.0)
        _, _, vaf_true = simulate_clone(time_points, S_FIXED, xhet0, r["h_true"])
        ax_vaf.plot(time_points, vaf_true, color=color, lw=1.5, alpha=0.7)

        # Inferred trajectory
        if not (np.isnan(r["s_inf"]) or np.isnan(r["h_inf"])):
            xhet0_inf = max(obs_vaf[0] * 2.0 * N_W / (1.0 + r["h_inf"]), 10.0)
            _, _, vaf_inf = simulate_clone(time_points, r["s_inf"], xhet0_inf, r["h_inf"])
            ax_vaf.plot(time_points, vaf_inf, color=color, lw=1.5, ls="--", alpha=0.9)

        # VAF ceiling
        ax_vaf.axhline(r["vaf_ceiling"], color=color, lw=0.5, ls=":", alpha=0.5)

        ax_vaf.scatter(time_points, obs_vaf, color=color, s=20, zorder=5, alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_vaf, label="True h")

    ax_vaf.set_xlabel("Time (years)")
    ax_vaf.set_ylabel("VAF")
    ax_vaf.set_ylim(0, 1)
    ax_vaf.set_title(
        "VAF trajectories by true h  "
        "(solid=true, dashed=inferred, dots=observed, dotted=ceiling)"
    )

    # ------------------------------------------------------------------
    # Row 3: joint (s, h) posterior for each h_true value
    # ------------------------------------------------------------------
    inner = gridspec.GridSpecFromSubplotSpec(
        1, n_h, subplot_spec=outer[2], wspace=0.05
    )

    for col, r in enumerate(ok):
        ax = fig.add_subplot(inner[col])
        jp = r["joint_posterior"]
        sr = r["s_range"]
        hr = r["h_range"]

        ax.imshow(
            jp,
            origin="lower",
            extent=[hr.min(), hr.max(), sr.min(), sr.max()],
            aspect="auto",
            cmap="hot_r",
        )

        # True values
        ax.axhline(S_FIXED,      color="steelblue", lw=1.0, ls="--")
        ax.axvline(r["h_true"],  color="tomato",    lw=1.0, ls="--")

        ax.set_xlabel(f"h={r['h_true']:.1f}", fontsize=7)
        if col == 0:
            ax.set_ylabel("s", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5)

    fig.text(0.5, 0.01,
             "Row 3: joint (s,h) posterior heatmaps — blue dashed=true s, red dashed=true h",
             ha="center", fontsize=9, style="italic")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {output_path}")
    plt.show()


# =============================================================================
# Summary
# =============================================================================

def print_summary(records):
    ok = [r for r in records if r["status"] == "ok"]
    s_errs = np.array([r["s_err"] for r in ok])
    h_errs = np.array([r["h_err"] for r in ok])

    print("\n" + "=" * 50)
    print(f"SUMMARY  (s_fixed={S_FIXED})")
    print("=" * 50)
    print(f"  s MAE:  {np.abs(s_errs).mean():.3f}   bias: {s_errs.mean():+.3f}")
    print(f"  h MAE:  {np.abs(h_errs).mean():.3f}   bias: {h_errs.mean():+.3f}")
    print(f"  s 90% CI coverage: {np.mean([r['s_in_ci'] for r in ok])*100:.1f}%")
    h_cov = np.nanmean([r["h_in_ci"] for r in ok if not isinstance(r["h_in_ci"], float) or not np.isnan(r["h_in_ci"])])
    print(f"  h 90% CI coverage: {h_cov*100:.1f}%")

    print(f"\n  {'h_true':>8}  {'h_inf':>8}  {'h_err':>8}  {'s_inf':>8}  {'final_VAF':>10}  {'ceiling':>8}")
    for r in ok:
        print(f"  {r['h_true']:>8.2f}  {r['h_inf']:>8.2f}  {r['h_err']:>+8.2f}  "
              f"{r['s_inf']:>8.2f}  {r['obs_vaf_final']:>10.3f}  {r['vaf_ceiling']:>8.3f}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    records = run_sweep()

    df_out = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("joint_posterior", "s_range", "h_range")}
                           for r in records])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved: {OUTPUT_CSV}")

    print_summary(records)
    plot_results(records, OUTPUT_FIG)