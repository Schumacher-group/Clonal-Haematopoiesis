"""
Synthetic VAF benchmark — mixed (KI_2) vs het_inference vs hom_inference.

Calls THREE separate modules (not stand-ins):
  - Mixed    : src.KI_2          — infers h (new maths, log-space, pole=(1+h)/2)
  - Het-only : src.het_inference — old linear maths, pole 0.5
               fit ONLY timepoints with observed VAF <= 0.5 (regime it can
               physically represent; above 0.5 the h=0 transform gives
               negative clone sizes).
  - Hom-only : src.hom_inference — old linear maths, pole 1.0
               all timepoints; where the old maths cannot cope the linear
               likelihood underflows -> nan -> shown as a gap.

Ground truth uses KI_2's convention: h = homozygous fraction of the clone,
VAF = (1+h) x / (2 (N_w + x)), clone grows as a linear birth-death process.

NOTE ON SPEED / HEAT: ensure none of the imported modules enable float64
(`jax.config.update("jax_enable_x64", True)`) — the config is process-wide,
so one module turning it on slows/heats them all. float32 is sufficient.
"""

import sys
sys.path.append("..")   # repo root so `src` is importable (run from scripts/)

import time
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax.numpy as jnp

# three ACTUAL pipelines (adjust module paths if your filenames differ)
import src.KI_2          as mix_mod   # mixed: infers h (new maths, log-space)
import src.het_inference as het_mod   # het-only: pole 0.5 (old linear maths)
import src.hom_inference as hom_mod   # hom-only: pole 1.0 (old linear maths)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S_TRUE       = 0.5
H_VALUES     = np.round(np.arange(0.0, 1.05, 0.1), 2)   # true homozygous fraction
N_W          = 1e5          # MUST match N_w hardcoded in the modules
LAMB         = 1.3          # MUST match lamb in the modules
DEPTH        = 2000
N_REPS       = 10           # bump up once you're happy with runtime
BASE_SEED    = 42
TIME_POINTS  = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
INITIAL_VAF  = 0.05
EPS          = 1e-8

# inference grids
MIN_S, MAX_S = 0.01, 1.5
S_RESOLUTION = 40
H_RESOLUTION = 21           # mixed-model h grid over [0, 1]

CS = [[0]]                  # single clone, single mutation

# posterior snapshots
SELECTED_H       = [0.0, 0.3, 0.6, 0.9]   # true-h values to show posteriors for
POSTERIOR_STORE  = {}

OUTPUT_CSV      = "ki2_benchmark_results.csv"
OUTPUT_FIG      = "ki2_benchmark_plot.png"
OUTPUT_POST_FIG = "ki2_benchmark_posteriors.png"

# Colorblind-safe palette (Wong 2011)
C_HET, C_HOM, C_MIX, C_IDEAL = "#2171b5", "#e6550d", "#009696", "#999999"

# precompute jax grids (compiled once, reused)
S_VEC = jnp.linspace(MIN_S, MAX_S, S_RESOLUTION)
H_VEC = np.linspace(0.0, 1.0, H_RESOLUTION)
TPS   = jnp.asarray(TIME_POINTS)


# ---------------------------------------------------------------------------
# Startup sanity check — confirm each module's signature / pole assumption
# ---------------------------------------------------------------------------
def sanity_check_modules():
    print("=" * 66)
    print("MODULE SANITY CHECK")
    print("=" * 66)

    ok = True
    for name, mod, expects_h in [("mixed  (KI_2)",          mix_mod, True),
                                 ("het    (het_inference)", het_mod, False),
                                 ("hom    (hom_inference)", hom_mod, False)]:
        try:
            sig_det = inspect.signature(mod.compute_deterministic_size)
            sig_ll  = inspect.signature(mod.jax_cs_hmm_ll_vec)
        except AttributeError as e:
            print(f"  [{name}] MISSING primitive: {e}")
            ok = False
            continue

        n_det = len(sig_det.parameters)
        n_ll  = len(sig_ll.parameters)
        # KI_2 (new): det takes (cs,AO,DP,n_mut,h)=5 ; ll takes (...,h)=8
        # old modules: det takes (cs,AO,DP,n_mut)=4 ; ll takes (...)=7
        has_h = (n_det >= 5)

        # try to read float64 state / N_w if present (best-effort)
        x64 = None
        try:
            import jax
            x64 = jax.config.read("jax_enable_x64")
        except Exception:
            pass

        flag = "OK" if has_h == expects_h else "MISMATCH!"
        if has_h != expects_h:
            ok = False
        print(f"  [{name}] det_args={n_det} ll_args={n_ll} "
              f"takes_h={has_h} (expected {expects_h})  -> {flag}")
        print(f"           det{sig_det}")
        print(f"           ll {sig_ll}")

    try:
        import jax
        print(f"\n  jax_enable_x64 = {jax.config.read('jax_enable_x64')} "
              f"(prefer False for speed/heat)")
    except Exception:
        pass

    print("=" * 66)
    if not ok:
        print("WARNING: signature mismatch — check the wrappers / module paths.")
    print()
    return ok


# ---------------------------------------------------------------------------
# Forward model (KI_2 convention)
# ---------------------------------------------------------------------------
def initial_x_from_vaf(v0, h, N_w=N_W):
    """Invert VAF = (1+h) x / (2 (N_w + x)) for x at t0."""
    pole  = (1.0 + h) / 2.0
    denom = max(pole - v0, EPS)
    return max(N_w * v0 / denom, 10.0)


def bd_step(x, s, dt, rng):
    """Linear birth-death transition (same NegBinom parametrisation as KI_2)."""
    if x <= 0:
        return 0.0
    e    = np.exp(s * dt)
    mean = x * e
    var  = x * (2 * LAMB + s) * e * (e - 1) / s
    if var <= mean:
        return mean
    p = mean / var
    n = mean ** 2 / (var - mean)
    return float(rng.negative_binomial(n, p))


def simulate(s, h, seed):
    """Return AO, DP of shape (n_tps, 1) for a single homozygous-fraction clone."""
    rng   = np.random.default_rng(seed)
    x     = initial_x_from_vaf(INITIAL_VAF, h)
    sizes = [x]
    for i in range(1, len(TIME_POINTS)):
        dt = TIME_POINTS[i] - TIME_POINTS[i - 1]
        x  = bd_step(x, s, dt, rng)
        sizes.append(x)
    sizes = np.asarray(sizes)

    vaf = (1.0 + h) * sizes / (2.0 * (N_W + sizes))
    vaf = np.clip(vaf, EPS, (1.0 + h) / 2.0 - EPS)

    AO = rng.binomial(DEPTH, vaf).astype(float)
    DP = np.full(len(TIME_POINTS), DEPTH, dtype=float)
    return AO[:, None], DP[:, None]


# ---------------------------------------------------------------------------
# Posterior helpers
# ---------------------------------------------------------------------------
def _norm_lin(p):
    p = np.nan_to_num(np.asarray(p), nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum()
    return p / s if s > 0 else None


# ---------------------------------------------------------------------------
# Call each ACTUAL module's own primitives
# ---------------------------------------------------------------------------
def module_post(mod, AO, DP, tps):
    """s-posterior from an OLD-signature module (het_mod / hom_mod)."""
    AOj = jnp.asarray(AO); DPj = jnp.asarray(DP); tpsj = jnp.asarray(tps)
    det, tot = mod.compute_deterministic_size(CS, AOj, DPj, AOj.shape[1])
    out = mod.jax_cs_hmm_ll_vec(S_VEC, AOj, DPj, tpsj, CS, det, tot)
    return np.nan_to_num(np.asarray(out)[:, 0], nan=0.0, posinf=0.0, neginf=0.0)


def mixed_post_h(AO, DP, h, tps):
    """s-posterior from KI_2 (NEW signature, takes h)."""
    AOj = jnp.asarray(AO); DPj = jnp.asarray(DP); tpsj = jnp.asarray(tps)
    det, tot = mix_mod.compute_deterministic_size(CS, AOj, DPj, AOj.shape[1], h)
    out = mix_mod.jax_cs_hmm_ll_vec(S_VEC, AOj, DPj, tpsj, CS, det, tot, h)
    return np.nan_to_num(np.asarray(out)[:, 0], nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def infer_het(AO, DP):
    """het_inference module (pole 0.5); fit ONLY timepoints with VAF <= 0.5."""
    vaf  = AO[:, 0] / DP[:, 0]
    keep = np.where(vaf <= 0.5)[0]
    if keep.size < 2:                       # not enough pre-saturation data
        return np.nan, None
    p = _norm_lin(module_post(het_mod, AO[keep], DP[keep], TIME_POINTS[keep]))
    if p is None:
        return np.nan, None
    return float(np.asarray(S_VEC)[np.argmax(p)]), p


def infer_hom(AO, DP):
    """hom_inference module (pole 1.0); all timepoints (old linear -> may nan)."""
    p = _norm_lin(module_post(hom_mod, AO, DP, TIME_POINTS))
    if p is None:                           # underflow -> old maths cannot fit
        return np.nan, None
    return float(np.asarray(S_VEC)[np.argmax(p)]), p


def infer_mixed(AO, DP):
    """KI_2 mixed: joint (s,h) grid; wrong-h cells underflow harmlessly."""
    grid = np.stack([mixed_post_h(AO, DP, float(h), TIME_POINTS) for h in H_VEC])
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)   # (n_h, n_s)
    tot  = grid.sum()
    if tot <= 0:
        return np.nan, np.nan, None
    joint = grid / tot
    p_s = joint.sum(axis=0); p_s /= max(p_s.sum(), EPS)
    p_h = joint.sum(axis=1); p_h /= max(p_h.sum(), EPS)
    s_map = float(np.asarray(S_VEC)[np.argmax(p_s)])
    h_map = float(H_VEC[np.argmax(p_h)])
    return s_map, h_map, joint


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark():
    records = []
    n_total = len(H_VALUES) * N_REPS
    done = 0
    print(f"s_true={S_TRUE} | {len(H_VALUES)} h-values x {N_REPS} reps "
          f"= {n_total} cells x 3 pipelines\n")

    for h_true in H_VALUES:
        s_mix, h_mix, s_het, s_hom = [], [], [], []

        for rep in range(N_REPS):
            seed = BASE_SEED + int(round(h_true * 100)) * 1000 + rep
            AO, DP = simulate(S_TRUE, h_true, seed)

            try:
                sm, hm, joint = infer_mixed(AO, DP)
            except Exception as e:
                print(f"  [mixed] h={h_true:.1f} rep={rep} FAILED: {e}")
                sm, hm, joint = np.nan, np.nan, None
            s_mix.append(sm); h_mix.append(hm)

            try:
                sh, p_het = infer_het(AO, DP)
            except Exception:
                sh, p_het = np.nan, None
            s_het.append(sh)

            try:
                sho, p_hom = infer_hom(AO, DP)
            except Exception:
                sho, p_hom = np.nan, None
            s_hom.append(sho)

            # stash posteriors for the first rep of selected h-values
            if rep == 0 and round(float(h_true), 2) in SELECTED_H:
                POSTERIOR_STORE[round(float(h_true), 2)] = dict(
                    p_het=p_het, p_hom=p_hom,
                    p_mix_s=(joint.sum(axis=0) if joint is not None else None),
                    joint=joint)

            done += 1
            if done % 10 == 0:
                print(f"  {done}/{n_total}", flush=True)

        def safe_mean(a):
            a = np.asarray(a, float)
            return np.nan if np.all(np.isnan(a)) else np.nanmean(a)

        def sem(a):
            a = np.asarray(a, float)
            n = int(np.sum(~np.isnan(a)))
            return np.nan if n == 0 else np.nanstd(a) / np.sqrt(n)

        def safe_rmse(a):
            a = np.asarray(a, float)
            return np.nan if np.all(np.isnan(a)) else np.sqrt(np.nanmean((a - S_TRUE) ** 2))

        s_mix = np.asarray(s_mix); h_mix = np.asarray(h_mix)
        s_het = np.asarray(s_het); s_hom = np.asarray(s_hom)

        records.append(dict(
            h_true     = h_true,
            s_mix_mean = safe_mean(s_mix), s_mix_sem = sem(s_mix),
            h_mix_mean = safe_mean(h_mix), h_mix_sem = sem(h_mix),
            s_het_mean = safe_mean(s_het), s_het_sem = sem(s_het),
            s_hom_mean = safe_mean(s_hom), s_hom_sem = sem(s_hom),
            s_mix_rmse = safe_rmse(s_mix),
            s_het_rmse = safe_rmse(s_het),
            s_hom_rmse = safe_rmse(s_hom),
            n_het      = int(np.sum(~np.isnan(s_het))),
            n_hom      = int(np.sum(~np.isnan(s_hom))),
        ))
        r = records[-1]
        print(f"  h={h_true:.1f}  s_mix={r['s_mix_mean']:.3f}  h_mix={r['h_mix_mean']:.3f}"
              f"  s_het={r['s_het_mean']:.3f} (n={r['n_het']})"
              f"  s_hom={r['s_hom_mean']:.3f} (n={r['n_hom']})")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot — summary (poster layout)
# ---------------------------------------------------------------------------
def plot_results(df, output_path):
    h_true = df["h_true"].values

    fig = plt.figure(figsize=(10, 11))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.08,
                           left=0.12, right=0.95, top=0.93, bottom=0.09)
    eb_kw = dict(lw=2.5, ms=8, capsize=5, capthick=2, elinewidth=2, zorder=5)

    # ---- TOP: fitness recovery ----
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("white")
    for sp in ax1.spines.values(): sp.set_edgecolor("#cccccc")
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False, labelsize=13)

    ax1.axhline(S_TRUE, color=C_IDEAL, ls="--", lw=2, label="True fitness", zorder=2)

    for mean, sem, col in [("s_het_mean", "s_het_sem", C_HET),
                           ("s_hom_mean", "s_hom_sem", C_HOM),
                           ("s_mix_mean", "s_mix_sem", C_MIX)]:
        ax1.fill_between(h_true,
                         df[mean] - 1.96 * df[sem],
                         df[mean] + 1.96 * df[sem],
                         color=col, alpha=0.13)

    ax1.errorbar(h_true, df["s_het_mean"], yerr=1.96*df["s_het_sem"],
                 fmt="s-", color=C_HET, label="Het-only (h=0)", **eb_kw)
    ax1.errorbar(h_true, df["s_hom_mean"], yerr=1.96*df["s_hom_sem"],
                 fmt="^-", color=C_HOM, label="Hom-only (h=1)", **eb_kw)
    ax1.errorbar(h_true, df["s_mix_mean"], yerr=1.96*df["s_mix_sem"],
                 fmt="o-", color=C_MIX, label="Mixed (infer h)", **eb_kw)

    ax1.set_xlim(-0.03, 1.03)
    ymax = np.nanmax([df["s_het_mean"].max(), df["s_hom_mean"].max(),
                      df["s_mix_mean"].max()])
    ax1.set_ylim(0.0, min(1.5, (0.6 if np.isnan(ymax) else ymax) + 0.2))
    ax1.set_ylabel("Inferred fitness (s)", fontsize=15)
    ax1.grid(True, color="#eeeeee", lw=0.8, zorder=0)

    rmse_mix = np.sqrt(np.nanmean((df["s_mix_mean"] - S_TRUE) ** 2))
    rmse_het = np.sqrt(np.nanmean((df["s_het_mean"] - S_TRUE) ** 2))
    rmse_hom = np.sqrt(np.nanmean((df["s_hom_mean"] - S_TRUE) ** 2))
    ax1.text(0.02, 0.97,
             f"RMSE\nMixed:    {rmse_mix:.3f}\nHet-only: {rmse_het:.3f}\nHom-only: {rmse_hom:.3f}",
             transform=ax1.transAxes, fontsize=11, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))
    ax1.legend(fontsize=12, edgecolor="#cccccc", loc="upper right", framealpha=1)
    ax1.set_title("Mixed model recovers fitness across zygosity\n"
                  "(vs fixed-zygosity het/hom pipelines)",
                  fontsize=16, fontweight="bold", pad=10)

    # ---- BOTTOM: h recovery ----
    ax2 = fig.add_subplot(gs[1], sharex=ax1); ax2.set_facecolor("white")
    for sp in ax2.spines.values(): sp.set_edgecolor("#cccccc")
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=13)

    ax2.plot([0, 1], [0, 1], "--", color=C_IDEAL, lw=2, label="Perfect recovery")
    ax2.errorbar(h_true, df["h_mix_mean"], yerr=1.96*df["h_mix_sem"],
                 fmt="o-", color=C_MIX, lw=2.5, ms=7,
                 capsize=5, capthick=2, elinewidth=2, label="Mixed model", zorder=5)

    ax2.set_xlim(-0.03, 1.03); ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("True homozygous fraction (h)", fontsize=15)
    ax2.set_ylabel("Inferred h", fontsize=14)
    ax2.grid(True, color="#eeeeee", lw=0.8)
    ax2.legend(fontsize=11, edgecolor="#cccccc", framealpha=1)

    fig.text(0.5, 0.01,
             f"s_true={S_TRUE}  |  {len(TIME_POINTS)} timepoints  |  "
             f"depth={DEPTH}x  |  n={N_REPS} replicates  |  "
             f"het fit to VAF<=0.5 timepoints only",
             ha="center", fontsize=10, color="#888888")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot — posterior distributions
# ---------------------------------------------------------------------------
def plot_posteriors(store, output_path):
    hs = [h for h in SELECTED_H if h in store]
    if not hs:
        print("No stored posteriors to plot."); return

    s_grid = np.asarray(S_VEC)
    fig, axes = plt.subplots(2, len(hs), figsize=(4.2 * len(hs), 8), squeeze=False)
    fig.patch.set_facecolor("white")

    for c, h in enumerate(hs):
        d = store[h]

        # --- top: s-posteriors, three pipelines ---
        ax = axes[0][c]; ax.set_facecolor("white")
        ax.axvline(S_TRUE, color=C_IDEAL, ls="--", lw=2, zorder=1)

        lines = []
        if d["p_het"] is not None:
            ax.plot(s_grid, d["p_het"] / d["p_het"].max(), color=C_HET, lw=2.5,
                    label="Het-only (h=0)")
            lines.append(("Het", s_grid[np.argmax(d["p_het"])]))
        else:
            ax.text(0.5, 0.55, "het: <2 usable\ntimepoints",
                    transform=ax.transAxes, ha="center", color=C_HET, fontsize=10)
        if d["p_hom"] is not None:
            ax.plot(s_grid, d["p_hom"] / d["p_hom"].max(), color=C_HOM, lw=2.5,
                    label="Hom-only (h=1)")
            lines.append(("Hom", s_grid[np.argmax(d["p_hom"])]))
        else:
            ax.text(0.5, 0.40, "hom: underflow\n(nan)",
                    transform=ax.transAxes, ha="center", color=C_HOM, fontsize=10)
        if d["p_mix_s"] is not None:
            pm = d["p_mix_s"]
            ax.plot(s_grid, pm / pm.max(), color=C_MIX, lw=2.5, label="Mixed")
            lines.append(("Mix", s_grid[np.argmax(pm)]))

        if lines:
            txt = "\n".join(f"{nm} s\u0302={sm:.2f}" for nm, sm in lines)
            ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9))

        ax.set_title(f"true h = {h:.1f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("fitness s"); ax.set_xlim(MIN_S, MAX_S); ax.set_ylim(0, 1.12)
        if c == 0:
            ax.set_ylabel("posterior (norm.)")
            ax.legend(fontsize=9, framealpha=1, edgecolor="#cccccc", loc="center left")
        for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")

        # --- bottom: mixed joint (s,h) heatmap ---
        ax = axes[1][c]; ax.set_facecolor("white")
        if d["joint"] is not None:
            J = d["joint"]                     # (n_h, n_s)
            ax.imshow(J, origin="lower", aspect="auto", cmap="viridis",
                      extent=[MIN_S, MAX_S, 0.0, 1.0])
            ax.scatter([S_TRUE], [h], marker="+", s=160, c="white",
                       linewidths=2.5, zorder=5)
        ax.set_xlabel("fitness s")
        if c == 0:
            ax.set_ylabel("homozygous fraction h")
        for sp in ax.spines.values(): sp.set_edgecolor("#cccccc")

    fig.suptitle("Posteriors: het/hom-only collapse or bias as zygosity grows;\n"
                 "mixed model stays peaked at true (s, h) [white +]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Posterior figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sanity_check_modules()

    t0 = time.perf_counter()
    df = run_benchmark()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved: {OUTPUT_CSV}")
    print(f"Total runtime: {time.perf_counter() - t0:.1f}s")
    plot_results(df, OUTPUT_FIG)
    plot_posteriors(POSTERIOR_STORE, OUTPUT_POST_FIG)
    print("Done.")
