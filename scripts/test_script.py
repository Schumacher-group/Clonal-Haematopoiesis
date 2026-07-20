"""
h_sweep_recovery.py -- fix s=0.5, sweep true h = 0.0..1.0, check h recovery.
Single saturating clone per patient so h can be identified via the VAF ceiling (1+h)/2.
Writes to a SAFE path (not the real cohort).
"""
import os, sys; sys.path.append("..")
import numpy as np, pandas as pd, pickle as pk, anndata as ad
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.KI_2 import (compute_clonal_models_prob_vec,
                      refine_optimal_model_posterior_vec)

# ---------------- config ----------------
SYNTH_FILE = "../exports/MDS/h_sweep_SYNTHETIC.pk"      # NOT the real cohort
PLOT_FILE  = "../exports/MDS/h_sweep_recovery.png"
SEED   = 42
N_W    = 1e5
LAMB   = 1.3
DEPTH  = 2000
S_TRUE = 0.50
X0     = 1e5                       # = N_w  -> strong saturation within window
TIME_POINTS = [0,1,2,3,4,5,6]      # enough growth to approach the ceiling
H_LIST = [round(h,1) for h in np.arange(0.0, 1.01, 0.1)]   # 11 patients

# fit grids (single clone -> h grid is cheap, make it FINE so CI can form)
S_RES, H_RES        = 20, 4
REFINE_S, REFINE_H  = 30, 10
MIN_S, MAX_S, MAX_H = 0.01, 1.0, 1.0

rng = np.random.default_rng(SEED)

# ---------------- forward model (matches KI_2) ----------------
def bd_step(x, s, dt):
    if x <= 0: return 0.0
    e = np.exp(dt*s); mean = x*e
    var = x*(2*LAMB+s)*e*(e-1)/s
    if var <= mean: return mean
    p = mean/var; n = mean**2/(var-mean)
    return float(rng.negative_binomial(n, p))

def simulate(x0, s, tp):
    tp = np.asarray(tp, float); xs = [float(x0)]
    for i in range(1, len(tp)):
        xs.append(bd_step(xs[-1], s, tp[i]-tp[i-1]))
    return np.array(xs)

def build_patient(h, pid):
    tp = np.asarray(TIME_POINTS, float)
    x = simulate(X0, S_TRUE, tp)
    total = N_W + x
    vaf = np.clip((1+h)*x/(2*total), 0.0, 0.999999)
    dp = np.clip(rng.poisson(DEPTH, size=len(tp)), 1, None).astype(float)
    ao = rng.binomial(dp.astype(int), vaf).astype(float)
    obs = pd.DataFrame({"p_key":[f"MUT_h{h:.1f}"], "true_s":[S_TRUE],
                        "true_h":[h], "true_clone":[0]}, index=[f"MUT_h{h:.1f}"])
    var = pd.DataFrame({"time_points": tp}, index=[f"t{t:g}" for t in tp])
    part = ad.AnnData(X=ao[None,:]/dp[None,:], obs=obs, var=var)
    part.layers["AO"] = ao[None,:]; part.layers["DP"] = dp[None,:]
    part.uns["participant_id"] = f"h{h:.1f}"; part.uns["warning"] = None
    return part

# ---------------- generate ----------------
cohort = [build_patient(h, i) for i, h in enumerate(H_LIST)]
os.makedirs(os.path.dirname(SYNTH_FILE), exist_ok=True)
with open(SYNTH_FILE, "wb") as f: pk.dump(cohort, f, protocol=4)
print(f"generated {len(cohort)} patients (s={S_TRUE}) -> {SYNTH_FILE}")
for part in cohort:
    vmax = float(np.nanmax(part.X)); ceil = (1+part.obs['true_h'][0])/2
    print(f"  h={part.obs['true_h'][0]:.1f}  Vmax={vmax:.3f}  ceiling={ceil:.3f}"
          f"  {'identifiable' if vmax>0.5 else 'NOT identifiable (Vmax<=0.5)'}")

# ---------------- fit + evaluate ----------------
rows = []
for part in tqdm(cohort, desc="fitting h-sweep", unit="patient"):
    vmax = float(np.nanmax(part.X))
    part = compute_clonal_models_prob_vec(part, s_resolution=S_RES, h_resolution=H_RES,
                                          min_s=MIN_S, max_s=MAX_S, max_h=MAX_H,
                                          disable_progressbar=True)
    part = refine_optimal_model_posterior_vec(part, s_resolution=REFINE_S,
                                              h_resolution=REFINE_H, min_s=MIN_S,
                                              max_s=MAX_S, max_h=MAX_H)
    o = part.obs.iloc[0]
    rows.append(dict(true_h=o["true_h"], h=o["homozygosity"],
                     h_lo=o["homozygosity_5"], h_hi=o["homozygosity_95"],
                     s=o["fitness"], vmax=vmax, ident=vmax>0.5,
                     covered=(o["true_h"]>=o["homozygosity_5"]-1e-9) and
                             (o["true_h"]<=o["homozygosity_95"]+1e-9),
                     ci_w=o["homozygosity_95"]-o["homozygosity_5"]))

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
idf = df[df["ident"]]
print(f"\nidentifiable patients (Vmax>0.5): {len(idf)}/{len(df)}")
if len(idf):
    print(f"  h MAP abs err (mean): {np.abs(idf['h']-idf['true_h']).mean():.3f}")
    print(f"  h CI coverage        : {idf['covered'].mean()*100:.0f}%")
    print(f"  zero-width CIs        : {(idf['ci_w']<1e-6).sum()}/{len(idf)}")

# ---------------- plot ----------------
plt.figure(figsize=(6,6))
c = np.where(df["ident"], "tab:blue", "0.7")
plt.errorbar(df["true_h"], df["h"], yerr=[df["h"]-df["h_lo"], df["h_hi"]-df["h"]],
             fmt="none", ecolor="lightgray", capsize=3, zorder=1)
plt.scatter(df["true_h"], df["h"], c=c, zorder=2)
plt.plot([0,1],[0,1],"k--",lw=1)
plt.axvspan(-0.02, 0.0, color="0.9")  # h=0 boundary reference
plt.xlabel("true h"); plt.ylabel("recovered h (MAP, 90% CI)")
plt.title(f"h recovery, s={S_TRUE} (blue = identifiable, Vmax>0.5)")
plt.xlim(-0.05,1.05); plt.ylim(-0.05,1.05); plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150); print(f"\nsaved -> {PLOT_FILE}")
