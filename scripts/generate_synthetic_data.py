"""
generate_synthetic_data.py

Generate synthetic longitudinal VAF data to test the KI_2 clonal-fitness /
homozygosity inference pipeline (scripts/7.KI_clonal_fit.py).

Forward model matches src/KI_2.py:
  - each clone k grows as a linear birth-death process with net fitness s_k
    (birth rate lamb+s_k, death rate lamb)  ->  NegBinom transitions
  - VAF_k(t) = (1 + h_k) * x_k(t) / (2 * (N_w + sum_j x_j(t)))
  - AO ~ Binomial(DP, VAF)

Ground-truth parameters are stored in part.obs (true_s, true_h, true_clone)
so parameter recovery can be checked after fitting.
"""

import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import pickle as pk
import anndata as ad


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OUTPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
SEED       = 42
N_W        = 1e5     # wild-type population size (matches src/KI_2.py)
LAMB       = 1.3     # death rate (matches src/KI_2.py)
DEPTH      = 2000    # mean sequencing depth
STOCHASTIC = True    # BD stochastic growth (False -> deterministic exp growth)

rng = np.random.default_rng(SEED)

# Each participant: list of clones. Each clone has net fitness s, homozygosity
# fraction h, initial size x0, and a list of mutation names.
PARTICIPANTS = [
    # 1. single clone, single mutation, no homozygosity
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.45, "h": 0.0, "x0": 2.5e4, "mutations": ["DNMT3A"]}]},

    # 2. single clone, two mutations (should be grouped together)
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.55, "h": 0.0, "x0": 1.5e4,
                 "mutations": ["TET2_a", "TET2_b"]}]},

    # 3. two independent clones with different fitness
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.30, "h": 0.0, "x0": 3.0e4, "mutations": ["ASXL1"]},
                {"s": 0.70, "h": 0.0, "x0": 1.0e4, "mutations": ["SRSF2"]}]},

    # 4. homozygous clone, moderate VAF (NOT saturating -> h weakly identified)
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.50, "h": 0.4, "x0": 3.0e4, "mutations": ["JAK2"]}]},

    # 5. mixed: one heterozygous clone + one partially homozygous clone
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.40, "h": 0.0, "x0": 2.0e4, "mutations": ["TP53"]},
                {"s": 0.60, "h": 0.3, "x0": 3.0e4, "mutations": ["EZH2"]}]},

    # 6. strongly homozygous clone driven into VAF saturation
    #    (the ONLY regime where h is identifiable; final VAF ~0.65, pole=0.70).
    #    Acceptance test: h should recover ~0.4 with a CI that excludes 0.
    {"time_points": [0, 1, 2, 3, 4],
     "clones": [{"s": 0.55, "h": 0.4, "x0": 1.5e5, "mutations": ["JAK2_sat"]}]},
]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def bd_step(x, s, dt):
    """Sample the next clone size from the linear birth-death transition
    (same NegBinom method-of-moments parametrisation used in src/KI_2.py)."""
    if x <= 0:
        return 0.0
    exp_term = np.exp(dt * s)
    mean = x * exp_term
    variance = x * (2 * LAMB + s) * exp_term * (exp_term - 1) / s
    if variance <= mean:               # guard for tiny dt / numerical edge
        return mean
    p = mean / variance
    n = mean ** 2 / (variance - mean)
    return float(rng.negative_binomial(n, p))


def simulate_clone(x0, s, time_points):
    """Clone size trajectory over the given time points."""
    tp = np.asarray(time_points, dtype=float)
    sizes = [float(x0)]
    for i in range(1, len(tp)):
        dt = tp[i] - tp[i - 1]
        if STOCHASTIC:
            sizes.append(bd_step(sizes[-1], s, dt))
        else:
            sizes.append(sizes[-1] * np.exp(s * dt))
    return np.array(sizes)


def build_participant(cfg, pid):
    tp = np.asarray(cfg["time_points"], dtype=float)
    n_tps = len(tp)

    # simulate clone-size trajectories
    clone_sizes = np.array([simulate_clone(c["x0"], c["s"], tp)
                            for c in cfg["clones"]])          # (n_clones, n_tps)
    total_pop = N_W + clone_sizes.sum(axis=0)                 # (n_tps,)

    mut_names, true_s, true_h, true_clone = [], [], [], []
    vaf_rows, ao_rows, dp_rows = [], [], []

    for k, clone in enumerate(cfg["clones"]):
        vaf_k = (1 + clone["h"]) * clone_sizes[k] / (2 * total_pop)
        vaf_k = np.clip(vaf_k, 0.0, 0.999999)
        for mut in clone["mutations"]:
            dp = np.clip(rng.poisson(DEPTH, size=n_tps), 1, None).astype(float)
            ao = rng.binomial(dp.astype(int), vaf_k).astype(float)
            mut_names.append(mut)
            true_s.append(clone["s"])
            true_h.append(clone["h"])
            true_clone.append(k)
            vaf_rows.append(ao / dp)
            ao_rows.append(ao)
            dp_rows.append(dp)

    X  = np.array(vaf_rows)   # observed VAF (n_mut, n_tps)
    AO = np.array(ao_rows)
    DP = np.array(dp_rows)

    obs = pd.DataFrame({
        "p_key": mut_names,
        "true_s": true_s,
        "true_h": true_h,
        "true_clone": true_clone,
    }, index=mut_names)

    var = pd.DataFrame({"time_points": tp},
                       index=[f"t{t:g}" for t in tp])

    part = ad.AnnData(X=X, obs=obs, var=var)
    part.layers["AO"] = AO
    part.layers["DP"] = DP
    part.uns["participant_id"] = pid
    part.uns["warning"] = None
    return part


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    cohort = [build_participant(cfg, i) for i, cfg in enumerate(PARTICIPANTS)]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pk.dump(cohort, f, protocol=4)

    print(f"Wrote {len(cohort)} synthetic participants -> {OUTPUT_FILE}\n")
    for i, part in enumerate(cohort):
        print(f"Participant {i + 1}: {part.shape[0]} mutations, "
              f"{part.shape[1]} timepoints")
        for j, mut in enumerate(part.obs.index):
            row = part.obs.loc[mut]
            final_vaf = part.X[j, -1]
            # saturation-implied lower bound on h for this mutation
            h_min = max(0.0, 2 * final_vaf - 1)
            print(f"   {mut:10s} s={row.true_s:.2f} h={row.true_h:.2f} "
                  f"clone={int(row.true_clone)} finalVAF={final_vaf:.3f} "
                  f"(h_min>={h_min:.2f})")
        print()


if __name__ == "__main__":
    main()
