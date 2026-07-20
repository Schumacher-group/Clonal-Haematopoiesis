"""
preprocess_cohort.py
--------------------
Converts long-format TSV files (one row per mutation per wave) into a list of
AnnData objects compatible with the KI clonal inference pipeline.

Accepts multiple input TSV files (e.g. one cohort split across two files)
which are concatenated before processing.

Each AnnData object represents one participant:
  - rows (obs)  : unique mutations, keyed by HGVSp (fallback: HGVSc)
  - columns (var): waves, with numeric time_points stored in var['time_points']
  - layers['AO'] : alt read counts  (n_mutations × n_waves)
  - layers['DP'] : total depth      (n_mutations × n_waves)

Usage
-----
    python preprocess_cohort.py

Outputs
-------
    ../exports/MDS/MDS_cohort_processed.pk
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import anndata as ad

# ── Configuration ────────────────────────────────────────────────────────────

# List all non-synonymous TSV files to merge — synonymous files are not needed.
INPUT_TSVS = [
    "../data/FabreData/Fabre_CHIP.05Jul24.1PCT_VAF_NON-SYNONYMOUS.tsv",   # ← change to your actual paths
    "../data/FabreData/Fabre_CHIP.05Jul24.2PCT_VAF_NON-SYNONYMOUS.tsv",
]
OUTPUT_FILE = "../exports/LBC/LBC_cohort_processed.pk"

# Wave → numeric time mapping.
# If waves are already the numeric times you want (e.g. years since diagnosis),
# set WAVE_TO_TIME = None and the integer wave values will be used directly.
# Otherwise provide a dict, e.g. {1: 0.0, 2: 1.5, 3: 3.0}
WAVE_TO_TIME = None

# Missing-wave fill strategy: 'median_zero' | 'zero' | 'drop_participant'
#   'median_zero'    : DP = participant's median DP across observed waves, AO = 0
#   'zero'           : DP = 0, AO = 0  (will trigger DP<=0 warnings in pipeline)
#   'drop_participant': exclude any participant with at least one missing wave
MISSING_WAVE_STRATEGY = "median_zero"

# Minimum number of waves a participant must have to be included
MIN_WAVES = 2

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_mutation_key(row: pd.Series) -> str:
    """HGVSp preferred, HGVSc fallback, positional last resort."""
    hgvsp = row.get("HGVSp", "")
    hgvsc = row.get("HGVSc", "")
    if pd.notna(hgvsp) and str(hgvsp).strip() not in ("", ".", "-"):
        return str(hgvsp).strip()
    if pd.notna(hgvsc) and str(hgvsc).strip() not in ("", ".", "-"):
        return str(hgvsc).strip()
    # last resort: chr_pos_ref_alt
    return f"{row.get('chromosome','?')}_{row.get('position','?')}_{row.get('reference','?')}_{row.get('mutation','?')}"


def build_anndata(participant_df: pd.DataFrame,
                  pid: str,
                  all_waves: list,
                  wave_to_time: dict,
                  missing_strategy: str) -> ad.AnnData:
    """
    Build one AnnData for a single participant from their long-format rows.
    Returns None if the participant should be excluded.
    """

    # Attach mutation key
    participant_df = participant_df.copy()
    participant_df["_mut_key"] = participant_df.apply(make_mutation_key, axis=1)

    # Warn if duplicate (mut_key, wave) — keep first occurrence
    dupes = participant_df.duplicated(subset=["_mut_key", "wave"])
    if dupes.any():
        warnings.warn(
            f"Participant {pid}: {dupes.sum()} duplicate (mutation, wave) rows — "
            "keeping first occurrence.",
            UserWarning,
        )
        participant_df = participant_df[~dupes]

    mutations = sorted(participant_df["_mut_key"].unique())
    n_mut = len(mutations)
    n_waves = len(all_waves)

    # Initialise matrices
    AO_mat = np.zeros((n_mut, n_waves), dtype=np.float32)
    DP_mat = np.zeros((n_mut, n_waves), dtype=np.float32)

    # Fill observed values
    mut_idx = {m: i for i, m in enumerate(mutations)}
    wave_idx = {w: j for j, w in enumerate(all_waves)}

    for _, row in participant_df.iterrows():
        i = mut_idx[row["_mut_key"]]
        j = wave_idx[row["wave"]]
        AO_mat[i, j] = float(row["AO"])
        DP_mat[i, j] = float(row["DP"])

    # Detect which (mut, wave) cells are truly missing vs observed-zero
    observed_mask = np.zeros((n_mut, n_waves), dtype=bool)
    for _, row in participant_df.iterrows():
        observed_mask[mut_idx[row["_mut_key"]], wave_idx[row["wave"]]] = True

    missing_mask = ~observed_mask  # True where data was absent

    if missing_mask.any():
        if missing_strategy == "drop_participant":
            return None

        elif missing_strategy == "zero":
            pass  # AO and DP already 0

        elif missing_strategy == "median_zero":
            # Per-mutation median DP over observed waves
            for i in range(n_mut):
                obs_dp = DP_mat[i, observed_mask[i]]
                med_dp = float(np.median(obs_dp)) if len(obs_dp) > 0 else 1.0
                med_dp = max(med_dp, 1.0)
                DP_mat[i, missing_mask[i]] = med_dp
                # AO stays 0 (mutation undetected at that wave)

        else:
            raise ValueError(f"Unknown MISSING_WAVE_STRATEGY: {missing_strategy!r}")

    # Build obs (mutation metadata) from last observed row per mutation
    obs_rows = (
        participant_df
        .sort_values("wave")
        .drop_duplicates(subset="_mut_key", keep="last")
        .set_index("_mut_key")
    )
    # Keep only scalar metadata columns (drop matrix-valued ones)
    drop_cols = {"AO", "DP", "AF", "wave", "DATA", "FORMAT", "INFO", "FILTER", "QUAL"}
    obs_meta = obs_rows.drop(columns=[c for c in drop_cols if c in obs_rows.columns],
                             errors="ignore")
    obs_meta = obs_meta.loc[mutations]  # ensure correct order

    # Build var (wave metadata)
    time_vals = np.array([wave_to_time[w] for w in all_waves], dtype=np.float32)
    var_df = pd.DataFrame({"time_points": time_vals}, index=[str(w) for w in all_waves])

    # Construct AnnData
    adata = ad.AnnData(
        X=AO_mat / np.maximum(DP_mat, 1.0),   # VAF matrix as X for convenience
        obs=obs_meta,
        var=var_df,
        layers={"AO": AO_mat, "DP": DP_mat},
    )
    adata.uns["participant_id"] = pid

    return adata


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("COHORT PREPROCESSING")
    print("=" * 60)

    # Load and concatenate all input files
    frames = []
    for fpath in INPUT_TSVS:
        print(f"\nLoading {fpath} ...")
        chunk = pd.read_csv(fpath, sep="\t", low_memory=False)
        print(f"  {len(chunk):,} rows, {chunk['participant_id'].nunique()} participants")
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined: {len(df):,} rows, {df['participant_id'].nunique()} participants")

    # Coerce AO / DP to numeric (in case they came in as strings)
    df["AO"] = pd.to_numeric(df["AO"], errors="coerce").fillna(0)
    df["DP"] = pd.to_numeric(df["DP"], errors="coerce").fillna(0)
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")
    df = df.dropna(subset=["wave"])
    df["wave"] = df["wave"].astype(int)

    # Determine wave universe and time mapping
    all_waves = sorted(df["wave"].unique())
    print(f"  Waves found: {all_waves}")

    if WAVE_TO_TIME is None:
        wave_to_time = {w: float(w) for w in all_waves}
        print(f"  Using wave integers as numeric time: {wave_to_time}")
    else:
        missing = set(all_waves) - set(WAVE_TO_TIME.keys())
        if missing:
            raise ValueError(f"WAVE_TO_TIME is missing entries for waves: {missing}")
        wave_to_time = WAVE_TO_TIME
        print(f"  Using custom time mapping: {wave_to_time}")

    # Build one AnnData per participant
    cohort = []
    excluded = []

    participants = df["participant_id"].unique()
    print(f"\nProcessing {len(participants)} participants...")

    for pid in participants:
        pdf = df[df["participant_id"] == pid]
        waves_present = sorted(pdf["wave"].unique())

        if len(waves_present) < MIN_WAVES:
            print(f"  SKIP {pid}: only {len(waves_present)} wave(s) (min={MIN_WAVES})")
            excluded.append((pid, f"fewer than {MIN_WAVES} waves"))
            continue

        adata = build_anndata(pdf, pid, all_waves, wave_to_time, MISSING_WAVE_STRATEGY)

        if adata is None:
            print(f"  SKIP {pid}: excluded by missing-wave strategy")
            excluded.append((pid, "missing wave + drop_participant strategy"))
            continue

        cohort.append(adata)
        print(f"  OK   {pid}: {adata.n_obs} mutations × {adata.n_vars} waves")

    print(f"\nIncluded : {len(cohort)}")
    print(f"Excluded : {len(excluded)}")

    if not cohort:
        print("\nERROR: No participants passed preprocessing. Check your data and config.")
        sys.exit(1)

    # Save
    print(f"\nSaving to {OUTPUT_FILE} ...")
    import os
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(cohort, f, protocol=4)

    # Verify
    with open(OUTPUT_FILE, "rb") as f:
        check = pickle.load(f)
    print(f"Verified: {len(check)} AnnData objects saved.")
    print("\nDone. You can now run the inference pipeline.")


if __name__ == "__main__":
    main()