import sys
sys.path.append("..")   # fix to import modules from root

from src.general_imports import *

from src.KI_2 import (
    compute_clonal_models_prob_vec,
    refine_optimal_model_posterior_vec,
)

import pickle as pk
import traceback


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
OUTPUT_FILE = "../exports/MDS/MDS_cohort_fitted.pk"

S_RESOLUTION = 20
H_RESOLUTION = 4        # compare: 4^n_clones stays sane

REFINE_S = 60           # fitness marginal is smooth; 60 is plenty
REFINE_H = 6            # refine: 6^n_clones tractable

MIN_S = 0.01
MAX_S = 3.0

MAX_H = 1.0

FILTER_INVALID = True


# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------

with open(INPUT_FILE, "rb") as f:
    cohort = pk.load(f)

# keep only participants with <= 4 mutations (speed)
_before = len(cohort)
kept = [p for p in cohort if p.shape[0] < 4]
dropped = [p.uns['participant_id'] for p in cohort if p.shape[0] > 4]
cohort = kept
print(f"Filtered {_before} -> {len(cohort)} participants (<=4 mutations)")
print(f"Dropped (>4 mut): {dropped}")


print(f"Loaded {len(cohort)} participants")


# ---------------------------------------------------------------------------
# Run clonal inference
# ---------------------------------------------------------------------------

processed_part_list = []

for i, part in enumerate(cohort):

    print(f"\nParticipant {i + 1} of {len(cohort)}")

    try:
        part.uns["warning"] = None
        part.uns["fit_failed"] = False
        part.uns["fit_failed_reason"] = None

        # -------------------------------------------------------------------
        # Model comparison
        # -------------------------------------------------------------------

        part = compute_clonal_models_prob_vec(
            part,
            s_resolution=S_RESOLUTION,
            h_resolution=H_RESOLUTION,
            min_s=MIN_S,
            max_s=MAX_S,
            max_h=MAX_H,
            filter_invalid=FILTER_INVALID,
            disable_progressbar=False,
        )

        if part.uns.get("warning") is not None:
            print(f"  WARNING after model comparison: {part.uns['warning']}")

        if "model_dict" not in part.uns or len(part.uns["model_dict"]) == 0:
            raise RuntimeError("No valid models found")

        top_model = list(part.uns["model_dict"].values())[0]
        print(f"  Top model: {top_model[0]}")
        print(f"  Top model raw probability: {top_model[1]:.3e}")

        # -------------------------------------------------------------------
        # Posterior refinement
        # -------------------------------------------------------------------

        part = refine_optimal_model_posterior_vec(
            part,
            s_resolution=REFINE_S,
            h_resolution=REFINE_H,
            max_h=MAX_H,
        )

        if part.uns.get("warning") is not None:
            print(f"  WARNING after refinement: {part.uns['warning']}")

        if "fitness" in part.obs.columns:
            print(part.obs[
                [
                    "fitness",
                    "fitness_5",
                    "fitness_95",
                    "homozygosity",
                    "homozygosity_5",
                    "homozygosity_95",
                    "clonal_index",
                ]
            ].to_string())

        processed_part_list.append(part)

        print(f"  -> participant {i + 1} OK")

    except Exception as e:
        print(f"  -> participant {i + 1} FAILED: {e}")
        traceback.print_exc()

        part.uns["fit_failed"] = True
        part.uns["fit_failed_reason"] = str(e)

        processed_part_list.append(part)


# ---------------------------------------------------------------------------
# Save fitted cohort
# ---------------------------------------------------------------------------

with open(OUTPUT_FILE, "wb") as f:
    pk.dump(processed_part_list, f, protocol=4)

print(f"\nSaved {len(processed_part_list)} participants -> {OUTPUT_FILE}")
