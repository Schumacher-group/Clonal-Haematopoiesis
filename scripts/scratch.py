import sys
sys.path.append("..")
import pickle as pk, numpy as np
from src.KI_2 import compute_clonal_models_prob_vec, refine_optimal_model_posterior_vec, clone_posteriors

with open("../exports/MDS/MDS_cohort_processed.pk", "rb") as f:
    cohort = pk.load(f)
by_id = {p.uns['participant_id']: p for p in cohort}
part = by_id['MDS581N49']

part = compute_clonal_models_prob_vec(part)          # current corrected code
part = refine_optimal_model_posterior_vec(part)

cs   = part.uns['optimal_model']['clonal_structure']
s    = np.array(part.uns['optimal_model']['s_range'])
post = clone_posteriors(part.uns['optimal_model']['posterior'],
                        part.uns['optimal_model']['h_combos'], s,
                        [np.array(g) for g in part.uns['optimal_model']['h_grids']])

print("winning structure:", [[list(part.obs.index)[j] for j in c] for c in cs])
for i, c in enumerate(cs):
    _, joint = post[i]
    ps = joint.sum(0); ps /= ps.sum()
    genes = ", ".join(list(part.obs.index)[j] for j in c)
    print(f"\nclone {i}: {genes}")
    print(f"  MAP s      = {s[np.argmax(ps)]:.3f}")
    print(f"  p(s) peak-region:", np.round(ps[::max(1,len(s)//10)], 3))
