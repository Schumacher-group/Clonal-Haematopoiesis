import sys; sys.path.append("..")
import pickle as pk
from src.KI_2 import compute_clonal_models_prob_vec, refine_optimal_model_posterior_vec

with open("../exports/MDS/MDS_cohort_processed.pk","rb") as f:
    cohort = pk.load(f)
by_id = {p.uns['participant_id']: p for p in cohort}

p = by_id['MDS581N49']
p = compute_clonal_models_prob_vec(p)
p = refine_optimal_model_posterior_vec(p)

cols = ['fitness','fitness_railed','homozygosity',
        'homozygosity_unidentified','clonal_index']
print(p.obs[cols].to_string())
print(p.obs[['homozygosity','homozygosity_5','homozygosity_95']].to_string())

