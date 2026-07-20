import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    mds_participant_list = pickle.load(f)

output_dir = '../exports/MDS_VAF_over_time'
os.makedirs(output_dir, exist_ok=True)

LABEL_SIZE  = 18
TITLE_SIZE  = 20
TICK_SIZE   = 14
LEGEND_SIZE = 13

fig, ax = plt.subplots(figsize=(12, 7))
sns.set_style("whitegrid")

palette = sns.color_palette("tab20", n_colors=len(mds_participant_list))

for p_idx, adata in enumerate(mds_participant_list):
    participant_id = adata.uns.get('participant_id', f'P{p_idx}')
    AO         = adata.layers['AO']
    DP         = adata.layers['DP']
    mutations  = adata.obs.index.values
    timepoints = adata.var['time_points'].values
    color      = palette[p_idx]

    for mut_idx, mut in enumerate(mutations):
        vaf   = AO[mut_idx, :] / DP[mut_idx, :]
        above = vaf >= 0.5
        label = f'{participant_id} – {mut}' if mut_idx == 0 else None

        # Grey out trajectories that never exceed 0.5
        if not any(above):
            ax.plot(timepoints, vaf, marker='o', color='lightgrey',
                    alpha=0.5, linewidth=1.0, zorder=1, label=None)
        else:
            ax.plot(timepoints, vaf, marker='o', color=color,
                    alpha=0.85, linewidth=1.5, zorder=2, label=label)

ax.axhline(0.5, color='grey', linestyle='--', linewidth=1.0, alpha=0.6)
ax.set_xlabel('Time (years)', fontsize=LABEL_SIZE)
ax.set_ylabel('VAF', fontsize=LABEL_SIZE)
ax.set_title('VAF of All Mutations Over Time – Full Cohort', fontsize=TITLE_SIZE)
ax.set_ylim(0, 1)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.legend(loc='upper left', fontsize=LEGEND_SIZE,
          bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=True)

plt.tight_layout()
out_path = os.path.join(output_dir, 'cohort_vaf_over_time.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f'Cohort plot saved to {out_path}')