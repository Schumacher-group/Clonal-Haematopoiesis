import pandas as pd
import numpy as np

CSV_FILE = '../data/MDS_cohort_combined.csv'

df = pd.read_csv(CSV_FILE, sep=';', parse_dates=['SAMPLE_DATE'], dayfirst=False)

# VAF is stored as a percentage (0-100), convert to fraction
df['VAF_frac'] = pd.to_numeric(df['VAF'], errors='coerce') / 100.0

records = []

for patient_id, pat in df.groupby('SAMPLE_ID'):

    # one date per visit (all mutations on the same visit share a date)
    visit_dates = (
        pat.drop_duplicates('VISIT_NUMBER')
           .sort_values('VISIT_NUMBER')[['VISIT_NUMBER', 'SAMPLE_DATE']]
           .reset_index(drop=True)
    )

    dates = visit_dates['SAMPLE_DATE'].dropna()
    if len(dates) < 1:
        continue

    # time points in years relative to first visit
    t0 = dates.iloc[0]
    time_points_years = [(d - t0).days / 365.25 for d in dates]

    # delta_t between consecutive visits
    if len(dates) >= 2:
        deltas = [(dates.iloc[i+1] - dates.iloc[i]).days / 365.25
                  for i in range(len(dates) - 1)]
        mean_delta_t = float(np.mean(deltas))
        min_delta_t  = float(np.min(deltas))
        max_delta_t  = float(np.max(deltas))
    else:
        deltas = []
        mean_delta_t = min_delta_t = max_delta_t = np.nan

    # first-timepoint VAF: mutations with a valid VAF at visit 1
    first_visit = pat['VISIT_NUMBER'].min()
    first_vafs = pat.loc[
        (pat['VISIT_NUMBER'] == first_visit) & pat['VAF_frac'].notna(),
        'VAF_frac'
    ]

    mean_first_vaf = float(first_vafs.mean()) if len(first_vafs) > 0 else np.nan
    min_first_vaf  = float(first_vafs.min())  if len(first_vafs) > 0 else np.nan
    max_first_vaf  = float(first_vafs.max())  if len(first_vafs) > 0 else np.nan

    records.append(dict(
        patient_id     = patient_id,
        n_visits       = len(dates),
        first_date     = t0.date(),
        last_date      = dates.iloc[-1].date(),
        time_points_yr = [round(t, 2) for t in time_points_years],
        mean_delta_t   = round(mean_delta_t, 3) if not np.isnan(mean_delta_t) else np.nan,
        min_delta_t    = round(min_delta_t,  3) if not np.isnan(min_delta_t)  else np.nan,
        max_delta_t    = round(max_delta_t,  3) if not np.isnan(max_delta_t)  else np.nan,
        n_mutations_t0 = len(first_vafs),
        mean_first_vaf = round(mean_first_vaf, 4) if not np.isnan(mean_first_vaf) else np.nan,
        min_first_vaf  = round(min_first_vaf,  4) if not np.isnan(min_first_vaf)  else np.nan,
        max_first_vaf  = round(max_first_vaf,  4) if not np.isnan(max_first_vaf)  else np.nan,
    ))

summary = pd.DataFrame(records)

print("=== Per-patient summary ===")
print(summary.to_string(index=False))

ok = summary.dropna(subset=['mean_first_vaf', 'mean_delta_t'])

print("\n=== Cohort-level summary ===")
print(f"Patients:                    {len(summary)}")
print(f"Mean first-tp VAF:           {ok['mean_first_vaf'].mean():.3f}  (SD {ok['mean_first_vaf'].std():.3f})")
print(f"Median first-tp VAF:         {ok['mean_first_vaf'].median():.3f}")
print(f"Mean delta_t (years):        {ok['mean_delta_t'].mean():.2f}  (SD {ok['mean_delta_t'].std():.2f})")
print(f"Median delta_t (years):      {ok['mean_delta_t'].median():.2f}")
print(f"Range delta_t:               {ok['min_delta_t'].min():.2f} – {ok['max_delta_t'].max():.2f}")