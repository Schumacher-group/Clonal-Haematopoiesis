import pandas as pd, numpy as np, os, re, pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from anndata import AnnData

LONG_XLSX       = '../data/LongitudinalMDS.xlsx'
CAMERON_XLSX    = '../data/Cameron_Patients.xlsx'
OUT_DIR         = '../exports/MDS'
SOURCE_PRIORITY = {'cameron': 0, 'longitudinal': 1}   # Cameron wins on conflict
CLOSE_DAYS      = 60                                    # near-duplicate diagnostic

# ── parsers ──────────────────────────────────────────────────────────────────
def parse_vaf(x, mode='percent'):        # mode: 'percent' (number=13→0.13) or 'fraction' (Excel %: 0.13→0.13)
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ('', 'null', 'not in panel', 'n/a', 'na'):
        return np.nan                    # not measured
    if s in ('nd', 'not detected'):
        return 0.0                       # measured, absent
    has_pct = '%' in s
    s = s.replace('%', '').replace(',', '.')
    if s.startswith('<'):
        try: return float(s[1:]) / 2 / 100
        except ValueError: return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if has_pct:                          # explicit '%' text → always /100
        return v / 100
    return v / 100 if mode == 'percent' else v


def parse_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.lower() in ('', 'null', 'not in panel', 'nd', 'not detected', 'n/a', 'na'):
        return np.nan
    s = s.replace(',', '.')
    try: return float(s)
    except ValueError: return np.nan

def parse_date(x):                       # handles ISO Excel dates AND dd/mm text
    if pd.isna(x): return pd.NaT
    s = str(x).strip()
    if s in ('', 'NULL'): return pd.NaT
    return pd.to_datetime(s, format='mixed', dayfirst=True, errors='coerce')

def normalise_cdna(x):                   # 'NM_x:c.123A>G'->'c.123A>G'; 'c818G>A'->'c.818G>A'
    if pd.isna(x): return x
    s = str(x).strip()
    if ':' in s: s = s.split(':', 1)[1]
    return re.sub(r'^c(?=\d)', 'c.', s)

# ── loaders ──────────────────────────────────────────────────────────────────
def load_longitudinal(path):
    df = pd.read_excel(path, sheet_name='GeneticData', dtype=str)
    df = df.rename(columns={'SAMPLE_ID': 'participant_id'})
    df['SAMPLE_DATE'] = df['SAMPLE_DATE'].apply(parse_date)
    df['cDNA_CHANGE'] = df['cDNA_CHANGE'].apply(normalise_cdna)
    df['VAF'] = df['VAF'].apply(lambda x: parse_vaf(x, mode='percent'))
    df['READ_DEPTH']  = df['READ_DEPTH'].apply(parse_num)
    df['source']      = 'longitudinal'
    return df[['participant_id','SAMPLE_DATE','GENE','cDNA_CHANGE','PROTEIN_CHANGE',
               'VAF','READ_DEPTH','VARIANT_ASSESSMENT','source']]

def load_cameron(path):
    xls, rows = pd.ExcelFile(path), []
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)

        # find the header row (contains 'Gene' and 'cDNA') — position-independent
        hdr_row = None
        for i in range(len(raw)):
            vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
            if 'gene' in vals and 'cdna' in vals:
                hdr_row = i
                break
        if hdr_row is None:
            print(f"  WARNING: no Gene/cDNA header in sheet '{sheet}', skipping")
            continue

        hdr = raw.iloc[hdr_row].astype(str).str.strip()
        low = hdr.str.lower()
        def col_of(name):
            hits = hdr.index[low == name]
            return hits[0] if len(hits) else None

        gcol, ccol, pcol = col_of('gene'), col_of('cdna'), col_of('protein')
        vaf_cols = list(hdr.index[low == 'vaf'])           # one per timepoint
        date_row = raw.iloc[hdr_row - 1] if hdr_row >= 1 else None
        data     = raw.iloc[hdr_row + 1:]

        for vcol in vaf_cols:
            dcol = vcol + 1                                # depth is right of VAF
            date_val = pd.NaT
            if date_row is not None:                       # date sits above the pair
                for probe in (vcol, vcol - 1, vcol + 1):
                    if 0 <= probe < len(date_row) and pd.notna(date_row[probe]):
                        cand = parse_date(date_row[probe])
                        if pd.notna(cand):
                            date_val = cand
                            break
            for _, r in data.iterrows():
                gene, cdna = r[gcol], r[ccol]
                if pd.isna(gene) and pd.isna(cdna):
                    continue
                rows.append({
                    'participant_id': sheet,               # sheet name = SAMPLE_ID
                    'SAMPLE_DATE': date_val,
                    'GENE': gene,
                    'cDNA_CHANGE': normalise_cdna(cdna),
                    'PROTEIN_CHANGE': r[pcol] if pcol is not None else np.nan,
                    'VAF': parse_vaf(r[vcol], mode='fraction'),
                    'READ_DEPTH': parse_num(r[dcol]),
                    'VARIANT_ASSESSMENT': np.nan,
                    'source': 'cameron',
                })
    return pd.DataFrame(rows)

def load_clinical(path):
    df = pd.read_excel(path, sheet_name='ClinicalData', dtype=str).replace('NULL', np.nan)
    df = df.rename(columns={'SAMPLE_ID': 'participant_id'})
    df['AGE_DIAGNOSIS'] = pd.to_numeric(df['AGE_DIAGNOSIS'], errors='coerce')
    for c in ['DATE_DIAGNOSIS','DATE_DEATH','DATE_TRANSPLANT']:
        df[c] = df[c].apply(parse_date)
    return df

# ── build reconciled long table ──────────────────────────────────────────────
L = load_longitudinal(LONG_XLSX)
C = load_cameron(CAMERON_XLSX)
print(f"loaded: longitudinal rows={len(L)} ids={L['participant_id'].nunique()} | "
      f"cameron rows={len(C)} ids={C['participant_id'].nunique()}")

gen  = pd.concat([L, C], ignore_index=True)
clin = load_clinical(LONG_XLSX)

const = clin.groupby('participant_id').agg(
    SEX=('SEX','first'), AGE_DIAGNOSIS=('AGE_DIAGNOSIS','first'),
    DATE_DIAGNOSIS=('DATE_DIAGNOSIS','first')).reset_index()
gen = gen.merge(const, on='participant_id', how='left')
gen['age'] = gen['AGE_DIAGNOSIS'] + (gen['SAMPLE_DATE'] - gen['DATE_DIAGNOSIS']).dt.days / 365.25

gen['key']      = gen['GENE'].astype(str) + ' ' + gen['cDNA_CHANGE'].astype(str)
gen['src_rank'] = gen['source'].map(SOURCE_PRIORITY)
gen = (gen.sort_values('src_rank')
          .drop_duplicates(['participant_id','SAMPLE_DATE','key'], keep='first'))

gen = gen.dropna(subset=['VAF','READ_DEPTH'])
gen = gen[gen['READ_DEPTH'] > 0]        # keep everything else

print("rows per source :", gen.groupby('source').size().to_dict())
print("ids per source  :", gen.groupby('source')['participant_id'].nunique().to_dict())
print("total unique ids:", gen['participant_id'].nunique())

# ── diagnostic: suspiciously close timepoints (possible overlap mis-merge) ────
print("── near-duplicate timepoint check ──")
for pid, pdat in gen.groupby('participant_id'):
    ds = np.sort(pdat['SAMPLE_DATE'].dropna().unique())
    if len(ds) < 2: continue
    gaps = np.diff(ds).astype('timedelta64[D]').astype(int)
    for i, g in enumerate(gaps):
        if g < CLOSE_DAYS:
            print(f"  {pid}: {pd.Timestamp(ds[i]).date()} & "
                  f"{pd.Timestamp(ds[i+1]).date()} are {g} days apart")

# ── per-participant AnnData ──────────────────────────────────────────────────
sex_map = const.set_index('participant_id')['SEX']
participants = []
for pid, pdat in gen.groupby('participant_id'):
    dates = sorted(pdat['SAMPLE_DATE'].unique())
    muts  = pdat['key'].unique()
    d2i   = {d: i for i, d in enumerate(dates)}
    m2i   = {m: i for i, m in enumerate(muts)}

    AO = np.zeros((len(muts), len(dates)))
    DP = np.zeros((len(muts), len(dates)))
    for _, r in pdat.iterrows():
        i, j = m2i[r['key']], d2i[r['SAMPLE_DATE']]
        AO[i, j] = round(r['VAF'] * r['READ_DEPTH'])   # integer alt counts
        DP[i, j] = r['READ_DEPTH']

    t0      = pd.Timestamp(dates[0])
    elapsed = np.array([(pd.Timestamp(d) - t0).days / 365.25 for d in dates])
    X       = np.where(DP > 0, AO / np.maximum(DP, 1.0), np.nan)   # NaN = unobserved

    obs = (pdat.groupby('key')[['GENE','cDNA_CHANGE','PROTEIN_CHANGE']]
               .first().reindex(muts))
    var = pd.DataFrame({'time_points': elapsed,
                        'sample_date': [pd.Timestamp(d) for d in dates]},
                       index=[f'tp_{i}' for i in range(len(dates))])

    ad = AnnData(X=X, obs=obs, var=var)
    ad.layers['AO'], ad.layers['DP'] = AO, DP
    ad.uns.update(participant_id=pid, cohort='MDS',
                  sex=sex_map.get(pid), elapsed_years=elapsed)
    participants.append(ad)

# ── diagnostic summary table ─────────────────────────────────────────────────
print("\n" + "=" * 78)
print("PER-PARTICIPANT SUMMARY")
print("=" * 78)
print(f"{'participant_id':<16}{'sex':<5}{'n_mut':>6}{'n_tp':>6}"
      f"{'span_yr':>9}{'VAF_min':>9}{'VAF_max':>9}{'DP_min':>9}{'DP_max':>9}")
print("-" * 78)
for ad in participants:
    pid   = ad.uns['participant_id']
    sex   = str(ad.uns.get('sex'))
    n_mut = ad.n_obs
    n_tp  = ad.n_vars
    span  = ad.uns['elapsed_years'][-1] - ad.uns['elapsed_years'][0]
    dp    = ad.layers['DP']
    vaf   = ad.X[dp > 0]
    dpobs = dp[dp > 0]
    vmin  = np.nanmin(vaf) if vaf.size else np.nan
    vmax  = np.nanmax(vaf) if vaf.size else np.nan
    dmin  = dpobs.min() if dpobs.size else np.nan
    dmax  = dpobs.max() if dpobs.size else np.nan
    print(f"{str(pid):<16}{sex:<5}{n_mut:>6}{n_tp:>6}{span:>9.2f}"
          f"{vmin:>9.3f}{vmax:>9.3f}{dmin:>9.0f}{dmax:>9.0f}")
print("=" * 78)

print("\nReconciled table dtypes:")
print(gen.dtypes)
print(f"\nReconciled table shape: {gen.shape}")
print("\nFirst 20 rows:")
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(gen.head(20).to_string(index=False))
    
# ── missing-data (gap) diagnostic ───────────────────────────────────────────
print("\n" + "=" * 78)
print("MISSING-DATA CHECK   (unobserved cell = DP == 0)")
print("=" * 78)

cohort_has_gaps = False
for ad in participants:
    pid         = ad.uns['participant_id']
    DP          = ad.layers['DP']                 # (n_mut, n_tp)
    n_mut, n_tp = DP.shape
    miss        = DP == 0
    n_missing   = int(miss.sum())

    if n_missing == 0:
        print(f"{pid:<14} {n_mut}x{n_tp}   complete")
        continue

    cohort_has_gaps = True
    print(f"{pid:<14} {n_mut}x{n_tp}   {n_missing} missing cell(s):")
    tp_labels = [str(pd.Timestamp(d).date()) for d in ad.var['sample_date']]
    for i in range(n_mut):
        gaps = np.where(miss[i])[0]
        if gaps.size:
            mut  = ad.obs.index[i]
            when = ", ".join(tp_labels[j] for j in gaps)
            print(f"    {mut:<28} missing at: {when}")

print("-" * 78)
print("COHORT VERDICT: " + (
    "GAPS PRESENT  ->  inference must mask/gap-fill unobserved cells"
    if cohort_has_gaps else
    "NO GAPS  ->  matrices complete, het-only-style dense math is safe"))
print("=" * 78)


# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
with open(f'{OUT_DIR}/MDS_cohort_processed.pk', 'wb') as f:
    pickle.dump(participants, f, protocol=4)
print(f"\nProcessed {len(participants)} participants -> {OUT_DIR}/MDS_cohort_processed.pk")

# ── VAF plots ────────────────────────────────────────────────────────────────
os.makedirs(f'{OUT_DIR}/vaf_plots', exist_ok=True)
for ad in participants:
    pid, elapsed = ad.uns['participant_id'], ad.uns['elapsed_years']
    labels, DP = ad.obs.index.tolist(), ad.layers['DP']
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (lab, col) in enumerate(zip(labels, cm.tab10(np.linspace(0, 1, len(labels))))):
        obs = DP[i] > 0
        ax.plot(elapsed[obs], ad.X[i][obs], marker='o', lw=1.8, ms=5, color=col, label=lab)
    ax.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Time (years)'); ax.set_ylabel('VAF')
    ax.set_title(f'Participant {pid}')
    ax.set_ylim(0, max(1.0, np.nanmax(ad.X) * 1.1))
    ax.legend(fontsize=8, frameon=False, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/vaf_plots/{pid}_vaf.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
print(f"VAF plots -> {OUT_DIR}/vaf_plots/")
