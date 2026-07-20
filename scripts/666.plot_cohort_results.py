import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = '../exports/MDS/MDS_cohort_fitted.pk'
OUTPUT_DIR  = '../exports/figures/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Shared: VAF-over-time panel ────────────────────────────────────────────────
def get_vaf(part):
    AO = part.layers['AO'].T
    DP = part.layers['DP'].T
    return AO / np.maximum(DP, 1.0), DP


def plot_vaf_panel(ax, part, cs, ms, colours):
    vaf_ratio, DP = get_vaf(part)
    time_points = np.array(part.var.time_points)

    for clone_idx, clone_muts in enumerate(cs):
        col = colours[clone_idx % len(colours)]
        clone_label = ', '.join(ms[clone_idx]) if ms else f'Clone {clone_idx}'

        clone_vafs = vaf_ratio[:, clone_muts]
        lead_idx   = int(np.argmax(clone_vafs.sum(axis=0)))
        lead_mut   = clone_muts[lead_idx]

        valid = DP[:, lead_mut] > 0
        ax.plot(time_points[valid], vaf_ratio[valid, lead_mut] * 100,
                'o-', color=col, lw=1.8, ms=5, label=clone_label)

        for m in clone_muts:
            if m == lead_mut:
                continue
            ok = DP[:, m] > 0
            ax.plot(time_points[ok], vaf_ratio[ok, m] * 100,
                    's--', color=col, alpha=0.45, lw=1, ms=3)

    ax.set_title('VAF over time', fontsize=9)
    ax.set_xlabel('Time (years)', fontsize=8)
    ax.set_ylabel('VAF (%)', fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, framealpha=0.6, title='Clone', title_fontsize=7,
              loc='upper left', bbox_to_anchor=(1.02, 1))


# ── Shared: normalisation / CI helpers ─────────────────────────────────────────
def _normalise_density(post, grid):
    post = np.nan_to_num(np.asarray(post, float), nan=0.0, posinf=0.0, neginf=0.0)
    area = np.trapz(post, grid)
    return post / area if area > 0 else post


def _map_ci_from_density(post_density, grid, lo=0.05, hi=0.95):
    if post_density.sum() <= 0:
        return float('nan'), (float('nan'), float('nan'))
    dx  = grid[1] - grid[0]
    cdf = np.cumsum(post_density) * dx
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    i_lo = int(np.searchsorted(cdf, lo))
    i_hi = int(min(np.searchsorted(cdf, hi), len(grid) - 1))
    return float(grid[np.argmax(post_density)]), (float(grid[i_lo]), float(grid[i_hi]))


def _draw_s_panel(ax, s_range, s_post, col, clone_idx):
    s_post = _normalise_density(s_post, s_range)
    s_map, s_ci = _map_ci_from_density(s_post, s_range)
    ax.fill_between(s_range, s_post, alpha=0.25, color=col)
    ax.plot(s_range, s_post, color=col, lw=1.5)
    ax.axvline(s_map, color=col, ls='--', lw=1.5, label=f'MAP = {s_map:.3f}')
    ax.axvspan(s_ci[0], s_ci[1], alpha=0.12, color=col,
               label=f'90% CI [{s_ci[0]:.2f}, {s_ci[1]:.2f}]')
    ax.set_title(f'Fitness — clone {clone_idx}', fontsize=9)
    ax.set_xlabel('Selection coefficient s', fontsize=8)
    ax.set_ylabel('Posterior density', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)


def _draw_h_panel(ax, h_range, h_post, col, clone_idx):
    tot = np.nansum(h_post)
    h_post = np.nan_to_num(h_post) / tot if tot > 0 else np.nan_to_num(h_post)
    # CI over discrete grid
    if h_post.sum() > 0:
        c = np.cumsum(h_post)
        h_map = float(h_range[np.argmax(h_post)])
        h_ci  = (float(h_range[np.searchsorted(c, 0.05)]),
                 float(h_range[min(np.searchsorted(c, 0.95), len(h_range) - 1)]))
    else:
        h_map, h_ci = float('nan'), (float('nan'), float('nan'))
    ax.fill_between(h_range, h_post, alpha=0.25, color=col)
    ax.plot(h_range, h_post, color=col, lw=1.5)
    ax.axvline(h_map, color=col, ls='--', lw=1.5, label=f'MAP = {h_map:.2f}')
    ax.axvspan(h_ci[0], h_ci[1], alpha=0.12, color=col,
               label=f'90% CI [{h_ci[0]:.2f}, {h_ci[1]:.2f}]')
    ax.axvline(0.0, color='grey', lw=0.8, ls=':', alpha=0.7)
    ax.axvline(1.0, color='grey', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0\n(het)', '.25', '.5', '.75', '1\n(hom)'], fontsize=7)
    ax.set_title(f'Zygosity — clone {clone_idx}', fontsize=9)
    ax.set_xlabel('Homozygous fraction h', fontsize=8)
    ax.set_ylabel('Posterior density', fontsize=8)
    ax.legend(fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)


# ── KI_3 coupled per-clone-h layout ────────────────────────────────────────────
def plot_coupled(part, participant_id, figsize=(16, 5)):
    model  = part.uns['optimal_model']
    cs     = model['clonal_structure']
    ms     = model['mutation_structure']
    joint  = model['joint_inference']
    gH     = model['global_H']
    K = len(cs)

    fig = plt.figure(figsize=(figsize[0], max(figsize[1] * K, 6)))
    gs  = gridspec.GridSpec(K + 1, 3, figure=fig, width_ratios=[1.4, 1, 1])
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig.suptitle(
        f'Participant {participant_id} — coupled per-clone h   '
        f'(global H = {gH["map"]:.2f} '
        f'[{gH["ci"][0]:.2f}, {gH["ci"][1]:.2f}])',
        fontsize=11, fontweight='bold', y=0.99)

    ax_vaf = fig.add_subplot(gs[0:K, 0])
    plot_vaf_panel(ax_vaf, part, cs, ms, colours)

    for k in range(K):
        col = colours[k % len(colours)]
        r   = joint[k]
        _draw_s_panel(fig.add_subplot(gs[k, 1]),
                      np.asarray(r['s_range']), np.asarray(r['s_posterior']),
                      col, k)
        _draw_h_panel(fig.add_subplot(gs[k, 2]),
                      np.asarray(r['h_range']), np.asarray(r['h_posterior']),
                      col, k)

    # global-H posterior spanning bottom middle+right
    axH = fig.add_subplot(gs[K, 1:3])
    Hg  = np.asarray(gH['grid']); Hp = np.nan_to_num(np.asarray(gH['posterior']))
    tot = Hp.sum(); Hp = Hp / tot if tot > 0 else Hp
    axH.fill_between(Hg, Hp, alpha=0.3, color='#555555')
    axH.plot(Hg, Hp, color='#555555', lw=1.5)
    axH.axvline(gH['map'], color='k', ls='--', lw=1.5, label=f"H MAP = {gH['map']:.2f}")
    axH.axvspan(gH['ci'][0], gH['ci'][1], alpha=0.12, color='#555555',
                label=f"90% CI [{gH['ci'][0]:.2f}, {gH['ci'][1]:.2f}]")
    axH.set_xlim(0, 1)
    axH.set_title('Global patient homozygosity H (size-weighted)', fontsize=9)
    axH.set_xlabel('H', fontsize=8)
    axH.set_ylabel('Posterior density', fontsize=8)
    axH.legend(fontsize=7, framealpha=0.6)
    axH.tick_params(labelsize=7); axH.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── KI_2 marginalised layout (3-D or 2-D posterior) ────────────────────────────
def _marginals(model, clone_idx):
    s_range = np.asarray(model['s_range'], float)
    post    = np.asarray(model['posterior'], float)
    h_range = model.get('h_range', None)

    if post.ndim == 3 and h_range is not None:
        h_range = np.asarray(h_range, float)
        cp = np.nan_to_num(post[:, :, clone_idx], nan=0.0, posinf=0.0, neginf=0.0)
        return (s_range, _normalise_density(cp.sum(axis=0), s_range),
                h_range, cp.sum(axis=1))
    cp = np.nan_to_num(post[:, clone_idx], nan=0.0, posinf=0.0, neginf=0.0)
    return s_range, _normalise_density(cp, s_range), None, None


def plot_optimal_model_marginal(part, participant_id, figsize=(16, 5)):
    model = part.uns['optimal_model']
    cs    = model['clonal_structure']
    ms    = model.get('mutation_structure', None)

    has_h = (np.asarray(model['posterior']).ndim == 3
             and model.get('h_range', None) is not None)
    n_cols = 3 if has_h else 2
    K = max(len(cs), 1)

    fig = plt.figure(figsize=(figsize[0] if has_h else figsize[0]*0.75,
                              max(figsize[1] * K, 6)))
    width_ratios = [1.4, 1, 1] if has_h else [1.4, 1]
    gs = gridspec.GridSpec(K, n_cols, figure=fig, width_ratios=width_ratios)
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

    kind = 'mixed (s + h)' if has_h else 'basic (s only)'
    fig.suptitle(f'Participant {participant_id} — {kind}',
                 fontsize=11, fontweight='bold', y=0.98)

    plot_vaf_panel(fig.add_subplot(gs[0:K, 0]), part, cs, ms, colours)

    for k in range(K):
        col = colours[k % len(colours)]
        s_range, s_post, h_range, h_post = _marginals(model, k)
        _draw_s_panel(fig.add_subplot(gs[k, 1]), s_range, s_post, col, k)
        if has_h and h_range is not None:
            _draw_h_panel(fig.add_subplot(gs[k, 2]), h_range, h_post, col, k)

    plt.tight_layout()
    return fig


# ── Dispatcher ─────────────────────────────────────────────────────────────────
def plot_optimal_model_full(part, participant_id=None, figsize=(16, 5)):
    if part.uns.get('warning') is not None:
        print(f'  WARNING: {part.uns["warning"]}')

    model = part.uns['optimal_model']
    pid   = participant_id or 'Participant'

    if 'joint_inference' in model:
        print('  Detected pipeline: coupled per-clone h (KI_3)')
        return plot_coupled(part, pid, figsize)

    print('  Detected pipeline: marginalised posterior (KI_2)')
    return plot_optimal_model_marginal(part, pid, figsize)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading fitted cohort from {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'rb') as f:
            cohort = pk.load(f)
        print(f"Loaded {len(cohort)} participants")
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found. Run the inference pipeline first.")
        return

    for i, part in enumerate(cohort):
        participant_id = part.uns.get('participant_id', f'participant_{i+1}')
        print(f"\nPlotting {participant_id}...")

        if 'optimal_model' not in part.uns:
            print("  SKIP: no optimal_model (fit may have failed).")
            continue

        try:
            fig = plot_optimal_model_full(part, participant_id=participant_id)
            out_path = os.path.join(OUTPUT_DIR, f'{participant_id}_posteriors.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved to {out_path}")
        except Exception as e:
            print(f"  ERROR plotting {participant_id}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"\nDone. Figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
