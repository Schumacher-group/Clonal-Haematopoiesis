"""
KI_3: coupled per-clone homozygosity inference.

Each clone k has its own homozygous fraction h_k. Clones share one population:
    N_tot(t) = N_w / (1 - sum_k 2 V_k(t)/(1+h_k))
so summed VAF saturates at (1+H)/2 with the SIZE-WEIGHTED global
    H = sum_k h_k x_k / sum_k x_k.
The ensemble constraint H >= 2S-1 is enforced automatically by feasibility
(N_tot > 0), WITHOUT flooring any individual clone. A clone only gets its own
floor when its OWN VAF > 0.5 (h_k >= 2 V_k - 1).

Grids the h-vector; per h-vector each clone's fitness runs KI_2's log-space HMM
and factorises. Cost ~ h_res^K * (s_res forward passes). Guarded for large K.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy.stats as jsp_stats
from itertools import product
from scipy.special import logsumexp   # add at top of KI_3.py

from src.KI_2 import (
    key, X_MAX,
    fitness_specific_computations,     # reused: per-clone L(s) in log-space
    compute_h_min,
    find_valid_clonal_structures,
)

N_W = 1e5


# ---------------------------------------------------------------------------
# coupled background + per-mutation deterministic sizes (per-clone h)
# ---------------------------------------------------------------------------
def compute_coupled_sizes(cs, AO, DP, h_clone):
    """h_clone: (K,) per-clone homozygous fraction.
    Returns total_cells (T,), det_size (T,M), h_mut (M,), feasible (bool),
            H_peak (global weighted H at peak-summed-VAF timepoint)."""
    M   = AO.shape[1]
    vaf = AO / DP                                     # (T, M)

    h_mut = jnp.zeros(M)
    lead  = []
    for k, idx in enumerate(cs):
        lead.append(idx[int(jnp.argmax(vaf[:, idx].sum(0)))])
        h_mut = h_mut.at[jnp.array(idx)].set(h_clone[k])

    V_lead = vaf[:, jnp.array(lead)]                  # (T, K)
    A_t    = jnp.sum(2.0 * V_lead / (1.0 + h_clone[None, :]), axis=1)  # (T,)  = 2*sum V/(1+h)
    denom  = 1.0 - A_t
    feasible = bool(jnp.all(denom > 1e-6))
    total_cells = N_W / jnp.clip(denom, 1e-6, None)   # (T,)

    det_size = 2.0 * total_cells[:, None] * vaf / (1.0 + h_mut[None, :])  # (T,M)
    det_size = jnp.clip(det_size, 1.0, X_MAX)

    # global weighted H at the peak-summed-VAF timepoint (most constrained)
    t_star = int(jnp.argmax(V_lead.sum(1)))
    Vk     = V_lead[t_star]                           # (K,)
    a      = Vk / (1.0 + h_clone)                     # per clone V/(1+h)
    b      = h_clone * a
    H_peak = float(jnp.sum(b) / jnp.clip(jnp.sum(a), 1e-12, None))

    return total_cells, det_size, h_mut, feasible, H_peak


# ---------------------------------------------------------------------------
# global HMM variables with PER-MUTATION h (mirrors KI_2 shapes exactly)
# ---------------------------------------------------------------------------
def compute_global_variables_h(s_vec, AO, DP, total_cells, det_size,
                               time_points, h_mut, resolution=500):
    M  = AO.shape[1]
    dt = jnp.diff(time_points)
    exp_s = jnp.exp(dt * s_vec[:, None])
    exp_s = jnp.reshape(exp_s, (*exp_s.shape, 1, 1))          # (s_res, T-1, 1, 1)

    beta = jrnd.beta(key, a=(AO + 1)[:, :, None],
                     b=(DP - AO + 1)[:, :, None],
                     shape=(AO.shape[0], M, resolution))
    beta = jnp.sort(beta)

    hm   = h_mut[None, :, None]                               # (1, M, 1)
    pole = (1.0 + hm) / 2.0
    beta = jnp.minimum(beta, pole - 1e-6)

    N_w_cond = (total_cells[:, None] - det_size)[:, :, None]  # (T, M, 1)
    x = jnp.clip(jnp.ceil(-N_w_cond * beta / (beta - pole)), 1.0, X_MAX)
    true_vaf = (1.0 + hm) * x / (2.0 * (N_w_cond + x))

    p_y = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf)
    rec = p_y[0, :, :] * (1.0 / resolution)
    return x, exp_s, rec, p_y, M


# ---------------------------------------------------------------------------
# per-clone likelihoods L_k(s ; h-vector) for ONE feasible h-vector
# ---------------------------------------------------------------------------
def clone_lls_for_hvec(s_vec, AO, DP, time_points, cs, h_clone, resolution=500):
    tot, det, h_mut, feasible, H_peak = compute_coupled_sizes(cs, AO, DP, h_clone)
    if not feasible:
        return None, None
    x, exp_s, rec, p_y, M = compute_global_variables_h(
        s_vec, AO, DP, tot, det, time_points, h_mut, resolution)
    s_idx = jnp.arange(s_vec.shape[0])
    ll = jax.vmap(fitness_specific_computations,
                  in_axes=(0, None, None, None, None, None, None, None, None))(
        s_idx, s_vec, x, exp_s, rec, p_y, time_points, M, cs)   # (s_res, K)
    return np.nan_to_num(np.asarray(ll), nan=0.0, posinf=0.0, neginf=0.0), H_peak


# ---------------------------------------------------------------------------
# full per-clone-h inference for ONE clonal structure
# ---------------------------------------------------------------------------

def infer_structure(AO, DP, time_points, cs, s_vec,
                    h_res=15, h_min_quantile=0.05, max_hvec=20000,
                    resolution=500, H_bins=50):
    K = len(cs)
    _, h_min_clone, _ = compute_h_min(cs, AO, DP, quantile=h_min_quantile)
    h_grids = [np.linspace(float(np.clip(h_min_clone[k], 0.0, 1.0)), 1.0, h_res)
               for k in range(K)]

    if h_res ** K > max_hvec:
        raise RuntimeError(f"{h_res**K} h-vectors > cap {max_hvec}; "
                           f"reduce h_res or restrict K.")

    s_res    = s_vec.shape[0]
    log_marg = [np.full((s_res, h_res), -np.inf) for _ in range(K)]   # log P(s_k,h_k)
    H_grid   = np.linspace(0.0, 1.0, H_bins)
    log_H    = np.full(H_bins, -np.inf)

    s_prior     = 1.0 / (float(s_vec[-1]) - float(s_vec[0]))
    log_h_pref  = K * np.log(1.0 / (h_res - 1)) if h_res > 1 else 0.0
    TINY        = 1e-300

    log_evidence = -np.inf
    for h_idx in product(range(h_res), repeat=K):
        h_clone = jnp.array([h_grids[k][h_idx[k]] for k in range(K)])
        ll, H_peak = clone_lls_for_hvec(s_vec, AO, DP, time_points, cs,
                                        h_clone, resolution)
        if ll is None:
            continue                                        # infeasible -> 0

        ll = np.clip(np.nan_to_num(ll, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        Z  = np.trapz(ll, s_vec, axis=0) * s_prior          # (K,)
        if not np.all(np.isfinite(Z)) or np.any(Z <= 0):
            continue

        log_Z     = np.log(Z)                               # (K,)
        log_prodZ = log_Z.sum()

        log_evidence = np.logaddexp(log_evidence, log_prodZ + log_h_pref)

        log_ll = np.log(np.clip(ll, TINY, None))            # (s_res, K)
        for k in range(K):
            col = log_ll[:, k] + (log_prodZ - log_Z[k])     # Π_{j≠k} Z_j in log
            log_marg[k][:, h_idx[k]] = np.logaddexp(log_marg[k][:, h_idx[k]], col)

        b = int(np.clip(round(H_peak * (H_bins - 1)), 0, H_bins - 1))
        log_H[b] = np.logaddexp(log_H[b], log_prodZ)

    # back to linear with a per-array offset (posteriors are normalised later)
    marg = []
    for k in range(K):
        m = log_marg[k].max()
        marg.append(np.exp(log_marg[k] - m) if np.isfinite(m)
                    else np.zeros_like(log_marg[k]))
    mH     = log_H.max()
    H_post = np.exp(log_H - mH) if np.isfinite(mH) else np.zeros_like(log_H)

    return dict(marg=marg, s_vec=np.asarray(s_vec), h_grids=h_grids,
                H_grid=H_grid, H_post=H_post,
                log_evidence=float(log_evidence), cs=cs)


# ---------------------------------------------------------------------------
# summarise -> per-clone joint_inference dicts (+ global H)
# ---------------------------------------------------------------------------
def _ci(grid, p):
    p = np.nan_to_num(p); s = p.sum()
    if s <= 0:
        return float('nan'), (float('nan'), float('nan'))
    p = p / s
    c = np.cumsum(p)
    lo = float(grid[np.searchsorted(c, 0.05)])
    hi = float(grid[min(np.searchsorted(c, 0.95), len(grid) - 1)])
    return float(grid[np.argmax(p)]), (lo, hi)


def summarise(res):
    joint = []
    for k, P in enumerate(res['marg']):
        P = np.nan_to_num(P)
        s_post = P.sum(1); h_post = P.sum(0)
        s_map, s_ci = _ci(res['s_vec'], s_post)
        h_map, h_ci = _ci(res['h_grids'][k], h_post)
        joint.append(dict(
            s_range=res['s_vec'], s_posterior=s_post, s_map=s_map, s_ci=s_ci,
            h_range=res['h_grids'][k], h_posterior=h_post, h_map=h_map, h_ci=h_ci))
    H_map, H_ci = _ci(res['H_grid'], res['H_post'])
    global_H = dict(grid=res['H_grid'], posterior=res['H_post'],
                    map=H_map, ci=H_ci)
    return joint, global_H


# ---------------------------------------------------------------------------
# model comparison across structures
# ---------------------------------------------------------------------------
def compute_clonal_models_prob_coupled(part, s_resolution=40, h_res=15,
                                       min_s=0.01, max_s=1.5,
                                       h_min_quantile=0.05, resolution=500,
                                       filter_invalid=True, max_hvec=20000,
                                       disable_progressbar=False):
    from tqdm import tqdm
    AO  = jnp.array(part.layers['AO'].T)
    DP  = jnp.array(part.layers['DP'].T)
    tps = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mut = part.shape[0]
    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mut)]]

    part.uns['model_dict'] = {}
    for i, cs in tqdm(list(enumerate(cs_list)), disable=disable_progressbar,
                      total=len(cs_list)):
        try:
            res = infer_structure(AO, DP, tps, cs, s_vec, h_res=h_res,
                                  h_min_quantile=h_min_quantile,
                                  resolution=resolution, max_hvec=max_hvec)
            prob = res['log_evidence']

        except RuntimeError as e:
            part.uns['warning'] = str(e)
            prob = 0.0
        part.uns['model_dict'][f'model_{i}'] = (cs, prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(
        part.uns['model_dict'].items(), key=lambda kv: kv[1][1], reverse=True)}
    return part


# ---------------------------------------------------------------------------
# refine winning structure -> store joint_inference + obs fields
# ---------------------------------------------------------------------------
def refine_optimal_model_coupled(part, s_resolution=80, h_res=25,
                                 min_s=0.01, max_s=1.5,
                                 h_min_quantile=0.05, resolution=500,
                                 max_hvec=200000):
    cs  = list(part.uns['model_dict'].values())[0][0]
    AO  = jnp.array(part.layers['AO'].T)
    DP  = jnp.array(part.layers['DP'].T)
    tps = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    res = infer_structure(AO, DP, tps, cs, s_vec, h_res=h_res,
                          h_min_quantile=h_min_quantile,
                          resolution=resolution, max_hvec=max_hvec)
    joint, global_H = summarise(res)

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[idx].index) for idx in cs],
        'joint_inference': joint,
        'global_H': global_H,
        's_range': res['s_vec'],
    }

    n = part.shape[0]
    fit  = np.zeros(n); fit5 = np.zeros(n); fit95 = np.zeros(n)
    hz   = np.zeros(n); hz5  = np.zeros(n); hz95  = np.zeros(n)
    cidx = np.zeros(n)
    for k, idx in enumerate(cs):
        r = joint[k]
        for m in idx:
            fit[m], fit5[m], fit95[m] = r['s_map'], r['s_ci'][0], r['s_ci'][1]
            hz[m],  hz5[m],  hz95[m]  = r['h_map'], r['h_ci'][0], r['h_ci'][1]
            cidx[m] = k
    part.obs['fitness']         = fit
    part.obs['fitness_5']       = fit5
    part.obs['fitness_95']      = fit95
    part.obs['homozygosity']    = hz
    part.obs['homozygosity_5']  = hz5
    part.obs['homozygosity_95'] = hz95
    part.obs['clonal_index']    = cidx
    part.obs['global_H']        = global_H['map']
    return part
