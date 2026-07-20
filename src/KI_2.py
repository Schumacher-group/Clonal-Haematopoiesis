import sys
sys.path.append("..")
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations
import itertools
from jax.scipy.special import logsumexp

"""
============================================================================
Mixed-zygosity clonal fitness inference  (LOG-SPACE likelihood)
============================================================================
Generalises the heterozygous-only model to continuous, per-clone homozygous
fraction h_i in [0, 1].

Per clone i:
    VAF_i = (1 + h_i) * x_i / (2 * T)        # h=0 heterozygous, h=1 homozygous
    T     = N_w / (1 - 2 * Σ_i V_lead_i/(1+h_i))

Feasibility (total cells T > 0 at EVERY timepoint) -- the exact generalisation
of the single-clone floor h_min = 2V - 1:
    Σ_i V_i,t / (1 + h_i) < 1/2

Each clone's h_i is gridded independently (floored at max(0, 2*Vmax_i - 1)) and
the JOINT grid is masked to this coupled frontier. Setting every h_i = 0 exactly
recovers the original het-only maths.

Missing data: any (mutation, timepoint) with DP == 0 is treated as unobserved.
Emissions there are neutralised (log 1 = 0) and the integration grid is carried
from the nearest observed timepoint, so the hidden BD state propagates across gaps.

The forward HMM recursion is computed in LOG SPACE (logpmf + log-sum-exp /
log-trapezoid) to avoid float64 underflow to exactly 0 at high VAF (~0.5) and
high read depth, where clone sizes reach 1e5-1e6 cells.

Output flags (per mutation, in part.obs):
    fitness_railed              MAP s sits on the max_s ceiling -> report as lower bound
    homozygosity_railed         MAP h sits on the max_h ceiling (real full-LOH state)
    homozygosity_unidentified   h 90% CI wider than H_CI_WIDTH -> do NOT read as LOH
============================================================================
"""

key = jrnd.PRNGKey(758493)
N_w = 1e5


# ============================================================================
# region  Helpers (log-space + missing-data)
# ============================================================================

def log_trapz(log_y, x, axis=-1):
    """Stable log( trapezoid(exp(log_y), x) ) along `axis`."""
    m = jnp.max(log_y, axis=axis, keepdims=True)
    m = jnp.where(jnp.isfinite(m), m, 0.0)          # guard all -inf slices
    val = jsp.integrate.trapezoid(jnp.exp(log_y - m), x=x, axis=axis)
    return jnp.squeeze(m, axis=axis) + jnp.log(val)


def _fill_nearest(v, obs):
    """Fill unobserved entries of a 1D array from nearest observed (ffill then bfill)."""
    v = np.array(v, dtype=float)
    obs = np.array(obs, dtype=bool)
    last = np.nan
    for t in range(len(v)):                       # forward fill
        if obs[t]:
            last = v[t]
        else:
            v[t] = last
    nxt = np.nan
    for t in range(len(v) - 1, -1, -1):           # back fill leading gaps
        if obs[t]:
            nxt = v[t]
        elif np.isnan(v[t]):
            v[t] = nxt
    return v


def compute_carry_idx(observed):
    """For each (timepoint, mutation) give the nearest observed timepoint index."""
    observed = np.array(observed, dtype=bool)
    n_tp, n_mut = observed.shape
    carry = np.full((n_tp, n_mut), -1, dtype=int)
    for j in range(n_mut):
        last = -1
        for t in range(n_tp):                     # forward
            if observed[t, j]:
                last = t
            carry[t, j] = last
        nxt = -1
        for t in range(n_tp - 1, -1, -1):         # backward for leading gaps
            if observed[t, j]:
                nxt = t
            if carry[t, j] == -1:
                carry[t, j] = nxt
    return carry


def _get_arrays(part):
    """Derive everything the core needs straight from the AO/DP layers.
       Returned arrays have the timepoint axis first: (n_tp, n_mut)."""
    AO = np.array(part.layers['AO'].T)            # (n_tp, n_mut)
    DP = np.array(part.layers['DP'].T)
    observed = DP > 0
    carry_idx = compute_carry_idx(observed)
    time_points = np.array(part.var['time_points'], dtype=float)
    if 'h_fixed' in part.obs.columns:
        h_fixed = np.array(part.obs['h_fixed'].values, dtype=float)
    else:
        h_fixed = np.full(part.shape[0], np.nan)  # all clones free
    return (jnp.array(AO), jnp.array(DP), jnp.array(observed),
            jnp.array(carry_idx), jnp.array(time_points), h_fixed)

# endregion


# ============================================================================
# region  Deterministic clone size (per-clone h, missing-aware)
# ============================================================================

def compute_deterministic_size(cs, AO, DP, n_mutations, h, observed=None):
    """h : (n_clones,) homozygous fraction per clone.
       Returns deterministic_size, total_cells, clonal_map, h_mut."""
    AO = np.array(AO)
    DP = np.array(DP)
    if observed is None:
        observed = DP > 0
    observed = np.array(observed, dtype=bool)
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)   # (n_tp, n_mut)

    # leading mutation per clone (from observed points only)
    lm = []
    clonal_map = np.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        summed = np.nansum(vaf[:, cs_idx], axis=0)
        lm.append(cs_idx[int(np.argmax(summed))])
        clonal_map[np.array(cs_idx)] = i

    # per-clone leading VAF, gap-filled -> drives total_cells
    v_lead = np.stack([_fill_nearest(vaf[:, lead], observed[:, lead])
                       for lead in lm], axis=1)                        # (n_tp, K)

    h = np.array(h, dtype=float)
    denom = 1.0 - 2.0 * np.sum(v_lead / (1.0 + h)[None, :], axis=1)    # (n_tp,)
    total_cells = np.ceil(N_w / denom)            # inf/negative if infeasible

    h_mut = h[clonal_map]                                             # (n_mut,)

    # per-mutation size (gap-filled VAF) with the 1/(1+h) correction
    vaf_filled = np.stack([_fill_nearest(vaf[:, j], observed[:, j])
                           for j in range(n_mutations)], axis=1)
    deterministic_size = 2.0 * vaf_filled * total_cells[:, None] / (1.0 + h_mut)[None, :]

    return (jnp.array(deterministic_size), jnp.array(total_cells),
            jnp.array(clonal_map), jnp.array(h_mut))

# endregion


# ============================================================================
# region  Partition enumeration
# ============================================================================

def partition(collection):
    """Iterable over all partitions of a set (all possible clonal structures)."""
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller

# endregion


# ============================================================================
# region  Vectorised likelihood core  (LOG SPACE)
# ============================================================================

@jit
def compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size,
                             time_points, h_mut, observed, carry_idx,
                             resolution=1_000):
    n_mutations = AO.shape[1]
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    DP_safe = jnp.where(observed, DP, 1.0)
    AO_safe = jnp.where(observed, AO, 0.0)

    beta_p_rvs_vec = jrnd.beta(key=key, a=(AO_safe + 1)[:, :, None],
                               b=(DP_safe - AO_safe + 1)[:, :, None],
                               shape=(AO.shape[0], AO.shape[1], resolution))
    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)

    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    half = (1.0 + h_mut)[None, :, None] / 2.0
    x_vec = jnp.ceil(-N_w_cond_vec * beta_p_rvs_vec / (beta_p_rvs_vec - half))
    valid = x_vec > 0
    x_vec = jnp.where(valid, x_vec, 1.0)
    true_vaf_vec = (1.0 + h_mut)[None, :, None] * x_vec / (2.0 * (N_w_cond_vec + x_vec))

    # log emission; invalid -> -inf
    log_p_y = jnp.where(
        valid,
        jsp_stats.binom.logpmf(AO_safe[:, :, None], n=DP_safe[:, :, None], p=true_vaf_vec),
        -jnp.inf)

    idx = jnp.broadcast_to(carry_idx[:, :, None], x_vec.shape)
    x_vec = jnp.take_along_axis(x_vec, idx, axis=0)

    # neutralise unobserved emissions -> log 1 = 0
    log_p_y = jnp.where(observed[:, :, None], log_p_y, 0.0)

    log_rec_vec = log_p_y[0, :, :] - jnp.log(resolution)     # log init
    return x_vec, exp_term_vec_s, log_rec_vec, log_p_y, n_mutations


@jit
def BD_process_dynamics(s, x_vec, exp_term_vec):
    lamb = 1.3
    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2 * lamb + s) * exp_term_vec * (exp_term_vec - 1) / s
    p_vec = mean_vec / variance_vec                       # neg-binom p
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)   # neg-binom n
    return p_vec, n_vec


@jit
def recursive_term_update(j, log_rec_i, x_i, p_i, n_i, log_p_y_i):
    """Forward HMM update in log space."""
    log_bd = jsp_stats.nbinom.logpmf(x_i[j][:, None], p=p_i[j - 1], n=n_i[j - 1])
    log_inner = log_bd + log_rec_i[None, :]
    return log_p_y_i[j] + log_trapz(log_inner, x_i[j - 1], axis=-1)


def mutation_specific_ll(i, log_rec_vec, x_vec, p_vec, n_vec, log_p_y_vec, n_tps):
    log_rec_i = log_rec_vec[i]
    x_i = x_vec[:, i]; p_i = p_vec[:, i]; n_i = n_vec[:, i]
    log_p_y_i = log_p_y_vec[:, i]
    for j in range(1, n_tps):
        log_rec_i = recursive_term_update(j, log_rec_i, x_i, p_i, n_i, log_p_y_i)
    return log_trapz(log_rec_i, x_i[-1])                     # log-likelihood


def fitness_specific_computations(s_idx, s_vec, x_vec, exp_term_vec_s,
                                  log_rec_vec, log_p_y_vec,
                                  time_points, n_mutations, cs):
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]
    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)

    mutation_loglik = jax.vmap(
        mutation_specific_ll, in_axes=(0, None, None, None, None, None, None))(
            jnp.arange(n_mutations, dtype=int), log_rec_vec, x_vec,
            p_vec, n_vec, log_p_y_vec, time_points.shape[0])

    clonal_loglik = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_loglik = clonal_loglik.at[i].set(          # sum of logs = log product
            jnp.sum(mutation_loglik[jnp.array(c_idx)]))
    return clonal_loglik


def jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs, deterministic_size,
                      total_cells, h_mut, observed, carry_idx):
    """Returns LOG clonal likelihoods, shape (s_res, K)."""
    gv = compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size,
                                  time_points, h_mut, observed, carry_idx)
    x_vec, exp_term_vec_s, log_rec_vec, log_p_y_vec, n_mutations = gv
    s_idx = jnp.arange(s_vec.shape[0])
    return jax.vmap(fitness_specific_computations,
                    in_axes=(0, None, None, None, None, None, None, None, None))(
                    s_idx, s_vec, x_vec, exp_term_vec_s, log_rec_vec,
                    log_p_y_vec, time_points, n_mutations, cs)

# endregion


# ============================================================================
# region  h-grid construction + (h x s) grid driver
# ============================================================================

def build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h=1.0):
    """Per-clone h grid, floored at max(0, 2*Vmax-1).
       A non-NaN h_fixed for any mutation in a clone pins that clone's h."""
    AO = np.array(AO)
    DP = np.array(DP)
    observed = np.array(observed, dtype=bool)
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)
    h_fixed = np.array(h_fixed)

    grids = []
    for cs_idx in cs:
        hf = h_fixed[np.array(cs_idx)]
        pinned = hf[~np.isnan(hf)]
        if pinned.size:
            assert np.allclose(pinned, pinned[0]), \
                "clone mixes conflicting fixed-h values"
            grids.append(jnp.array([float(pinned[0])]))         # pinned (size-1 grid)
        else:
            lead = cs_idx[int(np.argmax(np.nansum(vaf[:, cs_idx], axis=0)))]
            v_max = np.nanmax(vaf[:, lead])
            h_min = min(max(2 * v_max - 1, 0.0), max_h)         # single-clone floor
            grids.append(jnp.linspace(h_min, max_h, h_resolution))
    return grids


def compute_cs_posterior_grid_vec(s_vec, h_grids, AO, DP, time_points, cs,
                                  observed, carry_idx):
    """LOG clonal likelihood over the full per-clone (h) x (s) grid,
       joint-feasibility masked. Returns log_out_grid (n_combo, s_res, K)
       and h_combos (n_combo, K)."""
    n_clones = len(cs)
    out_list, hcombo_list = [], []
    for idx in itertools.product(*[range(len(g)) for g in h_grids]):
        h = jnp.array([h_grids[k][idx[k]] for k in range(n_clones)])
        det_size, total_cells, _, h_mut = compute_deterministic_size(
            cs, AO, DP, AO.shape[1], h, observed)
        feasible = jnp.all(jnp.isfinite(total_cells) & (total_cells > 0))  # coupled frontier
        out = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                                det_size, total_cells, h_mut, observed, carry_idx)
        out_list.append(jnp.where(feasible, out, -jnp.inf))   # log(0) = -inf for infeasible
        hcombo_list.append(h)
    return jnp.stack(out_list), jnp.stack(hcombo_list)

# endregion


# ============================================================================
# region  Model probability + per-clone marginals  (LOG SPACE)
# ============================================================================

def compute_model_likelihood(log_out_grid, cs, s_vec):
    """Log-space. Returns a LOG model probability (fine for ranking)."""
    s_prior = 1.0 / (s_vec.max() - s_vec.min())
    log_g = log_trapz(log_out_grid, x=s_vec, axis=1) + jnp.log(s_prior)   # (n_combo, K)
    log_model_per_combo = jnp.sum(log_g, axis=1)                          # (n_combo,)
    log_model = logsumexp(log_model_per_combo) - jnp.log(log_model_per_combo.shape[0])
    return float(log_model)


def clone_posteriors(log_out_grid, h_combos, s_vec, h_grids):
    """Log-space; returns joint p(h_i, s_i) as a LINEAR array (max-normalised)."""
    s_prior = 1.0 / (s_vec.max() - s_vec.min())
    log_g = np.array(log_trapz(log_out_grid, x=s_vec, axis=1) + jnp.log(s_prior))  # (n_combo,K)
    log_og = np.array(log_out_grid)                                                # (n_combo,s,K)
    K = log_g.shape[1]
    out = []
    for i in range(K):
        log_others = np.sum(np.delete(log_g, i, axis=1), axis=1)   # (n_combo,)
        h_vals = np.array(h_grids[i]); hcol = np.array(h_combos[:, i])
        joint_log = np.full((len(h_vals), log_og.shape[1]), -np.inf)
        for hk, hv in enumerate(h_vals):
            m = np.isclose(hcol, hv)
            terms = log_og[m, :, i] + log_others[m][:, None]       # (n_match, s)
            joint_log[hk] = np.array(logsumexp(terms, axis=0))
        mx = np.nanmax(joint_log)
        joint = np.exp(joint_log - mx) if np.isfinite(mx) else np.zeros_like(joint_log)
        out.append((h_vals, joint))
    return out

# endregion


# ============================================================================
# region  Drivers
# ============================================================================

def compute_clonal_models_prob_vec(part, s_resolution=20, h_resolution=4,
                                   min_s=0.01, max_s=3.0, max_h=1.0,
                                   filter_invalid=True, disable_progressbar=False):
    """COMPARISON stage: rank all valid clonal structures. Keep grids COARSE --
       cost per structure ~ s_resolution * h_resolution ** (n_clones).
       model_dict values are LOG probabilities (larger = better)."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    n_mutations = part.shape[0]
    part.uns['model_dict'] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]

    for i, cs in tqdm(enumerate(cs_list), disable=disable_progressbar,
                      total=len(cs_list)):
        h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                      h_resolution, max_h)
        n_combo = int(np.prod([len(g) for g in h_grids]))
        if n_combo > 5000:
            print(f"  note: model {i} -> {n_combo} h-combos "
                  f"({len(cs)} clones x h_res={h_resolution}); may be slow")
        out_grid, _ = compute_cs_posterior_grid_vec(
            s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx)
        model_prob = compute_model_likelihood(out_grid, cs, s_vec)   # LOG prob
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(
        part.uns['model_dict'].items(), key=lambda kv: kv[1][1], reverse=True)}
    return part


def refine_optimal_model_posterior_vec(part, s_resolution=40, h_resolution=6,
                                       min_s=0.01, max_s=3.0, max_h=1.0):
    """REFINE stage: fine (s, h) posterior for the winning structure only.
       Cost ~ s_resolution * h_resolution ** (n_clones)."""
    cs = list(part.uns['model_dict'].values())[0][0]
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed,
                                  h_resolution, max_h)
    out_grid, h_combos = compute_cs_posterior_grid_vec(
        s_vec, h_grids, AO, DP, time_points, cs, observed, carry_idx)

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'posterior': out_grid,                 # LOG grid
        'h_combos': h_combos,
        's_range': s_vec,
        'h_grids': [np.array(g) for g in h_grids]}

    posteriors = clone_posteriors(out_grid, h_combos, s_vec, h_grids)

    n = part.shape[0]
    fitness        = np.zeros(n); fitness_5       = np.zeros(n); fitness_95       = np.zeros(n)
    homozygosity   = np.zeros(n); homozygosity_5  = np.zeros(n); homozygosity_95  = np.zeros(n)
    clonal_index   = np.zeros(n)
    fitness_railed            = np.zeros(n, dtype=bool)   # MAP s at max_s ceiling
    homozygosity_railed       = np.zeros(n, dtype=bool)   # MAP h at max_h ceiling
    homozygosity_unidentified = np.zeros(n, dtype=bool)   # h 90% CI too wide -> not LOH

    s_top = float(np.array(s_vec).max())
    eps = 1e-9
    H_CI_WIDTH = 0.5          # 90% CI wider than this -> h is unconstrained

    for i, c_idx in enumerate(cs):
        h_vals, joint = posteriors[i]                 # joint p(h, s), linear
        joint = np.nan_to_num(np.asarray(joint), nan=0.0, posinf=0.0, neginf=0.0)

        p_s = joint.sum(axis=0)
        p_h = joint.sum(axis=1)
        if p_s.sum() <= 0 or p_h.sum() <= 0:          # now catches partial-NaN too
            part.uns['warning'] = 'Zero posterior'
            return part
        p_s = p_s / p_s.sum()
        p_h = p_h / p_h.sum()


        fitness_map = float(np.array(s_vec)[np.argmax(p_s)])
        h_map = float(h_vals[np.argmax(p_h)])

        s_cfd = np.quantile(
            np.random.choice(np.array(s_vec), p=p_s, size=1_000), [0.05, 0.95])
        if len(h_vals) > 1:
            h_cfd = np.quantile(
                np.random.choice(h_vals, p=p_h, size=1_000), [0.05, 0.95])
        else:
            h_cfd = [h_vals[0], h_vals[0]]            # pinned clone

        # boundary / identifiability detection
        s_railed  = fitness_map >= s_top - eps
        h_railed  = (len(h_vals) > 1) and (h_map >= max_h - eps)          # pinned can't rail
        h_unident = (len(h_vals) > 1) and ((h_cfd[1] - h_cfd[0]) > H_CI_WIDTH)

        fitness[c_idx]        = fitness_map
        fitness_5[c_idx]      = s_cfd[0]; fitness_95[c_idx]      = s_cfd[1]
        homozygosity[c_idx]   = h_map
        homozygosity_5[c_idx] = h_cfd[0]; homozygosity_95[c_idx] = h_cfd[1]
        clonal_index[c_idx]   = i
        fitness_railed[c_idx]            = s_railed
        homozygosity_railed[c_idx]       = h_railed
        homozygosity_unidentified[c_idx] = h_unident

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95
    part.obs['clonal_index'] = clonal_index
    part.obs['fitness_railed'] = fitness_railed
    part.obs['homozygosity_railed'] = homozygosity_railed
    part.obs['homozygosity_unidentified'] = homozygosity_unidentified

    mut_structure = part.uns['optimal_model']['mutation_structure']
    part.obs['clonal_structure'] = [
        next(s for s in mut_structure if mut in s) for mut in part.obs.index]
    return part

# endregion


# ============================================================================
# region  Clonal-structure validity (gap-filled correlation)
# ============================================================================

def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    """Flag mutation pairs whose VAF-vs-time correlations differ too much
       (unlikely to share a clone). Uses gap-filled VAF so NaNs don't poison corr."""
    DP = np.array(part.layers['DP'])              # (n_mut, n_tp)
    AO = np.array(part.layers['AO'])
    observed = DP > 0
    vaf = np.where(observed, AO / np.where(DP > 0, DP, 1.0), np.nan)
    vaf_filled = np.vstack([_fill_nearest(vaf[i], observed[i])
                            for i in range(vaf.shape[0])])
    tp = np.array(part.var['time_points'], dtype=float)

    corr = np.corrcoef(np.vstack([vaf_filled, tp]))
    corr_vec = corr[-1, :-1]
    dist = np.abs(corr_vec - corr_vec[:, None])

    res = []
    for i, j in np.argwhere(dist > pearson_distance_threshold):
        pair = sorted([int(i), int(j)])
        if pair not in res:
            res.append(pair)
    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """All partitions of the mutations, optionally filtered by correlation analysis."""
    n_mutations = part.shape[0]
    if n_mutations == 1:
        return [[[0]]]

    if filter_invalid:
        compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)

    cs_list = [cs for cs in partition(list(range(n_mutations)))]
    if not filter_invalid:
        return cs_list

    valid_cs = []
    for cs in cs_list:
        bad = 0
        for clone in cs:
            bad += sum(1 for comb in combinations(clone, 2)
                       if sorted(comb) in part.uns['invalid_combinations'])
        if bad == 0:
            valid_cs.append(cs)
    return valid_cs

# endregion


# ============================================================================
# region  Plotting
# ============================================================================

def plot_optimal_model(part):
    """Overlay each clone's fitness marginal p(s) for the optimal model."""
    if part.uns.get('warning') is not None:
        print('WARNING: ' + str(part.uns['warning']))
    model = part.uns['optimal_model']
    out_grid = model['posterior']
    cs = model['clonal_structure']
    s_range = model['s_range']
    h_combos = model['h_combos']
    h_grids = [jnp.array(g) for g in model['h_grids']]

    posteriors = clone_posteriors(out_grid, h_combos, s_range, h_grids)
    for i in range(len(cs)):
        _, joint = posteriors[i]
        p_s = joint.sum(axis=0)
        p_s = p_s / p_s.max()
        label = '\n'.join(str(part.obs.index[j]) for j in cs[i])
        sns.lineplot(x=np.array(s_range), y=np.array(p_s), label=label)

# endregion


# ============================================================================
# region  SLOW REFERENCE IMPLEMENTATION  (validation only -- not in pipeline)
# ============================================================================
# LOG-space pure-numpy twin of jax_cs_hmm_ll_vec, for numerically diffing the
# vectorised core. Shares the beta integration NODES (same jax RNG) but uses
# scipy logpmf + hand-written log-sum-exp/log-trapezoid loops, so any mismatch
# isolates a bug in the broadcasting / HMM / h-handling logic.

from scipy.stats import binom as _sp_binom, nbinom as _sp_nbinom
from scipy.special import logsumexp as _sp_logsumexp


def _log_trapz_np(log_y, x, axis=-1):
    m = np.max(log_y, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    val = np.trapz(np.exp(log_y - m), x=x, axis=axis)
    return np.squeeze(m, axis=axis) + np.log(val)


def jax_cs_hmm_ll_ref(s_vec, AO, DP, time_points, cs,
                      deterministic_size, total_cells, h_mut,
                      observed, carry_idx, resolution=1_000):
    """LOG-space pure-numpy twin. Returns (s_res, K) of log-likelihoods."""
    AO = np.array(AO); DP = np.array(DP)
    observed = np.array(observed, dtype=bool)
    carry_idx = np.array(carry_idx)
    time_points = np.array(time_points, dtype=float)
    deterministic_size = np.array(deterministic_size)
    total_cells = np.array(total_cells)
    h_mut = np.array(h_mut); s_vec = np.array(s_vec)
    n_tp, n_mut = AO.shape
    lamb = 1.3

    DP_safe = np.where(observed, DP, 1.0)
    AO_safe = np.where(observed, AO, 0.0)
    beta = np.array(jrnd.beta(key=key,
                    a=jnp.array((AO_safe + 1)[:, :, None]),
                    b=jnp.array((DP_safe - AO_safe + 1)[:, :, None]),
                    shape=(n_tp, n_mut, resolution)))
    beta = np.sort(beta, axis=-1)

    N_w_cond = (total_cells[:, None] - deterministic_size)[:, :, None]
    half = (1.0 + h_mut)[None, :, None] / 2.0
    x = np.ceil(-N_w_cond * beta / (beta - half))
    valid = x > 0
    x = np.where(valid, x, 1.0)
    true_vaf = (1.0 + h_mut)[None, :, None] * x / (2.0 * (N_w_cond + x))

    log_p_y = np.where(valid,
                       _sp_binom.logpmf(AO_safe[:, :, None], n=DP_safe[:, :, None], p=true_vaf),
                       -np.inf)
    x = np.take_along_axis(x, np.broadcast_to(carry_idx[:, :, None], x.shape), axis=0)
    log_p_y = np.where(observed[:, :, None], log_p_y, 0.0)

    delta_t = np.diff(time_points)
    out = np.zeros((len(s_vec), len(cs)))
    for si, s in enumerate(s_vec):
        exp_term = np.exp(delta_t * s)
        mut_ll = np.zeros(n_mut)
        for m in range(n_mut):
            x_m = x[:, m, :]; lpy_m = log_p_y[:, m, :]
            mean = x_m[:-1] * exp_term[:, None]
            var  = x_m[:-1] * (2*lamb + s) * exp_term[:, None] * (exp_term[:, None] - 1) / s
            p_nb = mean / var
            n_nb = mean**2 / (var - mean)
            log_rec = lpy_m[0] - np.log(resolution)
            for j in range(1, n_tp):
                log_bd = _sp_nbinom.logpmf(x_m[j][:, None], p=p_nb[j-1], n=n_nb[j-1])
                log_inner = log_bd + log_rec[None, :]
                log_rec = lpy_m[j] + _log_trapz_np(log_inner, x_m[j-1], axis=-1)
            mut_ll[m] = _log_trapz_np(log_rec, x_m[-1])
        for ci, c_idx in enumerate(cs):
            out[si, ci] = np.sum(mut_ll[np.array(c_idx)])   # sum of logs
    return out


def describe_structure(part, cs, h=None):
    """Human-readable: which mutations sit in which clone (+ that clone's h)."""
    lines = []
    for ci, c_idx in enumerate(cs):
        labels = []
        for j in c_idx:
            row = part.obs.iloc[j]
            gene = row.get('GENE', '')
            prot = row.get('PROTEIN_CHANGE', '')
            labels.append(f"{gene} {prot}".strip())
        htxt = f"  (h={float(np.array(h)[ci]):.2f})" if h is not None else ""
        lines.append(f"    clone {ci}{htxt}: " + " | ".join(labels))
    return "\n".join(lines)


def validate_participant(part, s_resolution=15, h_resolution=3,
                         min_s=0.01, max_s=3.0, max_h=1.0, rtol=1e-3):
    """Diff the vectorised LOG core against the LOG reference on one participant /
       one clonal structure / the first feasible h-combo that gives a finite
       (non-degenerate) likelihood."""
    AO, DP, observed, carry_idx, time_points, h_fixed = _get_arrays(part)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    cs = find_valid_clonal_structures(part, filter_invalid=True)[0]
    h_grids = build_clone_h_grids(cs, AO, DP, observed, h_fixed, h_resolution, max_h)

    # first feasible h-combo that yields a finite (non-degenerate) likelihood
    h = det = tot = h_mut = vec = None
    for idx in itertools.product(*[range(len(g)) for g in h_grids]):
        cand = jnp.array([h_grids[k][idx[k]] for k in range(len(cs))])
        d, t, _, hm = compute_deterministic_size(cs, AO, DP, AO.shape[1], cand, observed)
        if not bool(jnp.all(jnp.isfinite(t) & (t > 0))):
            continue
        trial = np.array(jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                                           d, t, hm, observed, carry_idx))
        if np.any(np.isfinite(trial)):
            h, det, tot, h_mut, vec = cand, d, t, hm, trial
            break
    if h is None:
        print("no feasible / finite h-combo found for this structure "
              "(structure is degenerate -> would score ~0 and never be selected).")
        return None

    ref = jax_cs_hmm_ll_ref(s_vec, AO, DP, time_points, cs,
                            det, tot, h_mut, observed, carry_idx)

    print(f"participant   : {part.uns.get('participant_id')}")
    print(f"structure     :\n{describe_structure(part, cs, h)}")
    print(f"max abs diff  : {np.abs(vec - ref).max():.3e}   (log-likelihood units)")
    finite = np.isfinite(ref)
    denom = np.abs(ref); denom[~finite] = np.nan
    print(f"max rel diff  : {np.nanmax(np.abs(vec - ref) / denom):.3e}")
    print(f"argmax-s agree: {np.array_equal(vec.argmax(0), ref.argmax(0))}")
    ok = np.allclose(vec[finite], ref[finite], rtol=rtol, atol=1e-6)
    print("PASS" if ok else "FAIL")
    return vec, ref

# endregion
