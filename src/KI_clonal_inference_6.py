from itertools import combinations

import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import numpy as np
from scipy.stats import binom
from tqdm import tqdm
from scipy.special import logsumexp



# NumPy 2 compatibility.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


DEFAULT_MASTER_KEY_SEED = 758493
DEFAULT_WILD_TYPE_POPULATION = 1e5
DEFAULT_BIRTH_RATE = 1.3
EPS = 1e-8
LIKELIHOOD_FLOOR = 1e-300


# ---------------------------------------------------------------------------
# Slope utilities (shared by structure inference and invalid-pair filtering)
# ---------------------------------------------------------------------------

def _compute_slope_and_se(part):
    """Compute per-mutation VAF slopes and their binomial standard errors.

    Uses the first and last timepoint only.  For participants with >2
    timepoints a weighted least-squares regression would use more of the
    data, but this formulation is consistent with the 2-timepoint case and
    avoids introducing a separate code path.

    Returns
    -------
    slopes   : (n_mutations,) VAF/year slope for each mutation
    se_slope : (n_mutations,) standard error on each slope
    t_range  : scalar, time span in years
    """
    AO = part.layers["AO"].astype(float)
    DP = part.layers["DP"].astype(float)
    vaf = AO / np.maximum(DP, 1.0)
    time_points = np.asarray(part.var.time_points, dtype=float)
    t_range = time_points[-1] - time_points[0]

    slopes = (vaf[:, -1] - vaf[:, 0]) / t_range

    # Binomial SE on each VAF estimate, propagated to slope SE
    se_v0 = np.sqrt(
        np.maximum(vaf[:, 0] * (1.0 - vaf[:, 0]), 1e-6)
        / np.maximum(DP[:, 0], 1.0)
    )
    se_vT = np.sqrt(
        np.maximum(vaf[:, -1] * (1.0 - vaf[:, -1]), 1e-6)
        / np.maximum(DP[:, -1], 1.0)
    )
    se_slope = np.sqrt(se_v0**2 + se_vT**2) / t_range

    return slopes, se_slope, t_range


# ---------------------------------------------------------------------------
# Slope-based clonal structure
# ---------------------------------------------------------------------------

def _slope_derived_structure(part, z_threshold=1.5):
    """Derive clonal structure from VAF slopes using uncertainty-aware union-find.

    Two mutations are merged into the same clone only if their slope difference
    is less than z_threshold pooled standard errors.  This replaces the fixed
    absolute threshold (slope_difference_threshold=0.05) which caused systematic
    over-merging under low sequencing depth and short follow-up, because a fixed
    0.05 VAF/year difference is within 1-2 sigma of noise at typical depths.

    z_threshold=1.5 is deliberately conservative (biases toward splitting)
    because the 2-timepoint setting has no power to average noise across the
    trajectory.  Expose as a parameter so it can be tuned against participants
    with known ground-truth structure.
    """
    slopes, se_slope, t_range = _compute_slope_and_se(part)
    n = part.shape[0]

    if t_range < 1e-6 or n == 1:
        return [list(range(n))]

    # Union-find with path compression
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            pooled_se = np.sqrt(se_slope[i]**2 + se_slope[j]**2)
            if pooled_se < 1e-10:
                # Both SEs negligible — fall back to tight absolute threshold
                should_merge = abs(slopes[i] - slopes[j]) < 0.01
            else:
                z = abs(slopes[i] - slopes[j]) / pooled_se
                should_merge = z < z_threshold
            if should_merge:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # Sort clones by descending mean slope (fastest growing first)
    return sorted(groups.values(), key=lambda g: -float(np.mean(slopes[g])))


def assign_clonal_structure_from_slopes(part, z_threshold=1.5):
    """Assign clonal structure directly from VAF slopes, bypassing the HMM.

    Populates part.uns['model_dict'] with the slope-derived structure.
    Use this instead of compute_clonal_models_prob_vec_mixed when timepoints
    are sparse (< 3) or follow-up is short (< 1.5 years).
    """
    cs = _slope_derived_structure(part, z_threshold)
    part.uns["model_dict"] = {"model_0": (cs, 1.0, 1.0)}
    part.uns["warning"] = None
    return part


# ---------------------------------------------------------------------------
# Deterministic size initialisation
# ---------------------------------------------------------------------------

def compute_deterministic_size_mixed(cs, AO, DP, n_mutations, N_w=1e5):
    """Compute initial clone sizes used to initialise the HMM.

    Parameterisation: x_het = (1-h)*x_tot, x_hom = h*x_tot
    VAF ceiling: v_max(h) = (1+h)/2

    Inverts v = x_tot*(1+h) / (2*(N_w+x_tot)) for x_tot:
        x_tot = N_w * v / ((1+h)/2 - v)

    Uses an adaptive h_min floor with a generous gap (0.15) above the peak
    observed VAF so that the inversion denominator ((1+h)/2 - v) is always
    well-conditioned, even when the observed VAF approaches its ceiling.

    h_min is derived from the correct ceiling formula:
        v_max = (1+h)/2  =>  h_min = 2*(v_peak + min_gap) - 1
    """
    vaf_ratio = AO / jnp.maximum(DP, 1.0)

    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax(vaf_ratio[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    leading_vaf_sum = jnp.sum(vaf_ratio[:, lm], axis=1)
    v_peak = jnp.max(leading_vaf_sum)

    min_gap = 0.15
    # FIX: derived from correct ceiling v_max = (1+h)/2, so h = 2*v_max - 1
    h_min = jnp.maximum(0.0, 2.0 * (v_peak + min_gap) - 1.0)
    h_min = jnp.minimum(h_min, 1.0)

    # FIX: invert v = x_tot*(1+h) / (2*(N_w+x_tot))
    # => x_tot = N_w * v / ((1+h)/2 - v)
    denom = (1.0 + h_min) / 2.0 - leading_vaf_sum
    denom = jnp.where(jnp.abs(denom) < EPS, EPS, denom)
    deterministic_clone_size = jnp.ceil(N_w * leading_vaf_sum / denom)

    total_cells = N_w + deterministic_clone_size

    # x_tot per mutation: invert per-mutation VAF using same h_min
    # v_mut = x_tot_mut*(1+h_min) / (2*(N_w + x_tot_clone))
    # x_tot_mut = 2 * v_mut * total_cells / (1+h_min)
    deterministic_size = vaf_ratio * 2.0 * total_cells[:, None] / (1.0 + h_min)

    max_total_per_mutation = jnp.max(deterministic_size, axis=0)

    return deterministic_size, total_cells, max_total_per_mutation, clonal_map


# ---------------------------------------------------------------------------
# HMM likelihood
# ---------------------------------------------------------------------------

def compute_global_variables_mixed(
    s_vec,
    h_vec,
    AO,
    DP,
    total_cells,
    deterministic_size,
    max_total_per_mutation,
    time_points,
    key,
    resolution=600,
):
    """Compute HMM global variables.

    Parameterisation: x_het = (1-h)*x_tot, x_hom = h*x_tot
    VAF = x_tot*(1+h) / (2*(N_w + x_tot))
    Ceiling: v_max(h) = (1+h)/2

    Inversion for x_tot given observed VAF v:
        x_tot = N_w * v / ((1+h)/2 - v)
    """
    n_timepoints, n_mutations = AO.shape
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = exp_term_vec_s.reshape((*exp_term_vec_s.shape, 1, 1, 1))

    key_beta, _ = jrnd.split(key)
    beta_p_rvs = jrnd.beta(
        key=key_beta,
        a=(AO + 1.0)[:, :, None],
        b=(DP - AO + 1.0)[:, :, None],
        shape=(n_timepoints, n_mutations, resolution),
    )
    beta_p_rvs = jnp.clip(beta_p_rvs, EPS, 1.0 - EPS)

    beta_p_rvs = beta_p_rvs[:, :, :, None]
    h_bcast = h_vec[None, None, None, :]

    N_w = (total_cells[:, None] - deterministic_size)[:, :, None, None]
    N_w = jnp.maximum(N_w, EPS)

    # FIX: correct ceiling is (1+h)/2
    v_max = (1.0 + h_bcast) / 2.0 - 1e-6
    v = jnp.minimum(beta_p_rvs, v_max)
    v = jnp.maximum(v, EPS)

    # FIX: invert v = x_tot*(1+h) / (2*(N_w+x_tot))
    # => x_tot = N_w * v / ((1+h)/2 - v)
    denom = (1.0 + h_bcast) / 2.0 - v
    denom = jnp.where(jnp.abs(denom) < EPS, EPS, denom)
    x_total = N_w * v / denom
    x_total = jnp.clip(x_total, 0.0, max_total_per_mutation[None, :, None, None])

    # Correct parameterisation: x_hom = h*x_tot, x_het = (1-h)*x_tot
    x_hom = x_total * h_bcast
    x_het = x_total * (1.0 - h_bcast)

    # VAF = (x_het + 2*x_hom) / (2*(N_w+x_tot))
    #     = x_tot*(1+h) / (2*(N_w+x_tot))
    numerator = x_het + 2.0 * x_hom
    denominator = 2.0 * (N_w + x_total)

    true_vaf = numerator / jnp.maximum(denominator, EPS)
    true_vaf = jnp.clip(true_vaf, EPS, 1.0 - EPS)

    p_y_cond_x = jsp_stats.binom.pmf(
        AO[:, :, None, None],
        n=DP[:, :, None, None],
        p=true_vaf,
    )

    p_y_cond_x = jnp.transpose(p_y_cond_x, (0, 3, 1, 2))
    x_total = jnp.transpose(x_total, (0, 3, 1, 2))

    # Trapezoidal integration requires x to be sorted along the integration axis.
    sort_idx = jnp.argsort(x_total, axis=-1)
    x_total = jnp.take_along_axis(x_total, sort_idx, axis=-1)
    p_y_cond_x = jnp.take_along_axis(p_y_cond_x, sort_idx, axis=-1)

    recursive_term = p_y_cond_x[0] * (1.0 / resolution)


    return (
        x_total,
        exp_term_vec_s,
        recursive_term,
        p_y_cond_x,
        n_mutations,
    )


def BD_process_dynamics_mixed(s, x_total, exp_term_vec, lamb=DEFAULT_BIRTH_RATE):
    s_safe = jnp.maximum(jnp.abs(s), EPS)
    mean_vec = x_total[:-1] * exp_term_vec
    variance_vec = (
        x_total[:-1]
        * (2.0 * lamb + s)
        * exp_term_vec
        * (exp_term_vec - 1.0)
        / s_safe
    )

    min_variance = mean_vec * 1.2 + 1e-6
    variance_vec = jnp.maximum(variance_vec, min_variance)

    p_vec = jnp.clip(mean_vec / variance_vec, EPS, 1.0 - EPS)
    n_vec = jnp.maximum(mean_vec**2 / jnp.maximum(variance_vec - mean_vec, EPS), EPS)
    return p_vec, n_vec


def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    log_bd_pmf = jsp_stats.nbinom.logpmf(
        x_i[j][:, None],
        p=p_i[j - 1],
        n=n_i[j - 1],
    )
    bd_pmf = jnp.exp(jnp.maximum(log_bd_pmf, -300.0))
    inner_sum = bd_pmf * recursive_term_i
    integrated = jsp.integrate.trapezoid(x=x_i[j - 1], y=inner_sum, axis=1)
    integrated = jnp.maximum(integrated, LIKELIHOOD_FLOOR)

    updated = jnp.log(jnp.maximum(p_y_cond_x_i[j], LIKELIHOOD_FLOOR)) + jnp.log(integrated)
    return jnp.exp(jnp.maximum(updated, -300.0))


def mutation_specific_ll(i, recursive_term_vec, x_vec, p_vec, n_vec, p_y_cond_x_vec, n_tps):
    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i)

    final_like = jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)
    return jnp.maximum(final_like, LIKELIHOOD_FLOOR)


def fitness_specific_computations_mixed(
    s_idx,
    s_vec,
    h_idx,
    x_total,
    exp_term_vec_s,
    recursive_term,
    p_y_cond_x,
    time_points,
    n_mutations,
    cs,
):
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]
    p_vec, n_vec = BD_process_dynamics_mixed(s, x_total, exp_term_vec)

    x_tot_h = x_total[:, h_idx, :, :]
    p_h = p_vec[:, h_idx, :, :]
    n_h = n_vec[:, h_idx, :, :]
    p_y_h = p_y_cond_x[:, h_idx, :, :]
    rec_h = recursive_term[h_idx, :, :]

    mutation_likelihood = jax.vmap(
        lambda mutation_index: mutation_specific_ll(
            mutation_index,
            rec_h,
            x_tot_h,
            p_h,
            n_h,
            p_y_h,
            time_points.shape[0],
        )
    )(jnp.arange(n_mutations, dtype=int))

    clonal_likelihood = jnp.zeros(len(cs))
    for clone_index, clone_mutations in enumerate(cs):
        clone_mutations = jnp.array(clone_mutations)
        log_mut_liks = jnp.log(jnp.maximum(mutation_likelihood[clone_mutations], LIKELIHOOD_FLOOR))
        clonal_likelihood = clonal_likelihood.at[clone_index].set(jnp.exp(jnp.sum(log_mut_liks)))

    return clonal_likelihood


def jax_cs_hmm_ll_vec_mixed(
    s_vec,
    h_vec,
    AO,
    DP,
    time_points,
    cs,
    deterministic_size,
    total_cells,
    max_total_per_mutation,
    key,
    resolution=600,
):
    x_total, exp_term_vec_s, recursive_term, p_y_cond_x, n_mut = compute_global_variables_mixed(
        s_vec,
        h_vec,
        AO,
        DP,
        total_cells,
        deterministic_size,
        max_total_per_mutation,
        time_points,
        key,
        resolution=resolution,
    )

    s_idx = jnp.arange(s_vec.shape[0])
    h_idx = jnp.arange(h_vec.shape[0])

    def for_one_s(si):
        def for_one_h(hi):
            return fitness_specific_computations_mixed(
                si,
                s_vec,
                hi,
                x_total,
                exp_term_vec_s,
                recursive_term,
                p_y_cond_x,
                time_points,
                n_mut,
                cs,
            )

        return jax.vmap(for_one_h)(h_idx)

    return jax.vmap(for_one_s)(s_idx)


def compute_model_log_likelihood_2d(output, cs, s_range, h_range):
    """Compute model evidence in log-space to avoid underflow."""
    s_range = np.asarray(s_range, dtype=float)
    h_range = np.asarray(h_range, dtype=float)

    ds = float(np.mean(np.diff(s_range)))
    dh = float(np.mean(np.diff(h_range)))

    log_s_prior = -np.log(float(s_range.max() - s_range.min()))
    log_h_prior = -np.log(float(h_range.max() - h_range.min()))

    total_log_like = 0.0

    for clone_index in range(len(cs)):
        grid = np.asarray(output[:, :, clone_index], dtype=float)
        grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
        grid = np.maximum(grid, 0.0)

        log_grid = np.log(np.maximum(grid, LIKELIHOOD_FLOOR))

        clone_log_like = (
            logsumexp(log_grid)
            + np.log(ds)
            + np.log(dh)
            + log_s_prior
            + log_h_prior
        )

        total_log_like += clone_log_like

    return float(total_log_like)



# ---------------------------------------------------------------------------
# Model selection — slope-anchored
# ---------------------------------------------------------------------------

def compute_clonal_models_prob_vec_mixed(
    part,
    s_resolution=50,
    h_resolution=20,
    min_s=0.01,
    max_s=3.0,
    filter_invalid=True,
    disable_progressbar=False,
    resolution=600,
    master_key_seed=DEFAULT_MASTER_KEY_SEED,
    z_threshold=1.5,
):
    """Model selection using slope-derived structure as anchor.

    Rather than enumerating all valid partitions and ranking with the HMM,
    this function:
      1. Derives the clonal structure directly from VAF slopes (uncertainty-
         aware union-find, replacing the fixed absolute threshold).
      2. Evaluates that single structure with the HMM to get (s, h) likelihoods.
      3. Falls back gracefully if the HMM returns all-zero probabilities.

    Parameters
    ----------
    z_threshold : float
        Number of pooled standard errors required to split two mutations into
        separate clones.  Lower values split more aggressively.  Default 1.5
        is conservative for the 2-timepoint setting.
    """
    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    part.uns["model_dict"] = {}
    part.uns["warning"] = None

    # Derive clonal structure from slopes — this is the anchor
    cs = _slope_derived_structure(part, z_threshold)

    model_keys = jrnd.split(jrnd.PRNGKey(master_key_seed), 1)

    deterministic_size, total_cells, max_total_per_mutation, _ = compute_deterministic_size_mixed(
        cs, AO, DP, AO.shape[1],
    )

    output = jax_cs_hmm_ll_vec_mixed(
        s_vec, h_vec, AO, DP, time_points,
        cs, deterministic_size, total_cells, max_total_per_mutation,
        model_keys[0], resolution=resolution,
    )

    model_log_prob = compute_model_log_likelihood_2d(
        output,
        cs,
        np.asarray(s_vec),
        np.asarray(h_vec),
    )
    
    if not np.isfinite(model_log_prob):
        part.uns["warning"] = "HMM log evidence non-finite — slope structure used directly"

    # Store temporary entry.
    part.uns["model_dict"]["model_0"] = (cs, model_log_prob, None)

    # Normalise model posterior probabilities in log-space.
    model_items = list(part.uns["model_dict"].items())

    log_evidences = np.array(
        [entry[1] for _, entry in model_items],
        dtype=float,
    )

    finite_mask = np.isfinite(log_evidences)
    posterior_probs = np.zeros_like(log_evidences, dtype=float)

    if finite_mask.any():
        log_norm = logsumexp(log_evidences[finite_mask])
        posterior_probs[finite_mask] = np.exp(
        log_evidences[finite_mask] - log_norm
        )
    else:
        posterior_probs[:] = 1.0 / len(posterior_probs)

    for idx, (model_name, entry) in enumerate(model_items):
        cs_i, log_evidence_i, _ = entry
        part.uns["model_dict"][model_name] = (
        cs_i,
        log_evidence_i,
        posterior_probs[idx],
    )

    return part



# ---------------------------------------------------------------------------
# Partition utilities (kept for reference / future use)
# ---------------------------------------------------------------------------

def partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for subset_index, subset in enumerate(smaller):
            yield smaller[:subset_index] + [[first] + subset] + smaller[subset_index + 1:]
        yield [[first]] + smaller


def compute_invalid_combinations(part, z_threshold=1.5):
    """Flag mutation pairs whose VAF slopes differ significantly.

    Replaces the Pearson correlation approach, which is degenerate with 2
    timepoints (correlation is always ±1 for any monotone series, giving
    stddev=0 and NaN in np.corrcoef).

    Populates part.uns['invalid_combinations'] with pairs [i, j] where
    the slope z-score exceeds z_threshold.
    """
    slopes, se_slope, t_range = _compute_slope_and_se(part)
    n = part.shape[0]

    if t_range < 1e-6:
        part.uns["invalid_combinations"] = []
        return

    invalid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pooled_se = np.sqrt(se_slope[i]**2 + se_slope[j]**2)
            if pooled_se < 1e-10:
                is_invalid = abs(slopes[i] - slopes[j]) >= 0.01
            else:
                z = abs(slopes[i] - slopes[j]) / pooled_se
                is_invalid = z >= z_threshold
            if is_invalid:
                invalid_pairs.append([i, j])

    part.uns["invalid_combinations"] = invalid_pairs


def _leading_mutation_for_clone(vaf_ratio, clone_mutations):
    clone_vafs = vaf_ratio[:, clone_mutations]
    lead_idx_within_clone = int(np.argmax(clone_vafs.sum(axis=0)))
    return clone_mutations[lead_idx_within_clone]


def find_valid_clonal_structures(part, z_threshold=1.5, filter_invalid=True):
    n_mutations = part.shape[0]

    if n_mutations == 1:
        return [[[0]]]

    if filter_invalid:
        compute_invalid_combinations(part, z_threshold=z_threshold)

    cs_list = list(partition(list(range(n_mutations))))

    if not filter_invalid:
        return cs_list

    valid_cs = []
    for cs in cs_list:
        invalid_combinations_in_cs = 0
        for clone in cs:
            mut_pairs = list(combinations(clone, 2))
            invalid_combinations_in_cs += len(
                [pair for pair in mut_pairs if list(pair) in part.uns["invalid_combinations"]]
            )
        if invalid_combinations_in_cs == 0:
            valid_cs.append(cs)

    return valid_cs


# ---------------------------------------------------------------------------
# Deterministic trajectory inference
# ---------------------------------------------------------------------------

def _project_clone_vaf(
    time_points,
    initial_mutant_cells,
    s,
    h,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
):
    """Project clone VAF trajectory forward in time.

    Parameterisation:
        x_het = (1-h) * x_total,  x_hom = h * x_total

    VAF = (x_het + 2*x_hom) / (2*(N_w + x_total))
        = x_total * (1+h) / (2*(N_w + x_total))

    Ceiling: v_max(h) = (1+h)/2
        h=0 (fully het)  -> v_max = 0.5
        h=1 (fully hom)  -> v_max = 1.0

    h is the homozygous fraction of the mutant population (LOH fraction),
    not a dominance coefficient.
    """
    time_points = np.asarray(time_points, dtype=float)

    x_tot = (
        initial_mutant_cells
        * np.exp(s * (time_points - time_points[0]))
    )

    # FIX: was (1+2h) — correct formula is (1+h)
    projected_vaf = x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))

    return np.clip(projected_vaf, EPS, (1.0 + h) / 2.0 - EPS)


def _extract_clone_joint_posterior(part, clone_index):
    model = part.uns["optimal_model"]
    grid = np.array(model["posterior_2d"][:, :, clone_index], copy=False)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.maximum(grid, 0.0)

    s_range = np.asarray(model["s_range"])
    h_range = np.asarray(model["h_range"])

    normalizer = np.trapz(np.trapz(grid, x=h_range, axis=1), x=s_range)
    if normalizer <= 0 or not np.isfinite(normalizer):
        return None

    return grid / normalizer


def infer_sh_jointly_from_dynamics(
    cs,
    AO,
    DP,
    time_points,
    s_resolution=30,
    h_resolution=20,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
):
    """Infer (s, h) jointly via a deterministic trajectory likelihood.

    Parameterisation: x_het = (1-h)*x_tot, x_hom = h*x_tot
    VAF = x_tot*(1+h) / (2*(N_w + x_tot))
    Ceiling: v_max(h) = (1+h)/2

    Inversion for initial clone size from observed VAF v0:
        x_tot0 = 2 * v0 * N_w / ((1+h) - 2*v0)

    Uses only the leading mutation per clone.  MAP estimates and CIs are
    extracted from the joint (s, h) posterior via argmax + unravel_index.

    CIs use linear interpolation on the CDF rather than discrete searchsorted,
    which removes the grid-resolution quantisation in the reported intervals.
    """
    s_range = np.linspace(0.01, 3.0, s_resolution)
    h_range = np.linspace(0.0, 1.0, h_resolution)

    vaf_ratio = (
        np.asarray(AO, dtype=float)
        / np.maximum(np.asarray(DP, dtype=float), 1.0)
    )

    results = []

    for clone_index, clone_mutations in enumerate(cs):

        lead_mutation = _leading_mutation_for_clone(vaf_ratio, clone_mutations)

        ao_clone = np.asarray(AO[:, lead_mutation], dtype=float)
        dp_clone = np.asarray(DP[:, lead_mutation], dtype=float)

        valid_mask = dp_clone > 0
        valid_time_points = np.asarray(time_points, dtype=float)[valid_mask]
        ao_valid = ao_clone[valid_mask]
        dp_valid = dp_clone[valid_mask]
        observed_vaf = ao_valid / np.maximum(dp_valid, 1.0)

        joint_log_likelihood = np.zeros((s_resolution, h_resolution))

        for s_idx, s in enumerate(s_range):
            for h_idx, h in enumerate(h_range):

                v0 = observed_vaf[0]

                # FIX: invert v = x_tot*(1+h) / (2*(N_w+x_tot))
                # => x_tot0 = 2*v0*N_w / ((1+h) - 2*v0)
                denom = (1.0 + h) - 2.0 * v0
                denom = max(denom, EPS)

                initial_mutant_cells = max(
                    2.0 * v0 * N_w / denom,
                    100.0,
                )

                projected_vaf = _project_clone_vaf(
                    valid_time_points,
                    initial_mutant_cells,
                    s,
                    h,
                    N_w=N_w,
                )

                log_lik = 0.0
                for time_index, vaf_expected in enumerate(projected_vaf):
                    log_lik += binom.logpmf(
                        int(ao_valid[time_index]),
                        int(dp_valid[time_index]),
                        vaf_expected,
                    )

                joint_log_likelihood[s_idx, h_idx] = log_lik

        # Stabilise and normalise joint posterior
        joint_likelihood = np.exp(
            np.clip(joint_log_likelihood - joint_log_likelihood.max(), -700, 0)
        )
        joint_posterior = joint_likelihood / np.maximum(joint_likelihood.sum(), LIKELIHOOD_FLOOR)

        map_flat = np.argmax(joint_posterior)
        s_map_idx, h_map_idx = np.unravel_index(map_flat, joint_posterior.shape)
        s_map = s_range[s_map_idx]
        h_map = h_range[h_map_idx]

        # Marginal posteriors from the full joint (not outer product approximation)
        s_posterior = joint_posterior.sum(axis=1)
        h_posterior = joint_posterior.sum(axis=0)

        s_posterior = s_posterior / (s_posterior.sum() + LIKELIHOOD_FLOOR)
        h_posterior = h_posterior / (h_posterior.sum() + LIKELIHOOD_FLOOR)

        # CIs via interpolation on CDF — removes grid-resolution quantisation
        s_cumsum = np.cumsum(s_posterior)
        h_cumsum = np.cumsum(h_posterior)

        s_ci = (
            float(np.interp(0.05, s_cumsum, s_range)),
            float(np.interp(0.95, s_cumsum, s_range)),
        )
        h_ci = (
            float(np.interp(0.05, h_cumsum, h_range)),
            float(np.interp(0.95, h_cumsum, h_range)),
        )

        # FIX: invert v = x_tot*(1+h) / (2*(N_w+x_tot)) at MAP estimates
        # => x_tot0 = 2*v0*N_w / ((1+h_map) - 2*v0)
        denom = (1.0 + h_map) - 2.0 * observed_vaf[0]
        denom = max(denom, EPS)
        initial_mutant_cells = max(
            2.0 * observed_vaf[0] * N_w / denom,
            100.0,
        )

        projected_vaf = _project_clone_vaf(
            valid_time_points,
            initial_mutant_cells,
            s_map,
            h_map,
            N_w=N_w,
        )

        results.append(
            {
                "s_map": s_map,
                "h_map": h_map,
                "s_posterior": s_posterior,
                "h_posterior": h_posterior,
                "joint_posterior": joint_posterior,
                "s_range": s_range,
                "h_range": h_range,
                "s_ci": s_ci,
                "h_ci": h_ci,
                "leading_mutation_index": lead_mutation,
                "valid_mask": valid_mask,
                "time_points_valid": valid_time_points,
                "observed_vaf_valid": observed_vaf,
                "projected_vaf_valid": projected_vaf,
                "initial_mutant_cells": initial_mutant_cells,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Posterior refinement
# ---------------------------------------------------------------------------

def refine_optimal_model_posterior_vec_mixed(
    part,
    s_resolution=30,
    h_resolution=20,
    resolution=1000,
    master_key_seed=42,
):
    """Refine (s, h) posterior for the optimal model using deterministic trajectories.

    Uses infer_sh_jointly_from_dynamics (leading mutation per clone, det likelihood).
    Populates part.obs with fitness, zygosity, and clonal_index columns.

    posterior_2d is stored as the true joint posterior per clone (not an outer
    product of marginals), preserving the (s, h) correlation structure.
    """
    if len(part.uns["model_dict"]) == 0:
        part.uns["warning"] = "No models to refine"
        return part

    cs = list(part.uns["model_dict"].values())[0][0]

    AO = np.asarray(part.layers["AO"].T, dtype=float)
    DP = np.asarray(part.layers["DP"].T, dtype=float)
    time_points = np.asarray(part.var.time_points, dtype=float)

    joint_results = infer_sh_jointly_from_dynamics(
        cs,
        AO,
        DP,
        time_points,
        s_resolution=s_resolution,
        h_resolution=h_resolution,
    )

    s_arr = np.asarray(joint_results[0]["s_range"])
    h_arr = np.asarray(joint_results[0]["h_range"])

    # Use the true joint posterior directly, not the outer product approximation
    posterior_2d = np.stack(
        [result["joint_posterior"] for result in joint_results],
        axis=2,
    )

    part.uns["optimal_model"] = {
        "clonal_structure": cs,
        "mutation_structure": [list(part.obs.iloc[clone_mutations].index) for clone_mutations in cs],
        "posterior_2d": posterior_2d,
        "s_range": s_arr,
        "h_range": h_arr,
        "joint_inference": joint_results,
    }

    fitness      = np.zeros(part.shape[0])
    fitness_5    = np.zeros(part.shape[0])
    fitness_95   = np.zeros(part.shape[0])
    zygosity     = np.zeros(part.shape[0])
    zygosity_5   = np.zeros(part.shape[0])
    zygosity_95  = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for clone_index, clone_mutations in enumerate(cs):
        result = joint_results[clone_index]

        fitness[clone_mutations]      = result["s_map"]
        fitness_5[clone_mutations]    = result["s_ci"][0]
        fitness_95[clone_mutations]   = result["s_ci"][1]
        zygosity[clone_mutations]     = result["h_map"]
        zygosity_5[clone_mutations]   = result["h_ci"][0]
        zygosity_95[clone_mutations]  = result["h_ci"][1]
        clonal_index[clone_mutations] = clone_index

    part.obs["fitness"]      = fitness
    part.obs["fitness_5"]    = fitness_5
    part.obs["fitness_95"]   = fitness_95
    part.obs["zygosity"]     = zygosity
    part.obs["zygosity_5"]   = zygosity_5
    part.obs["zygosity_95"]  = zygosity_95
    part.obs["clonal_index"] = clonal_index

    mutation_structure = part.uns["optimal_model"]["mutation_structure"]
    part.obs["clonal_structure"] = [
        next(structure for structure in mutation_structure if mutation in structure)
        for mutation in part.obs.index
    ]

    return part