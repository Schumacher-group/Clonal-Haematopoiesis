import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations


key = jrnd.PRNGKey(758493)


# ---------------------------------------------------------------------------
# Homozygosity (h) constraint helpers
# ---------------------------------------------------------------------------

def compute_h_min_by_clone(cs, AO, DP, use_leading_only=True):
    """
    Compute the minimum h required per clone to explain observed VAFs.

    From the VAF model:
        v = (1+h)x / (2(Nw+x))

    The maximum possible VAF as x -> inf is:
        v_max = (1+h)/2

    So to explain observed VAF v:
        h >= 2*v - 1
    """

    VAF = jnp.where(DP > 0, AO / DP, 0.0)

    h_min = []
    leading_mutations = []

    for clone in cs:
        clone = list(clone)

        if use_leading_only:
            clone_vaf_sum = VAF[:, clone].sum(axis=0)
            leading_idx = int(jnp.argmax(clone_vaf_sum))
            leading_mutation = clone[leading_idx]
            vmax = float(jnp.max(VAF[:, leading_mutation]))
            leading_mutations.append(leading_mutation)
        else:
            vmax = float(jnp.max(VAF[:, clone]))
            leading_mutations.append(None)

        h_required = max(0.0, 2.0 * vmax - 1.0)
        h_min.append(h_required)

    return jnp.array(h_min), leading_mutations


def compute_h_vec_for_clone(h_min_clone, max_h, h_resolution):
    """
    Compute a clone-specific h grid starting from h_min_clone.
    Returns None if h_min_clone > max_h (clone is mathematically infeasible).
    """
    if h_min_clone > max_h:
        return None
    return jnp.linspace(h_min_clone, max_h, h_resolution)


def compute_low_h_prior(h_vec, low_h_strength=2.0):
    """
    Exponential prior favouring low h.
        p(h) ∝ exp(-low_h_strength * h)

    Normalised over the h_vec range.
    """
    weights = jnp.exp(-low_h_strength * h_vec)
    weights = weights / jnp.trapezoid(weights, h_vec)
    return weights


# ---------------------------------------------------------------------------
# Non-vectorised functions
# ---------------------------------------------------------------------------

def compute_deterministic_size(cs, AO, DP, n_mutations, h=0.0):
    """
    Deterministic size of clones, allowing for homozygosity fraction h.

    VAF model:
        v = (1+h)x / (2(Nw+x))

    Inverse:
        x = -Nw*v / (v - (1+h)/2)
    """

    N_w = 1e5
    vaf_threshold = (1 + h) / 2

    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(
            jnp.repeat(i, len(cs_idx))
        )

    leading_vaf_sum = jnp.sum((AO / DP)[:, lm], axis=1)

    deterministic_clone_size = jnp.array(
        -N_w * leading_vaf_sum / (leading_vaf_sum - vaf_threshold)
    )

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)

    total_cells = N_w + deterministic_clone_size

    deterministic_size = (
        (AO / DP) * 2 * total_cells[:, None] / (1 + h)
    )

    return deterministic_size, total_cells


def jax_cs_hmm_ll(s, AO, DP, time_points,
                  cs,
                  lamb=1.3,
                  h=0.0):

    N_w = 1e5
    n_mutations = AO.shape[1]
    vaf_threshold = (1 + h) / 2

    leading_mutation_in_cs_idx = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        leading_mutation_in_cs_idx.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(
            jnp.repeat(i, len(cs_idx))
        )

    mutation_likelihood = jnp.zeros(n_mutations)

    leading_vaf_sum = jnp.sum((AO / DP)[:, leading_mutation_in_cs_idx], axis=1)

    deterministic_clone_size = jnp.array(
        -N_w * leading_vaf_sum / (leading_vaf_sum - vaf_threshold)
    )

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)

    total_cells = N_w + deterministic_clone_size

    deterministic_size = (
        (AO / DP) * 2 * total_cells[:, None] / (1 + h)
    )

    for j in range(n_mutations):

        s_clone = s[clonal_map[j]]

        beta_p_rvs = jrnd.beta(
            key=key,
            a=AO[:, j][:, None] + 1,
            b=DP[:, j][:, None] - AO[:, j][:, None] + 1,
            shape=(AO.shape[0], 1_000)
        )

        beta_p_rvs = jnp.sort(beta_p_rvs)

        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]

        x_range = jnp.array(
            -N_w_cond * beta_p_rvs / (beta_p_rvs - vaf_threshold)
        )

        x = jnp.ceil(x_range)

        true_vaf = (1 + h) * x / (2 * (N_w_cond + x))

        delta_t = jnp.diff(jnp.array(time_points))

        x_weight = jnp.repeat(1 / x[0].shape[0], x[0].shape[0])

        recursive_term = (
            jsp_stats.binom.pmf(AO[0, j], n=DP[0, j], p=true_vaf[0])
            * x_weight
        )

        for i in range(1, x.shape[0]):

            next_size = x[i]

            p_y_cond_x = jsp_stats.binom.pmf(
                AO[i, j],
                n=DP[i, j],
                p=true_vaf[i]
            )

            exp_term = jnp.exp(delta_t[i - 1] * s_clone)
            mean = x[i - 1] * exp_term
            variance = (
                x[i - 1]
                * (2 * lamb + s_clone)
                * exp_term
                * (exp_term - 1)
                / s_clone
            )

            p = mean / variance
            n = jnp.power(mean, 2) / (variance - mean)

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)
            inner_sum = bd_pmf * recursive_term

            recursive_term = (
                p_y_cond_x
                * jsp.integrate.trapezoid(x=x[i - 1], y=inner_sum)
            )

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    clonal_likelihood = jnp.zeros(len(cs))

    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            jnp.prod(mutation_likelihood[jnp.array(c_idx)])
        )

    return clonal_likelihood


def partition(collection):
    """Iterable over all set partitions of a collection."""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


def single_cs_posterior(part, cs,
                        s_resolution=100,
                        h_resolution=50,
                        min_h=0.0,
                        max_h=1.0):
    """Compute posterior for a single clonal structure over both s and h."""

    s_range = jnp.linspace(0.01, 1, s_resolution)
    h_range = jnp.linspace(min_h, max_h, h_resolution)

    AO = part.layers['AO'].T
    DP = part.layers['DP'].T
    time_points = part.var.time_points

    def eval_for_h(h):
        multi_s_range = jnp.broadcast_to(
            s_range,
            (len(cs), s_resolution)
        ).T

        return jax.vmap(
            jax_cs_hmm_ll,
            in_axes=(0, None, None, None, None, None, None)
        )(
            multi_s_range, AO, DP, time_points, cs, 1.3, h
        )

    output = jax.vmap(eval_for_h)(h_range)

    # shape: s x h x clone
    output = jnp.transpose(output, axes=(1, 0, 2))

    return output, s_range, h_range


def compute_model_likelihood(output, cs, s_range, h_range=None,
                             h_prior_weights=None):
    """
    Compute model likelihood by marginalising over s (and optionally h).

    Parameters
    ----------
    output : array
        Shape (s, h, clone) if h_range provided, else (s, clone).
    cs : list of lists
    s_range : array
    h_range : array or None
    h_prior_weights : array or None
        If provided, weights applied to each h before integrating.
        Shape must match h_range.
    """

    clonal_prob = np.zeros(len(cs))

    s_range_size = float(s_range.max() - s_range.min())
    s_prior = 1.0 / s_range_size if s_range_size > 0 else 1.0

    if h_range is None:
        for i, out in enumerate(output.T):
            clonal_prob[i] = s_prior * np.trapezoid(x=s_range, y=out)

    else:
        h_range_size = float(h_range.max() - h_range.min())
        h_prior = 1.0 / h_range_size if h_range_size > 0 else 1.0

        for i in range(len(cs)):
            out = np.array(output[:, :, i])

            # Apply h prior weights if provided
            if h_prior_weights is not None:
                out = out * np.array(h_prior_weights)[None, :]

            int_s = np.trapezoid(x=s_range, y=out, axis=0)
            int_h = np.trapezoid(x=h_range, y=int_s)

            clonal_prob[i] = s_prior * h_prior * int_h

    model_probability = np.prod(clonal_prob)

    return model_probability


# ---------------------------------------------------------------------------
# Vectorised functions
# ---------------------------------------------------------------------------

def jax_cs_hmm_ll_vec(s_vec, AO, DP,
                      time_points, cs,
                      deterministic_size,
                      total_cells,
                      h=0.0,
                      resolution=1_000):

    global_variables = compute_global_variables(
        s_vec, AO, DP,
        total_cells, deterministic_size,
        time_points, h, resolution
    )

    x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables

    s_idx = jnp.arange(s_vec.shape[0])

    clonal_likelihood = jax.vmap(
        fitness_specific_computations,
        in_axes=(0, None, None, None, None, None, None, None, None)
    )(
        s_idx, s_vec, x_vec, exp_term_vec_s,
        recursive_term_vec, p_y_cond_x_vec,
        time_points, n_mutations, cs
    )

    return clonal_likelihood


@jit
def compute_global_variables(s_vec, AO, DP,
                             total_cells, deterministic_size,
                             time_points,
                             h,
                             resolution=1_000):

    n_mutations = AO.shape[1]
    vaf_threshold = (1 + h) / 2

    delta_t = jnp.diff(time_points)

    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    beta_p_rvs_vec = jrnd.beta(
        key=key,
        a=(AO + 1)[:, :, None],
        b=DP[:, :, None] - AO[:, :, None] + 1,
        shape=(AO.shape[0], AO.shape[1], resolution)
    )

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)

    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]

    x_range_vec = jnp.array(
        -N_w_cond_vec * beta_p_rvs_vec
        / (beta_p_rvs_vec - vaf_threshold)
    )

    x_vec = jnp.ceil(x_range_vec)

    true_vaf_vec = (1 + h) * x_vec / (2 * (N_w_cond_vec + x_vec))

    p_y_cond_x_vec = jsp_stats.binom.pmf(
        AO[:, :, None],
        n=DP[:, :, None],
        p=true_vaf_vec
    )

    recursive_term_vec = p_y_cond_x_vec[0, :, :] * 1 / resolution

    return (
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        n_mutations
    )


@jit
def BD_process_dynamics(s, x_vec, exp_term_vec):
    lamb = 1.3
    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2 * lamb + s) * exp_term_vec * (exp_term_vec - 1) / s

    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)

    return p_vec, n_vec


def fitness_specific_computations(s_idx, s_vec, x_vec, exp_term_vec_s,
                                  recursive_term_vec, p_y_cond_x_vec,
                                  time_points, n_mutations, cs):

    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)

    mutation_likelihood = jax.vmap(
        mutation_specific_ll,
        in_axes=(0, None, None, None, None, None, None)
    )(
        jnp.arange(n_mutations, dtype=int),
        recursive_term_vec, x_vec, p_vec,
        n_vec, p_y_cond_x_vec, time_points.shape[0]
    )

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            jnp.prod(mutation_likelihood[jnp.array(c_idx)])
        )

    return clonal_likelihood


def mutation_specific_ll(i, recursive_term_vec, x_vec, p_vec,
                         n_vec, p_y_cond_x_vec, n_tps):

    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(
            j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i
        )

    return jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)


@jit
def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    """Update the recursive term for mutation i at timepoint j."""
    bd_pmf_i = jsp_stats.nbinom.pmf(x_i[j][:, None], p=p_i[j - 1], n=n_i[j - 1])
    inner_sum_i = bd_pmf_i * recursive_term_i
    recursive_term_i = p_y_cond_x_i[j] * jsp.integrate.trapezoid(x=x_i[j - 1], y=inner_sum_i)
    return recursive_term_i


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compute_clonal_models_prob_vec(part,
                                   s_resolution=50,
                                   h_resolution=50,
                                   min_s=0.01,
                                   max_s=1.0,
                                   min_h=0.0,
                                   max_h=1.0,
                                   filter_invalid=True,
                                   disable_progressbar=False,
                                   use_leading_only_h_min=True,
                                   prefer_low_h=True,
                                   low_h_strength=2.0,
                                   resolution=1_000):
    """
    Compute model probabilities for each clonal structure,
    marginalising over both fitness s and homozygosity fraction h.

    Key additions:
    - Per-clone h_min constraint:  h >= max(0, 2*v_max - 1)
    - Optional exponential low-h prior:  p(h) ∝ exp(-low_h_strength * h)
    """

    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mutations = part.shape[0]

    part.uns['model_dict'] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None

    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]

    for i, cs in tqdm(
        enumerate(cs_list),
        disable=disable_progressbar,
        total=len(cs_list)
    ):
        # ----------------------------------------------------------------
        # Per-clone h_min from observed VAFs
        # ----------------------------------------------------------------
        h_min_by_clone, leading_mutations = compute_h_min_by_clone(
            cs, AO, DP, use_leading_only=use_leading_only_h_min
        )

        # Global h_min: must be satisfied by all clones
        global_h_min = float(jnp.max(h_min_by_clone))

        # Skip this clonal structure if it is physically infeasible
        if global_h_min > max_h:
            continue

        # h grid starting from the minimum required value
        h_vec_model = jnp.linspace(
            max(min_h, global_h_min),
            max_h,
            h_resolution
        )

        # ----------------------------------------------------------------
        # Optional low-h prior
        # ----------------------------------------------------------------
        if prefer_low_h and float(h_vec_model.max() - h_vec_model.min()) > 0:
            h_prior_weights = compute_low_h_prior(h_vec_model, low_h_strength)
        else:
            h_prior_weights = None

        h_outputs = []

        for h in h_vec_model:

            deterministic_size, total_cells = compute_deterministic_size(
                cs, AO, DP, AO.shape[1], h=float(h)
            )

            output_h = jax_cs_hmm_ll_vec(
                s_vec, AO, DP,
                time_points, cs,
                deterministic_size, total_cells,
                h=float(h),
                resolution=resolution
            )

            h_outputs.append(output_h)

        output = jnp.stack(h_outputs, axis=1)  # shape: s x h x clone

        model_prob = compute_model_likelihood(
            output, cs, s_vec, h_vec_model,
            h_prior_weights=h_prior_weights
        )

        part.uns['model_dict'][f'model_{i}'] = (
            cs,
            model_prob,
            {
                'homozygosity_min_by_clone': np.array(h_min_by_clone),
                'global_homozygosity_min': global_h_min,
                'homozygosity_range': np.array(h_vec_model),
                'leading_mutations': leading_mutations,
            }
        )

    part.uns['model_dict'] = {
        k: v
        for k, v in sorted(
            part.uns['model_dict'].items(),
            key=lambda item: item[1][1],
            reverse=True
        )
    }

    return part


# ---------------------------------------------------------------------------
# Posterior refinement
# ---------------------------------------------------------------------------

def refine_optimal_model_posterior_vec(part,
                                       s_resolution=100,
                                       h_resolution=100,
                                       min_s=0.01,
                                       max_s=1.0,
                                       min_h=0.0,
                                       max_h=1.0,
                                       use_leading_only_h_min=True,
                                       prefer_low_h=True,
                                       low_h_strength=2.0,
                                       resolution=1_000):
    """
    Refine the posterior for the optimal model over both s and h.

    Applies the same h_min constraint and low-h prior as model comparison.
    """

    cs = list(part.uns['model_dict'].values())[0][0]

    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    # ----------------------------------------------------------------
    # Per-clone h_min constraint
    # ----------------------------------------------------------------
    h_min_by_clone, leading_mutations = compute_h_min_by_clone(
        cs, AO, DP, use_leading_only=use_leading_only_h_min
    )

    global_h_min = float(jnp.max(h_min_by_clone))
    effective_min_h = max(min_h, global_h_min)

    h_vec = jnp.linspace(effective_min_h, max_h, h_resolution)

    if prefer_low_h and float(h_vec.max() - h_vec.min()) > 0:
        h_prior_weights = compute_low_h_prior(h_vec, low_h_strength)
    else:
        h_prior_weights = None

    h_outputs = []

    for h in h_vec:

        deterministic_size, total_cells = compute_deterministic_size(
            cs, AO, DP, AO.shape[1], h=float(h)
        )

        output_h = jax_cs_hmm_ll_vec(
            s_vec, AO, DP,
            time_points, cs,
            deterministic_size, total_cells,
            h=float(h),
            resolution=resolution
        )

        h_outputs.append(output_h)

    output = jnp.stack(h_outputs, axis=1)  # shape: s x h x clone

    # Apply low-h prior weights
    if h_prior_weights is not None:
        output = output * jnp.array(h_prior_weights)[None, :, None]

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [
            list(part.obs.iloc[cs_idx].index) for cs_idx in cs
        ],
        'posterior': output,
        's_range': s_vec,
        'homozygosity_range': h_vec,
        'homozygosity_min_by_clone': np.array(h_min_by_clone),
        'global_homozygosity_min': global_h_min,
        'leading_mutations': leading_mutations,
    }

    # ----------------------------------------------------------------
    # Extract MAP and credible intervals for s and h per clone
    # ----------------------------------------------------------------
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])

    homozygosity = np.zeros(part.shape[0])
    homozygosity_5 = np.zeros(part.shape[0])
    homozygosity_95 = np.zeros(part.shape[0])

    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        joint_p = np.array(output[:, :, i])

        if np.nansum(joint_p) == 0:
            part.uns['warning'] = 'Zero posterior'
            return part

        joint_p = np.nan_to_num(joint_p, nan=0.0, posinf=0.0, neginf=0.0)
        joint_p = joint_p / np.sum(joint_p)

        # MAP
        max_idx = np.unravel_index(np.argmax(joint_p), joint_p.shape)
        fitness_map = float(s_vec[max_idx[0]])
        h_map = float(h_vec[max_idx[1]])

        # Bootstrap credible intervals from joint posterior
        flat_p = joint_p.ravel()
        flat_p = flat_p / flat_p.sum()

        s_grid, h_grid = np.meshgrid(
            np.array(s_vec),
            np.array(h_vec),
            indexing='ij'
        )

        sampled_idx = np.random.choice(
            np.arange(flat_p.size),
            p=flat_p,
            size=1_000
        )

        sampled_s = s_grid.ravel()[sampled_idx]
        sampled_h = h_grid.ravel()[sampled_idx]

        fitness_ci = np.quantile(sampled_s, [0.05, 0.95])
        h_ci = np.quantile(sampled_h, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = fitness_ci[0]
        fitness_95[c_idx] = fitness_ci[1]

        homozygosity[c_idx] = h_map
        homozygosity_5[c_idx] = h_ci[0]
        homozygosity_95[c_idx] = h_ci[1]

        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95

    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95

    part.obs['clonal_index'] = clonal_index

    mut_structure = part.uns['optimal_model']['mutation_structure']

    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)

    part.obs['clonal_structure'] = clonal_structure_list

    return part


# ---------------------------------------------------------------------------
# Clonal structure helpers
# ---------------------------------------------------------------------------

def compute_invalid_combinations(part, pearson_distance_threshold=0.5):

    correlation_matrix = np.corrcoef(
        np.vstack([part.X, part.var.time_points])
    )
    correlation_vec = correlation_matrix[-1, :-1]

    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])

    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    not_valid_comb = [list(i) for i in not_valid_comb]

    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i)

    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """Find all valid clonal structures using Pearson correlation analysis."""

    n_mutations = part.shape[0]

    if n_mutations == 1:
        return [[[0]]]

    if filter_invalid:
        compute_invalid_combinations(
            part, pearson_distance_threshold=p_distance_threshold
        )

    a = partition(list(range(n_mutations)))
    cs_list = list(a)

    if not filter_invalid:
        return cs_list

    valid_cs = []

    for cs in cs_list:
        invalid_combinations_in_cs = 0

        for clone in cs:
            mut_comb = list(combinations(clone, 2))
            n_invalid = len(
                [comb for comb in mut_comb
                 if list(comb) in part.uns['invalid_combinations']]
            )
            invalid_combinations_in_cs += n_invalid

        if invalid_combinations_in_cs == 0:
            valid_cs.append(cs)

    return valid_cs
