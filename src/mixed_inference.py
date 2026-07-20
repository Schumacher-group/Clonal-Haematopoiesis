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


key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX

# ---------------------------------------------------------------------------
# Deterministic size
# ---------------------------------------------------------------------------
# h=0 : fully heterozygous  ->  VAF = x_tot / (2*(N_w + x_tot))
# h=1 : fully homozygous    ->  VAF = x_tot / (N_w + x_tot)
#
# General:  VAF = x_tot*(1+h) / (2*(N_w + x_tot))
#
# Inverting for x_tot given observed VAF v and zygosity h:
#   x_tot = -N_w * v / (v - (1+h)/2)
#
# NOTE: this function is called once per clonal structure before the (s, h)
# grid sweep.  We therefore pick a single representative h for the inversion
# so that total_cells / deterministic_size are well-defined scalars.  We use
# h=0 (fully heterozygous) as the conservative anchor; the VAF→x inversion
# inside the HMM likelihood uses the correct swept h at each grid point.
# ---------------------------------------------------------------------------

def compute_deterministic_size(cs, AO, DP, n_mutations, h=0.0):
    """Compute a fixed-h estimate of clone sizes used to set N_w_cond.

    Parameters
    ----------
    h : float
        Zygosity value used for the VAF→x_tot inversion.  Defaults to 0
        (fully heterozygous).  Pass the grid-minimum or a representative h
        when calling for h > 0 models.
    """
    N_w = 1e5

    # Determine leading mutation for each clone
    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    leading_vaf_sum = jnp.sum((AO / DP)[:, lm], axis=1)

    # Guard: VAF must be strictly below (1+h)/2 for inversion to be valid
    vaf_ceiling = (1.0 + h) / 2.0
    leading_vaf_sum = jnp.clip(leading_vaf_sum, 1e-6, vaf_ceiling - 1e-6)

    deterministic_clone_size = jnp.array(
        -N_w * leading_vaf_sum / (leading_vaf_sum - vaf_ceiling)
    )
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    # Back-project to per-mutation clone sizes using the same h
    deterministic_size = AO / DP * 2.0 * total_cells[:, None] / (1.0 + h)

    return deterministic_size, total_cells


# ---------------------------------------------------------------------------
# Non-vectorised HMM likelihood (kept for reference, mirrors original exactly)
# ---------------------------------------------------------------------------

def jax_cs_hmm_ll(s, AO, DP, time_points,
                  cs, h=0.0,
                  lamb=1.3):

    N_w = 1e5
    n_mutations = AO.shape[1]

    leading_mutation_in_cs_idx = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        leading_mutation_in_cs_idx.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    mutation_likelihood = jnp.zeros(n_mutations)

    leading_vaf_sum = jnp.sum((AO / DP)[:, leading_mutation_in_cs_idx], axis=1)
    vaf_ceiling = (1.0 + h) / 2.0
    leading_vaf_sum = jnp.clip(leading_vaf_sum, 1e-6, vaf_ceiling - 1e-6)

    deterministic_clone_size = jnp.array(
        -N_w * leading_vaf_sum / (leading_vaf_sum - vaf_ceiling)
    )
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO / DP * 2.0 * total_cells[:, None] / (1.0 + h)

    for j in range(n_mutations):

        s_clone = s[clonal_map[j]]

        beta_p_rvs = jrnd.beta(key=key, a=AO[:, j][:, None] + 1,
                               b=DP[:, j][:, None] - AO[:, j][:, None] + 1,
                               shape=(AO.shape[0], 1_000))

        beta_p_rvs = jnp.sort(beta_p_rvs)

        # Clip to below zygosity ceiling before inversion
        beta_p_rvs = jnp.clip(beta_p_rvs, 1e-6, vaf_ceiling - 1e-6)

        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]

        # VAF to x_tot, then VAF from x_tot using h
        x_range = jnp.array(-N_w_cond * beta_p_rvs / (beta_p_rvs - vaf_ceiling))
        x = jnp.ceil(x_range)

        true_vaf = x * (1.0 + h) / (2.0 * (N_w_cond + x))

        delta_t = jnp.diff(jnp.array(time_points))

        x_weight = jnp.repeat(1 / x[0].shape[0], x[0].shape[0])
        recursive_term = jsp_stats.binom.pmf(AO[0, j], n=DP[0, j], p=true_vaf[0]) * x_weight

        for i in range(1, x.shape[0]):
            init_size = x[i - 1]
            next_size = x[i]
            p_y_cond_x = jsp_stats.binom.pmf(AO[i, j], n=DP[i, j], p=true_vaf[i])

            exp_term = jnp.exp(delta_t[i - 1] * s_clone)
            mean = init_size * exp_term
            variance = init_size * (2 * lamb + s_clone) * exp_term * (exp_term - 1) / s_clone

            p = mean / variance
            n = jnp.power(mean, 2) / (variance - mean)

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)

            inner_sum = bd_pmf * recursive_term
            recursive_term = p_y_cond_x * jsp.integrate.trapezoid(x=x[i - 1], y=inner_sum)

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)]))

    return clonal_likelihood


def partition(collection):
    """Module computing an iterable over all partitions of a set"""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        yield [[first]] + smaller


# ---------------------------------------------------------------------------
# Vectorised functions
# ---------------------------------------------------------------------------

def jax_cs_hmm_ll_vec(s_vec, h_vec, AO, DP,
                      time_points, cs,
                      deterministic_size,
                      total_cells):
    """Compute clonal likelihoods over a grid of (s, h) values.

    Returns array of shape (n_s, n_h, n_clones).
    """

    n_h = h_vec.shape[0]
    n_s = s_vec.shape[0]
    n_clones = len(cs)

    output = jnp.zeros((n_s, n_h, n_clones))

    for h_idx in range(n_h):
        h = h_vec[h_idx]

        global_variables = compute_global_variables(s_vec, AO, DP, total_cells,
                                                    deterministic_size, time_points,
                                                    h=h)
        x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables

        s_idx = jnp.arange(s_vec.shape[0])

        clonal_likelihood = jax.vmap(fitness_specific_computations,
                                     in_axes=(0, None, None, None, None, None, None, None, None))(
                                     s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec,
                                     p_y_cond_x_vec, time_points, n_mutations, cs)

        output = output.at[:, h_idx, :].set(clonal_likelihood)

    return output


@jit
def compute_global_variables(s_vec, AO, DP,
                             total_cells, deterministic_size,
                             time_points,
                             h=0.0,
                             resolution=1_000):

    n_mutations = AO.shape[1]

    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    # FIX: use jnp.reshape, not np.reshape, inside a jit-compiled function
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    beta_p_rvs_vec = jrnd.beta(key=key, a=(AO + 1)[:, :, None],
                               b=DP[:, :, None] - AO[:, :, None] + 1,
                               shape=(AO.shape[0], AO.shape[1], resolution))

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)

    # Clip sampled VAFs to strictly below the zygosity ceiling (1+h)/2.
    vaf_ceiling = (1.0 + h) / 2.0
    beta_p_rvs_vec = jnp.clip(beta_p_rvs_vec, 1e-6, vaf_ceiling - 1e-6)

    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]

    # VAF -> x_tot using h
    x_range_vec = jnp.array(-N_w_cond_vec * beta_p_rvs_vec / (beta_p_rvs_vec - vaf_ceiling))
    x_vec = jnp.ceil(x_range_vec)

    # x_tot -> true VAF using h
    true_vaf_vec = x_vec * (1.0 + h) / (2.0 * (N_w_cond_vec + x_vec))

    p_y_cond_x_vec = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)

    recursive_term_vec = p_y_cond_x_vec[0, :, :] * 1 / resolution

    return (x_vec, exp_term_vec_s, recursive_term_vec,
            p_y_cond_x_vec, n_mutations)


@jit
def BD_process_dynamics(s, x_vec, exp_term_vec):
    lamb = 1.3
    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2 * lamb + s) * exp_term_vec * (exp_term_vec - 1) / s

    variance_vec = jnp.maximum(variance_vec, mean_vec * 1.01)  # variance must exceed mean for NegBin

    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)

    return p_vec, n_vec


def fitness_specific_computations(s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec,
                                   p_y_cond_x_vec, time_points, n_mutations, cs):

    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)

    mutation_likelihood = jax.vmap(mutation_specific_ll,
                                   in_axes=(0, None, None, None, None, None, None))(
                                   jnp.arange(n_mutations, dtype=int),
                                   recursive_term_vec, x_vec, p_vec,
                                   n_vec, p_y_cond_x_vec, time_points.shape[0])

    clonal_likelihood = jnp.zeros(len(cs))
    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)]))

    return clonal_likelihood


def mutation_specific_ll(i, recursive_term_vec, x_vec, p_vec,
                         n_vec, p_y_cond_x_vec, n_tps):

    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i)

    return jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)


@jit
def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    """Update the recursive term associated with mutation i for data point j"""

    bd_pmf_i = jsp_stats.nbinom.pmf(x_i[j][:, None], p=p_i[j - 1], n=n_i[j - 1])
    inner_sum_i = bd_pmf_i * recursive_term_i
    recursive_term_i = p_y_cond_x_i[j] * jsp.integrate.trapezoid(x=x_i[j - 1], y=inner_sum_i)

    return recursive_term_i


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def compute_model_likelihood(output, cs, s_range, h_range):
    """2-D marginalisation over (s, h) grid.

    output : (n_s, n_h, n_clones)
    Returns scalar model probability.
    """
    clonal_prob = np.zeros(len(cs))

    s_range_size = s_range.max() - s_range.min()
    h_range_size = h_range.max() - h_range.min()
    s_prior = 1 / s_range_size
    h_prior = 1 / h_range_size

    for i in range(len(cs)):
        # marginalise h first, then s
        int_h = np.trapz(x=h_range, y=np.array(output[:, :, i]), axis=1)   # shape (n_s,)
        int_sh = np.trapz(x=s_range, y=int_h)                               # scalar
        clonal_prob[i] = s_prior * h_prior * int_sh

    model_probability = np.prod(clonal_prob)
    return model_probability


def compute_clonal_models_prob_vec(part, s_resolution=50, h_resolution=10,
                                   min_s=0.01, max_s=1,
                                   filter_invalid=True, disable_progressbar=False):
    """Compute model probabilities for each clonal structure over (s, h) grid."""

    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    n_mutations = part.shape[0]

    part.uns['model_dict'] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'
        cs_list = [[[i] for i in range(n_mutations)]]

    for i, cs in tqdm(enumerate(cs_list), disable=disable_progressbar,
                      total=len(cs_list)):
        # Use h=0 as the anchor for deterministic size (conservative; see docstring)
        deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, AO.shape[1], h=0.0)

        output = jax_cs_hmm_ll_vec(s_vec, h_vec, AO, DP,
                                   time_points, cs,
                                   deterministic_size,
                                   total_cells)

        model_prob = compute_model_likelihood(output, cs, s_vec, h_vec)

        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(),
                                                        key=lambda item: item[1][1], reverse=True)}
    return part


# ---------------------------------------------------------------------------
# Posterior refinement
# ---------------------------------------------------------------------------

def refine_optimal_model_posterior_vec(part, s_resolution=100, h_resolution=20):
    """Compute finer (s, h) posterior for the optimal model."""

    cs = list(part.uns['model_dict'].values())[0][0]

    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    deterministic_size, total_cells = compute_deterministic_size(cs, AO, DP, AO.shape[1], h=0.0)

    s_vec = jnp.linspace(0.01, 1, s_resolution)
    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    # output shape: (n_s, n_h, n_clones)
    output = jax_cs_hmm_ll_vec(s_vec, h_vec, AO, DP, time_points, cs,
                                deterministic_size, total_cells)

    part.uns['optimal_model'] = {'clonal_structure': cs,
                                 'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
                                 'posterior': output,
                                 's_range': s_vec,
                                 'h_range': h_vec}

    fitness       = np.zeros(part.shape[0])
    fitness_5     = np.zeros(part.shape[0])
    fitness_95    = np.zeros(part.shape[0])
    zygosity      = np.zeros(part.shape[0])
    zygosity_5    = np.zeros(part.shape[0])
    zygosity_95   = np.zeros(part.shape[0])
    clonal_index  = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        p2d = np.array(output[:, :, i])

        # Clean up numerical noise before any operations
        p2d = np.nan_to_num(p2d, nan=0.0, posinf=0.0, neginf=0.0)
        p2d = np.maximum(p2d, 0.0)

        if p2d.sum() == 0:
            part.uns['warning'] = 'Zero posterior'
            return part

        # Normalise joint posterior
        p2d /= p2d.sum()

        # --- FIX: extract MAP from 2D joint posterior ---
        map_flat_idx = np.argmax(p2d)
        s_map_idx, h_map_idx = np.unravel_index(map_flat_idx, p2d.shape)
        s_map = float(s_vec[s_map_idx])
        h_map = float(h_vec[h_map_idx])
        # ------------------------------------------------

        # Marginal posteriors for CI sampling
        p_s = p2d.sum(axis=1)   # marginal over h  -> shape (n_s,)
        p_h = p2d.sum(axis=0)   # marginal over s  -> shape (n_h,)

        p_s /= p_s.sum()
        p_h /= p_h.sum()

        if np.any(np.isnan(p_s)) or np.any(np.isnan(p_h)):
            part.uns['warning'] = 'NaN in posterior marginals'
            return part

        s_samples = np.random.choice(s_vec, p=p_s, size=1_000)
        h_samples = np.random.choice(h_vec, p=p_h, size=1_000)

        s_ci = np.quantile(s_samples, [0.05, 0.95])
        h_ci = np.quantile(h_samples, [0.05, 0.95])

        fitness[c_idx]      = s_map
        fitness_5[c_idx]    = s_ci[0]
        fitness_95[c_idx]   = s_ci[1]
        zygosity[c_idx]     = h_map
        zygosity_5[c_idx]   = h_ci[0]
        zygosity_95[c_idx]  = h_ci[1]
        clonal_index[c_idx] = i

    part.obs['fitness']     = fitness
    part.obs['fitness_5']   = fitness_5
    part.obs['fitness_95']  = fitness_95
    part.obs['zygosity']    = zygosity
    part.obs['zygosity_5']  = zygosity_5
    part.obs['zygosity_95'] = zygosity_95
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
# Plotting
# ---------------------------------------------------------------------------

def plot_optimal_model(part):
    if part.uns['warning'] is not None:
        print('WARNING: ' + part.uns['warning'])

    model   = part.uns['optimal_model']
    output  = model['posterior']    # (n_s, n_h, n_clones)
    cs      = model['clonal_structure']
    ms      = model['mutation_structure']
    s_range = model['s_range']
    h_range = model['h_range']

    for i in range(len(cs)):
        p_key_str = ''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k > 0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        p2d = np.array(output[:, :, i])
        p2d /= p2d.sum()

        p_s = p2d.sum(axis=1); p_s /= p_s.sum()
        p_h = p2d.sum(axis=0); p_h /= p_h.sum()

        axes[0].plot(s_range, p_s / p_s.max(), label=p_key_str)
        axes[0].set_xlabel('s (fitness)')
        axes[0].set_ylabel('Normalised posterior')
        axes[0].legend()

        axes[1].plot(h_range, p_h / p_h.max(), label=p_key_str)
        axes[1].set_xlabel('h (zygosity)')
        axes[1].legend()

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Clonal structure utilities
# ---------------------------------------------------------------------------

def compute_invalid_combinations(part, slope_difference_threshold=0.05):
    """Flag mutation pairs whose VAF slopes differ too much to be co-clonal.
    
    With only 2 timepoints, Pearson correlation is degenerate (always ±1
    for any monotone trajectory). Slope difference is more informative:
    two mutations with significantly different VAF/year cannot share the
    same clone fitness s.
    """
    vaf = part.X                    # (n_mutations, n_timepoints)
    time_points = np.asarray(part.var.time_points, dtype=float)

    t_range = time_points[-1] - time_points[0]
    if t_range < 1e-6:
        part.uns["invalid_combinations"] = []
        return

    # Fit slope per mutation via first/last timepoint
    slopes = (vaf[:, -1] - vaf[:, 0]) / t_range   # (n_mutations,) in VAF/year

    invalid_pairs = []
    n = vaf.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(slopes[i] - slopes[j]) > slope_difference_threshold:
                invalid_pairs.append([i, j])

    part.uns["invalid_combinations"] = invalid_pairs


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """Find all valid clonal structures using pearson correlation analysis"""

    n_mutations = part.shape[0]

    if n_mutations == 1:
        valid_cs = [[[0]]]
        return valid_cs

    else:
        if filter_invalid is True:
            compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)

        a = partition(list(range(n_mutations)))
        cs_list = [cs for cs in a]

        if filter_invalid is False:
            return cs_list

        else:
            valid_cs = []

            for cs in cs_list:
                invalid_combinations_in_cs = 0
                for clone in cs:
                    mut_comb = list(combinations(clone, 2))
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb
                         if list(comb) in part.uns['invalid_combinations']])

                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                if invalid_combinations_in_cs == 0:
                    valid_cs.append(cs)

            return valid_cs