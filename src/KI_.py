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


key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX12
#region Non vectorised functions
# Non vectorised
def compute_deterministic_size(cs, AO, DP, n_mutations, h):
    # Deterministic size of clones
    # s independent, but cs dependent
    # h is the fraction of homozygosity
    N_w = 1e5

    # determine leading mutation for each clone
    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, lm], axis=1)
                                            / (jnp.sum((AO/DP)[:, lm], axis=1)-(1+h)/2))

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP*2*total_cells[:, None]

    return deterministic_size, total_cells

def jax_cs_hmm_ll(s, AO, DP, time_points,
                  cs, h,
                  lamb=1.3):

    N_w = 1e5
    n_mutations = AO.shape[1]

    # determine leading mutation for each clone
    leading_mutation_in_cs_idx = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)


    for i, cs_idx in enumerate(cs):

        # Sum over time points to find biggest overall mutation (summing over time points)
        # argamx detects leading mutation
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        leading_mutation_in_cs_idx.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    mutation_likelihood = jnp.zeros(n_mutations)

    deterministic_clone_size = jnp.array(-N_w*jnp.sum((AO/DP)[:, leading_mutation_in_cs_idx], axis=1)
                                            / (jnp.sum((AO/DP)[:, leading_mutation_in_cs_idx], axis=1)-(1+h)/2))

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP*2*total_cells[:, None]

    # Loop through each mutation
    for j in range(n_mutations):

        s_clone = s[clonal_map[j]]

        # Sample hidden Vaf sizes for all time points
        beta_p_rvs = jrnd.beta(key=key, a=AO[:, j][:, None]+1,
                            b=DP[:,j][:, None]- AO[:,j][:, None] + 1,
                            shape=(AO.shape[0], 1_000))

        # sort sampled vafs  (for integration purposes)
        beta_p_rvs = jnp.sort(beta_p_rvs)

        # Transform VAFs to clone sizes

        # Wild type cells plus other clones
        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]

        # VAF to Size transformation
        x_range = jnp.array(-N_w_cond*beta_p_rvs / (beta_p_rvs-(1+h)/2))
        x = jnp.ceil(x_range)

        # add endpoints right endpoint is set to 2*max_x
        # x = jnp.c_[jnp.ones(x.shape[0]), x, x[:,-1]*2]
        true_vaf = (1+h)*x/(2*(N_w_cond+x))

        # # computation of BD process
        delta_t = jnp.diff(jnp.array(time_points))
        # exp_term = jnp.exp(delta_t*s_clone[j])

        x_weight = jnp.repeat(1/x[0].shape[0], x[0].shape[0])
        recursive_term = jsp_stats.binom.pmf(AO[0, j], n=DP[0,j], p=true_vaf[0])*x_weight

        for i in range(1, x.shape[0]):
            init_size = x[i-1]
            next_size = x[i]
            p_y_cond_x = jsp_stats.binom.pmf(AO[i, j], n=DP[i, j], p=true_vaf[i])


            # Predict pmf of BD process
            exp_term = jnp.exp(delta_t[i-1]*s_clone)
            mean = init_size*exp_term
            variance = init_size*(2*lamb + s_clone)*exp_term*(exp_term-1)/s_clone

            # Neg Binom parametrization
            p = mean / variance
            n = jnp.power(mean, 2) / (variance-mean)

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)

            inner_sum = bd_pmf*recursive_term
            recursive_term = p_y_cond_x*jsp.integrate.trapezoid(x=x[i-1], y=inner_sum)

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
        yield [ collection ]
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def single_cs_posterior(part, cs, s_resolution=100,
                        h_resolution=20, min_h=0., max_h=1.):
    """Compute posterior for a single clonal structure over the (s, h) grid"""

    # create finer s and h arrays
    s_range = jnp.linspace(0.01, 1, s_resolution)
    h_range = jnp.linspace(min_h, max_h, h_resolution)
    multi_s_range = jnp.broadcast_to(s_range, (len(cs), s_resolution)).T

    # loop over homozygosity fraction, vmap over fitness
    outputs = []
    for h in h_range:
        out = jax.vmap(jax_cs_hmm_ll,
                       in_axes=(0, None, None, None, None, None, None))(
                       multi_s_range, part.layers['AO'].T, part.layers['DP'].T,
                       part.var.time_points, cs, h, 1.3)
        outputs.append(out)

    # output shape: (h_resolution, s_resolution, n_clones)
    output = jnp.stack(outputs)

    return output, s_range, h_range

def compute_clonal_models_prob (part, s_resolution=50, h_resolution=20):
    """Compute model probabilities for each clonal structure.
    Append results in unstructued dictionary."""

    # Compute all possible clonal structures (or partition sets)
    n_mutations = part.shape[0]
    a = partition(list(range(n_mutations)))
    part.uns['model_dict'] = {}

    for i, cs in tqdm(enumerate(a)):

        output, s_range, h_range = single_cs_posterior(part, cs, s_resolution, h_resolution)
        model_prob = compute_model_likelihood(output, cs, s_range, h_range)

        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}

    return part

def compute_model_likelihood(output, cs, s_range, h_range=None):
    # initialise clonal probability
    clonal_prob = np.zeros(len(cs))

    s_range_size = s_range.max() - s_range.min()
    s_prior = 1/s_range_size

    if h_range is None:
        # 1-D marginalisation (fitness only) - backwards compatible
        for i, out in enumerate(output.T):
            clonal_prob[i] = s_prior*np.trapz(x=s_range, y=out)
    else:
        # 2-D marginalisation over fitness (s) and homozygosity (h)
        # output shape: (h_resolution, s_resolution, n_clones)
        h_range_size = h_range.max() - h_range.min()
        h_prior = 1/h_range_size if h_range_size > 0 else 1.0

        for i in range(len(cs)):
            out = np.array(output[:, :, i])          # (h_res, s_res)
            marg_s = np.trapz(out, x=s_range, axis=1)  # marginalise s -> (h_res,)
            if h_range_size > 0:
                clonal_prob[i] = h_prior*s_prior*np.trapz(marg_s, x=h_range)
            else:
                clonal_prob[i] = s_prior*marg_s[0]

    # Model probability as product of clonal probabilities
    # because of independence
    model_probability = np.prod(clonal_prob)

    return model_probability

def refine_optimal_model_posterior(part, s_resolution=100,
                                   h_resolution=40, min_h=0., max_h=1.):
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]
    output, s_range, h_range = single_cs_posterior(part, cs, s_resolution,
                                                   h_resolution, min_h, max_h)
    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_range,
                                'h_range': h_range}

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    homozygosity = np.zeros(part.shape[0])
    homozygosity_5 = np.zeros(part.shape[0])
    homozygosity_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        # posterior for clone i over the (h, s) grid
        p = np.array(output[:, :, i])

        # marginal posteriors
        p_s = p.sum(axis=0)          # p(s)
        p_h = p.sum(axis=1)          # p(h)
        p_s = p_s/p_s.sum()
        p_h = p_h/p_h.sum()

        fitness_map = s_range[np.argmax(p_s)]
        h_map = h_range[np.argmax(p_h)]

        s_sample = np.random.choice(s_range, p=p_s, size=1_000)
        h_sample = np.random.choice(h_range, p=p_h, size=1_000)
        s_cfd = np.quantile(s_sample, [0.05, 0.95])
        h_cfd = np.quantile(h_sample, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = s_cfd[0]
        fitness_95[c_idx] = s_cfd[1]
        homozygosity[c_idx] = h_map
        homozygosity_5[c_idx] = h_cfd[0]
        homozygosity_95[c_idx] = h_cfd[1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95
    part.obs['clonal_index'] = clonal_index


    return part

def plot_optimal_model(part):
    if part.uns['warning'] is not None:
        print('WARNING: ' + part.uns['warning'])
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']

    # marginalise over homozygosity to obtain the fitness posterior
    # output shape: (h_res, s_res, n_clones)
    output_s = np.array(output).sum(axis=0)

    # normalisation constant
    norm_max = np.max(output_s, axis=0)

    # Plot
    for i in range(len(cs)):
        p_key_str = f''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k>0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        sns.lineplot(x=s_range,
                    y=output_s[:, i]/ norm_max[i],
                    label=p_key_str)
#endregion

#region vectorised and jit optimised functions

def jax_cs_hmm_ll_vec(s_vec, AO, DP,
                      time_points, cs,
                      deterministic_size,
                      total_cells, h):

    global_variables = compute_global_variables(s_vec, AO, DP, total_cells, deterministic_size, time_points, h)
    x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables

    # create s indexes for vectorisation
    s_idx = jnp.arange(s_vec.shape[0])

    clonal_likelihood = jax.vmap(fitness_specific_computations,
                                 in_axes=(0, None, None, None, None, None, None, None, None))(
                                 s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec,
                                 p_y_cond_x_vec, time_points, n_mutations, cs)

    return clonal_likelihood

@jit
def compute_global_variables(s_vec, AO, DP,
                             total_cells, deterministic_size,
                             time_points, h,
                             resolution=1_000):

    # number of mutations in the participant
    n_mutations = AO.shape[1]

    # BD process exponential term for pmf
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t*s_vec[:, None])
    exp_term_vec_s = np.reshape(exp_term_vec_s, (*exp_term_vec_s.shape,1, 1))

    beta_p_rvs_vec = jrnd.beta(key=key, a=(AO+1)[:, :, None],
                    b=DP[:, :, None]- AO[:, :, None] + 1,
                    shape=(AO.shape[0], AO.shape[1], resolution))

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    x_range_vec = jnp.array(-N_w_cond_vec*beta_p_rvs_vec / (beta_p_rvs_vec-(1+h)/2))
    x_vec = jnp.ceil(x_range_vec)

    # add endpoints right endpoint is set to 2*max_x
    true_vaf_vec = (1+h)*x_vec/(2*(N_w_cond_vec+x_vec))

    # compute conditional observation probabilities over all time points
    p_y_cond_x_vec = jsp_stats.binom.pmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)

    # Compute probability of first timepoint
    recursive_term_vec = p_y_cond_x_vec[0, :, :]*1/resolution

    return (x_vec, exp_term_vec_s, recursive_term_vec,
            p_y_cond_x_vec, n_mutations)

@jit
def BD_process_dynamics (s, x_vec, exp_term_vec):
    lamb=1.3
    # Vectorised computation of BD process dynamics
    mean_vec = x_vec[:-1, : , :]*exp_term_vec
    variance_vec = x_vec[:-1]*(2*lamb + s)*exp_term_vec*(exp_term_vec-1)/s

    # Neg Binom parametrization
    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec-mean_vec)

    return p_vec, n_vec

def fitness_specific_computations (s_idx, s_vec, x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, time_points, n_mutations, cs):

    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    # Vectorised computation of BD process dynamics
    p_vec, n_vec = BD_process_dynamics(s,x_vec, exp_term_vec)

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

    # locate mutation specific data
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

    bd_pmf_i = jsp_stats.nbinom.pmf(x_i[j][:, None], p=p_i[j-1], n=n_i[j-1])
    inner_sum_i = bd_pmf_i*recursive_term_i
    recursive_term_i = p_y_cond_x_i[j]*jsp.integrate.trapezoid(x=x_i[j-1], y=inner_sum_i)

    return recursive_term_i

def compute_cs_posterior_grid_vec(s_vec, h_vec, AO, DP, time_points, cs):
    """Compute the clonal likelihood over the full (h, s) grid for a
    single clonal structure. h (homozygosity fraction) enters the
    deterministic clone size / VAF transformation, so we loop over it and
    recompute the deterministic sizes and global variables for each value."""

    outputs = []
    for h in h_vec:
        deterministic_size, total_cells = compute_deterministic_size(
            cs, AO, DP, AO.shape[1], h)

        out = jax_cs_hmm_ll_vec(s_vec, AO, DP, time_points, cs,
                                deterministic_size, total_cells, h)
        outputs.append(out)

    # output shape: (h_resolution, s_resolution, n_clones)
    return jnp.stack(outputs)

def compute_clonal_models_prob_vec (part, s_resolution=50, h_resolution=20,
                                    min_s=0.01, max_s=1, min_h=0., max_h=1.,
                                    filter_invalid=True, disable_progressbar=False):
    """Compute model probabilities for each clonal structure.
    Append results in unstructued dictionary."""


    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_vec = jnp.linspace(min_h, max_h, h_resolution)

    # Compute all possible clonal structures (or partition sets)
    n_mutations = part.shape[0]

    # initialise model dictionary
    part.uns['model_dict'] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns['warning'] = None
    if len(cs_list) > 100:
        part.uns['warning'] = 'Too many possible structures'

        # Set only possibility as all mutations inddependent
        cs_list = [[[i] for i in range(n_mutations)]]

    # Compute clonal structure probability
    for i, cs in tqdm(enumerate(cs_list), disable=disable_progressbar,
                      total=len(cs_list)):

        # compute clonal posteriors over the (h, s) grid
        output = compute_cs_posterior_grid_vec(s_vec, h_vec, AO, DP,
                                               time_points, cs)

        # compute clonal structure probability (marginalise s and h)
        model_prob = compute_model_likelihood(output, cs, s_vec, h_vec)

        # save model probability
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)

    part.uns['model_dict'] = {k: v for k, v in sorted(part.uns['model_dict'].items(), key=lambda item: item[1][1], reverse=True)}

    return part

def refine_optimal_model_posterior_vec(part, s_resolution=100, h_resolution=40,
                                       min_h=0., max_h=1.):
    # Compute finer posterior for optimal model
    # retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]

    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    # Create refined s_range and h_range
    s_vec = jnp.linspace(0.01, 1, s_resolution)
    h_vec = jnp.linspace(min_h, max_h, h_resolution)

    # compute clonal posteriors over the (h, s) grid
    output = compute_cs_posterior_grid_vec(s_vec, h_vec, AO, DP, time_points, cs)

    part.uns['optimal_model'] = {'clonal_structure': cs,
                                'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
                                'posterior': output,
                                's_range': s_vec,
                                'h_range': h_vec}

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    homozygosity = np.zeros(part.shape[0])
    homozygosity_5 = np.zeros(part.shape[0])
    homozygosity_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):

        # Extract posterior for each clone over the (h, s) grid
        p = np.array(output[:, :, i])
        if np.nansum(p).sum() == 0:
            part.uns['warning'] = 'Zero posterior'
            return part

        # marginal posteriors
        p_s = p.sum(axis=0)   # p(s)
        p_h = p.sum(axis=1)   # p(h)
        p_s = p_s/p_s.sum()
        p_h = p_h/p_h.sum()

        # maximum a posteriori estimates
        fitness_map = s_vec[np.argmax(p_s)]
        h_map = h_vec[np.argmax(p_h)]

        # 95% CI using bootstrap
        s_sample = np.random.choice(s_vec, p=p_s, size=1_000)
        h_sample = np.random.choice(h_vec, p=p_h, size=1_000)
        s_cfd = np.quantile(s_sample, [0.05, 0.95])
        h_cfd = np.quantile(h_sample, [0.05, 0.95])

        # Append information
        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = s_cfd[0]
        fitness_95[c_idx] = s_cfd[1]
        homozygosity[c_idx] = h_map
        homozygosity_5[c_idx] = h_cfd[0]
        homozygosity_95[c_idx] = h_cfd[1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['homozygosity'] = homozygosity
    part.obs['homozygosity_5'] = homozygosity_5
    part.obs['homozygosity_95'] = homozygosity_95
    part.obs['clonal_index'] = clonal_index


    # Append mutational structure to each mutation
    mut_structure = part.uns['optimal_model']['mutation_structure']

    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(
                    structure)

    part.obs['clonal_structure'] = clonal_structure_list

    return part

#endregion

def compute_invalid_combinations (part, pearson_distance_threshold=0.5):

    # Compute pearson of each mutation with with time
    correlation_matrix = np.corrcoef(
        np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]

    # Compute distance between pearsonr's
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])

    # label invalid combinations if pearson correlation is too different
    not_valid_comb = np.argwhere(distance_matrix
                                 >pearson_distance_threshold)
    not_valid_comb = [list(i) for i in not_valid_comb]

    # Extract unique tuples from list(Order Irrespective)
    # using list comprehension + set()
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i)

    part.uns['invalid_combinations'] = res

def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """
    Find all valid clonal structures using pearson correlation analysis"""

    n_mutations = part.shape[0]

    if n_mutations == 1:
        valid_cs = [[[0]]]
        return valid_cs

    else:
        if filter_invalid is True:
            # compute invalid clonal structures using correlation analysis
            compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)

        # create list of all possible clonal structures
        a = partition(list(range(n_mutations)))
        cs_list = [cs for cs in a]

        if filter_invalid is False:
            return cs_list

        else:
            # find all valid clonal structures
            valid_cs = []

            for cs in cs_list:
                invalid_combinations_in_cs = 0
                for clone in cs:
                    # compute all pairs of mutations inside clone
                    mut_comb = list(combinations(clone, 2))
                    # check if any pair is invalid
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb
                            if list(comb) in part.uns['invalid_combinations']])

                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                # append valid clonal structure
                if invalid_combinations_in_cs == 0:
                    valid_cs.append(cs)

            return valid_cs
