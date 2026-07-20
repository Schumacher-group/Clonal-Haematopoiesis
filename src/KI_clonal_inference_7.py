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


# =============================================================================
# Non-vectorised original functions
# =============================================================================

def compute_deterministic_size(cs, AO, DP, n_mutations):
    """
    Original heterozygous deterministic size function.
    Kept for backwards compatibility.
    """
    N_w = 1e5

    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    deterministic_clone_size = jnp.array(
        -N_w * jnp.sum((AO / DP)[:, lm], axis=1)
        / (jnp.sum((AO / DP)[:, lm], axis=1) - 0.5)
    )

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO / DP * 2 * total_cells[:, None]

    return deterministic_size, total_cells


def jax_cs_hmm_ll(
    s,
    AO,
    DP,
    time_points,
    cs,
    lamb=1.3,
):
    """
    Original non-vectorised heterozygous likelihood.
    Kept for backwards compatibility.
    """
    N_w = 1e5
    n_mutations = AO.shape[1]

    leading_mutation_in_cs_idx = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO / DP)[:, cs_idx].sum(axis=0))
        leading_mutation_in_cs_idx.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    mutation_likelihood = jnp.zeros(n_mutations)

    deterministic_clone_size = jnp.array(
        -N_w * jnp.sum((AO / DP)[:, leading_mutation_in_cs_idx], axis=1)
        / (jnp.sum((AO / DP)[:, leading_mutation_in_cs_idx], axis=1) - 0.5)
    )

    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO / DP * 2 * total_cells[:, None]

    for j in range(n_mutations):
        s_clone = s[clonal_map[j]]

        beta_p_rvs = jrnd.beta(
            key=key,
            a=AO[:, j][:, None] + 1,
            b=DP[:, j][:, None] - AO[:, j][:, None] + 1,
            shape=(AO.shape[0], 1_000),
        )

        beta_p_rvs = jnp.sort(beta_p_rvs)

        N_w_cond = (total_cells - deterministic_size[:, j])[:, None]

        x_range = jnp.array(-N_w_cond * beta_p_rvs / (beta_p_rvs - 0.5))
        x = jnp.ceil(x_range)

        true_vaf = x / (2 * (N_w_cond + x))

        delta_t = jnp.diff(jnp.array(time_points))

        x_weight = jnp.repeat(1 / x[0].shape[0], x[0].shape[0])
        recursive_term = jsp_stats.binom.pmf(
            AO[0, j],
            n=DP[0, j],
            p=true_vaf[0],
        ) * x_weight

        for i in range(1, x.shape[0]):
            init_size = x[i - 1]
            next_size = x[i]
            p_y_cond_x = jsp_stats.binom.pmf(
                AO[i, j],
                n=DP[i, j],
                p=true_vaf[i],
            )

            exp_term = jnp.exp(delta_t[i - 1] * s_clone)
            mean = init_size * exp_term
            variance = init_size * (2 * lamb + s_clone) * exp_term * (exp_term - 1) / s_clone

            p = mean / variance
            n = jnp.power(mean, 2) / (variance - mean)

            bd_pmf = jsp_stats.nbinom.pmf(next_size[:, None], p=p, n=n)

            inner_sum = bd_pmf * recursive_term
            recursive_term = p_y_cond_x * jsp.integrate.trapezoid(
                x=x[i - 1],
                y=inner_sum,
            )

        likelihood = jsp.integrate.trapezoid(x=x[-1], y=recursive_term)
        mutation_likelihood = mutation_likelihood.at[j].set(likelihood)

    clonal_likelihood = jnp.zeros(len(cs))

    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)])
        )

    return clonal_likelihood


def partition(collection):
    """Module computing an iterable over all partitions of a set."""
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]

    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]

        yield [[first]] + smaller


def single_cs_posterior(part, cs, s_resolution=100):
    """Compute posterior for a single clonal structure using original model."""
    s_range = jnp.linspace(0.01, 1, s_resolution)
    multi_s_range = jnp.broadcast_to(s_range, (len(cs), s_resolution)).T

    output = jax.vmap(
        jax_cs_hmm_ll,
        in_axes=(0, None, None, None, None, None),
    )(
        multi_s_range,
        part.layers["AO"].T,
        part.layers["DP"].T,
        part.var.time_points,
        cs,
        1.3,
    )

    return output, s_range


def compute_model_likelihood(output, cs, s_range):
    """
    Compute model probability by marginalising clone-specific fitness.

    output shape: S,C
    """
    clonal_prob = np.zeros(len(cs))

    s_range_np = np.asarray(s_range)
    output_np = np.asarray(output)

    s_range_size = s_range_np.max() - s_range_np.min()
    s_prior = 1 / s_range_size

    for i, out in enumerate(output_np.T):
        clonal_prob[i] = s_prior * np.trapz(x=s_range_np, y=out)

    model_probability = np.prod(clonal_prob)

    return model_probability


def compute_clonal_models_prob(part, s_resolution=50):
    """Original non-vectorised model probability computation."""
    n_mutations = part.shape[0]
    a = partition(list(range(n_mutations)))
    part.uns["model_dict"] = {}

    for i, cs in tqdm(enumerate(a)):
        output, s_range = single_cs_posterior(part, cs, s_resolution)
        model_prob = compute_model_likelihood(output, cs, s_range)

        part.uns["model_dict"][f"model_{i}"] = (cs, model_prob)

    part.uns["model_dict"] = {
        k: v
        for k, v in sorted(
            part.uns["model_dict"].items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    }

    return part


def refine_optimal_model_posterior(part, s_resolution=100):
    """Original non-vectorised optimal posterior refinement."""
    cs = list(part.uns["model_dict"].values())[0][0]

    output, s_range = single_cs_posterior(part, cs, s_resolution)

    part.uns["optimal_model"] = {
        "clonal_structure": cs,
        "mutation_structure": [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        "posterior": output,
        "s_range": s_range,
    }

    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):
        p = np.array(output[:, i])
        p /= p.sum()

        sample_range = np.random.choice(s_range, p=p, size=1_000)
        fitness_map = s_range[np.argmax(p)]
        cfd_int = np.quantile(sample_range, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = cfd_int[0]
        fitness_95[c_idx] = cfd_int[1]
        clonal_index[c_idx] = i

    part.obs["fitness"] = fitness
    part.obs["fitness_5"] = fitness_5
    part.obs["fitness_95"] = fitness_95
    part.obs["clonal_index"] = clonal_index

    return part


def plot_optimal_model(part):
    """Original plotting function."""
    if part.uns["warning"] is not None:
        print("WARNING: " + part.uns["warning"])

    model = part.uns["optimal_model"]
    output = model["posterior"]
    cs = model["clonal_structure"]
    s_range = model["s_range"]

    norm_max = np.max(output, axis=0)

    for i in range(len(cs)):
        p_key_str = ""

        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f"{part[j].obs.p_key.values[0]}"
            if k > 0:
                p_key_str += f"\n{part[j].obs.p_key.values[0]}"

        sns.lineplot(
            x=s_range,
            y=output[:, i] / norm_max[i],
            label=p_key_str,
        )


# =============================================================================
# Original vectorised heterozygous model
# Kept for backwards compatibility
# =============================================================================

def jax_cs_hmm_ll_vec(
    s_vec,
    AO,
    DP,
    time_points,
    cs,
    deterministic_size,
    total_cells,
):
    global_variables = compute_global_variables(
        s_vec,
        AO,
        DP,
        total_cells,
        deterministic_size,
        time_points,
    )

    x_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mutations = global_variables

    s_idx = jnp.arange(s_vec.shape[0])

    clonal_likelihood = jax.vmap(
        fitness_specific_computations,
        in_axes=(0, None, None, None, None, None, None, None, None),
    )(
        s_idx,
        s_vec,
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        time_points,
        n_mutations,
        cs,
    )

    return clonal_likelihood



def compute_global_variables(
    s_vec,
    AO,
    DP,
    total_cells,
    deterministic_size,
    time_points,
    resolution=1_000,
):
    n_mutations = AO.shape[1]

    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    beta_p_rvs_vec = jrnd.beta(
        key=key,
        a=(AO + 1)[:, :, None],
        b=DP[:, :, None] - AO[:, :, None] + 1,
        shape=(AO.shape[0], AO.shape[1], resolution),
    )

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)

    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    x_range_vec = jnp.array(-N_w_cond_vec * beta_p_rvs_vec / (beta_p_rvs_vec - 0.5))
    x_vec = jnp.ceil(x_range_vec)

    true_vaf_vec = x_vec / (2 * (N_w_cond_vec + x_vec))

    p_y_cond_x_vec = jsp_stats.binom.pmf(
        AO[:, :, None],
        n=DP[:, :, None],
        p=true_vaf_vec,
    )

    recursive_term_vec = p_y_cond_x_vec[0, :, :] * 1 / resolution

    return (
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        n_mutations,
    )


@jit
def BD_process_dynamics(s, x_vec, exp_term_vec):
    lamb = 1.3

    mean_vec = x_vec[:-1, :, :] * exp_term_vec
    variance_vec = x_vec[:-1] * (2 * lamb + s) * exp_term_vec * (exp_term_vec - 1) / s

    p_vec = mean_vec / variance_vec
    n_vec = jnp.power(mean_vec, 2) / (variance_vec - mean_vec)

    return p_vec, n_vec


def fitness_specific_computations(
    s_idx,
    s_vec,
    x_vec,
    exp_term_vec_s,
    recursive_term_vec,
    p_y_cond_x_vec,
    time_points,
    n_mutations,
    cs,
):
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    p_vec, n_vec = BD_process_dynamics(s, x_vec, exp_term_vec)

    mutation_likelihood = jax.vmap(
        mutation_specific_ll,
        in_axes=(0, None, None, None, None, None, None),
    )(
        jnp.arange(n_mutations, dtype=int),
        recursive_term_vec,
        x_vec,
        p_vec,
        n_vec,
        p_y_cond_x_vec,
        time_points.shape[0],
    )

    clonal_likelihood = jnp.zeros(len(cs))

    for i, c_idx in enumerate(cs):
        clonal_likelihood = clonal_likelihood.at[i].set(
            np.prod(mutation_likelihood[jnp.array(c_idx)])
        )

    return clonal_likelihood


def mutation_specific_ll(
    i,
    recursive_term_vec,
    x_vec,
    p_vec,
    n_vec,
    p_y_cond_x_vec,
    n_tps,
):
    recursive_term_i = recursive_term_vec[i]
    x_i = x_vec[:, i]
    p_i = p_vec[:, i]
    n_i = n_vec[:, i]
    p_y_cond_x_i = p_y_cond_x_vec[:, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(
            j,
            recursive_term_i,
            x_i,
            p_i,
            n_i,
            p_y_cond_x_i,
        )

    return jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)


@jit
def recursive_term_update(j, recursive_term_i, x_i, p_i, n_i, p_y_cond_x_i):
    """Update the recursive term associated with mutation i for data point j."""
    bd_pmf_i = jsp_stats.nbinom.pmf(
        x_i[j][:, None],
        p=p_i[j - 1],
        n=n_i[j - 1],
    )

    inner_sum_i = bd_pmf_i * recursive_term_i

    recursive_term_i = p_y_cond_x_i[j] * jsp.integrate.trapezoid(
        x=x_i[j - 1],
        y=inner_sum_i,
    )

    return recursive_term_i


def compute_clonal_models_prob_vec(
    part,
    s_resolution=50,
    min_s=0.01,
    max_s=1,
    filter_invalid=True,
    disable_progressbar=False,
):
    """Original vectorised heterozygous model."""
    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var.time_points)
    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mutations = part.shape[0]

    part.uns["model_dict"] = {}

    cs_list = find_valid_clonal_structures(part, filter_invalid=filter_invalid)

    part.uns["warning"] = None

    if len(cs_list) > 100:
        part.uns["warning"] = "Too many possible structures"
        cs_list = [[[i] for i in range(n_mutations)]]

    for i, cs in tqdm(
        enumerate(cs_list),
        disable=disable_progressbar,
        total=len(cs_list),
    ):
        deterministic_size, total_cells = compute_deterministic_size(
            cs,
            AO,
            DP,
            AO.shape[1],
        )

        output = jax_cs_hmm_ll_vec(
            s_vec,
            AO,
            DP,
            time_points,
            cs,
            deterministic_size,
            total_cells,
        )

        model_prob = compute_model_likelihood(output, cs, s_vec)

        part.uns["model_dict"][f"model_{i}"] = (cs, model_prob)

    part.uns["model_dict"] = {
        k: v
        for k, v in sorted(
            part.uns["model_dict"].items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    }

    return part


def refine_optimal_model_posterior_vec(part, s_resolution=100):
    """Original vectorised heterozygous optimal model refinement."""
    cs = list(part.uns["model_dict"].values())[0][0]

    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var.time_points)

    deterministic_size, total_cells = compute_deterministic_size(
        cs,
        AO,
        DP,
        AO.shape[1],
    )

    s_vec = jnp.linspace(0.01, 1, s_resolution)

    output = jax_cs_hmm_ll_vec(
        s_vec,
        AO,
        DP,
        time_points,
        cs,
        deterministic_size,
        total_cells,
    )

    part.uns["optimal_model"] = {
        "clonal_structure": cs,
        "mutation_structure": [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        "posterior": output,
        "s_range": s_vec,
    }

    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):
        p = np.array(output[:, i])

        if np.nansum(p).sum() == 0:
            part.uns["warning"] = "Zero posterior"
            return part

        p /= p.sum()

        fitness_map = s_vec[np.argmax(p)]

        sample_range = np.random.choice(s_vec, p=p, size=1_000)
        cfd_int = np.quantile(sample_range, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = cfd_int[0]
        fitness_95[c_idx] = cfd_int[1]
        clonal_index[c_idx] = i

    part.obs["fitness"] = fitness
    part.obs["fitness_5"] = fitness_5
    part.obs["fitness_95"] = fitness_95
    part.obs["clonal_index"] = clonal_index

    mut_structure = part.uns["optimal_model"]["mutation_structure"]

    clonal_structure_list = []

    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)

    part.obs["clonal_structure"] = clonal_structure_list

    return part


# =============================================================================
# Clone-level LOH / mixed zygosity model
# =============================================================================

def mixed_loh_clone_size_to_vaf(x, N_w, h):
    """
    Mixed heterozygous/LOH VAF model.

    V = (1+h)x / [2(Nw+x)]

    h = fraction of mutant clone cells that are homozygous/LOH.
    h=0 gives the heterozygous model.
    h=1 gives the homozygous/complete LOH model.
    """
    return (1.0 + h) * x / (2.0 * (N_w + x))


def mixed_loh_vaf_to_clone_size(vaf, N_w, h, eps=1e-8):
    """
    Inverse mixed-LOH VAF model.

    x = 2 V Nw / (1+h-2V)
    """
    denom = 1.0 + h - 2.0 * vaf
    valid = denom > eps

    x = 2.0 * vaf * N_w / jnp.maximum(denom, eps)
    x = jnp.ceil(x)
    x = jnp.where(valid, x, jnp.nan)

    return x, valid


def compute_leading_mutations_and_clonal_map(cs, AO, DP, n_mutations):
    """
    Determine leading mutation for each clone and map each mutation to clone index.

    AO, DP shape: T,M
    cs: list of clone mutation-index lists
    """
    vaf = AO / DP

    leading_mutations = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for c, cs_idx in enumerate(cs):
        cs_idx_arr = jnp.array(cs_idx)
        max_idx = jnp.argmax(vaf[:, cs_idx_arr].sum(axis=0))
        leading_mut = cs_idx_arr[max_idx]

        leading_mutations.append(int(leading_mut))
        clonal_map = clonal_map.at[cs_idx_arr].set(c)

    return leading_mutations, clonal_map


def compute_clone_h_min(
    AO,
    DP,
    cs,
    use_leading_only=True,
    h_margin=0.02,
):
    """
    Compute clone-level minimum h required by observed VAF.

        h_min = max(0, 2*VAF_max - 1)

    If h_min > 0, add h_margin to avoid the exact boundary:

        1 + h - 2V = 0

    If use_leading_only=True, h_min is computed from the leading mutation.
    If False, h_min is computed from all mutations assigned to the clone.
    """
    vaf = AO / DP
    h_mins = []

    for cs_idx in cs:
        cs_idx_arr = jnp.array(cs_idx)

        if use_leading_only:
            max_idx = jnp.argmax(vaf[:, cs_idx_arr].sum(axis=0))
            leading_mut = cs_idx_arr[max_idx]
            vmax = jnp.max(vaf[:, leading_mut])
        else:
            vmax = jnp.max(vaf[:, cs_idx_arr])

        raw_h_min = 2.0 * vmax - 1.0

        h_min = jnp.where(
            raw_h_min > 0.0,
            raw_h_min + h_margin,
            0.0,
        )

        h_min = jnp.clip(h_min, 0.0, 0.98)

        h_mins.append(h_min)

    return jnp.array(h_mins)



def make_h_prior_by_clone(
    h_vec,
    h_min_by_clone,
    prefer_low=True,
    low_h_strength=3.0,
):
    """
    Construct clone-specific priors over h.

    h_vec shape: H
    h_min_by_clone shape: C

    returns shape: C,H
    """
    priors = []

    for h_min in h_min_by_clone:
        prior = jnp.where(h_vec >= h_min, 1.0, 0.0)

        if prefer_low:
            prior = prior * jnp.exp(-low_h_strength * h_vec)

        prior_sum = prior.sum()

        prior = jnp.where(
            prior_sum > 0,
            prior / prior_sum,
            jnp.ones_like(h_vec) / h_vec.shape[0],
        )

        priors.append(prior)

    return jnp.stack(priors)


def get_clone_representative_vaf(AO, DP, cs):
    """
    Use leading mutation from each clone as representative clone VAF.

    AO, DP shape: T,M

    returns shape: T,C
    """
    vaf = AO / DP
    clone_vafs = []

    for cs_idx in cs:
        cs_idx_arr = jnp.array(cs_idx)
        max_idx = jnp.argmax(vaf[:, cs_idx_arr].sum(axis=0))
        leading_mut = cs_idx_arr[max_idx]
        clone_vafs.append(vaf[:, leading_mut])

    return jnp.stack(clone_vafs, axis=1)


def competing_vaf_sum_valid(AO, DP, cs, threshold=1.0):
    """
    Reject clonal structures whose representative competing clone VAFs
    sum above threshold at any time point.

    This is appropriate for partition-based, mutually exclusive clones.
    If nested/subclonal trees are introduced later, this should be replaced
    with a tree-aware cellular prevalence constraint.
    """
    clone_vaf = get_clone_representative_vaf(AO, DP, cs)
    return bool(jnp.all(jnp.sum(clone_vaf, axis=1) <= threshold))


def compute_deterministic_size_loh(cs, AO, DP, n_mutations, h_by_clone):
    """
    LOH-aware deterministic size approximation.

    Each clone has a fixed h value.
    Leading mutation determines clone size.
    Piggybacking mutations use the same clone h.

    AO, DP shape: T,M
    h_by_clone shape: C

    returns:
        deterministic_size: T,M
        total_cells: T
        clonal_map: M
        leading_mutations: list length C
        clone_sizes: T,C
    """
    N_w = 1e5
    vaf = AO / DP

    leading_mutations = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    clone_sizes = []

    for c, cs_idx in enumerate(cs):
        cs_idx_arr = jnp.array(cs_idx)

        max_idx = jnp.argmax(vaf[:, cs_idx_arr].sum(axis=0))
        leading_mut = cs_idx_arr[max_idx]
        leading_mutations.append(int(leading_mut))

        clonal_map = clonal_map.at[cs_idx_arr].set(c)

        V = vaf[:, leading_mut]
        h = h_by_clone[c]

        denom = 1.0 + h - 2.0 * V
        x_c = 2.0 * V * N_w / jnp.maximum(denom, 1e-8)
        x_c = jnp.ceil(x_c)
        x_c = jnp.where(denom > 1e-8, x_c, jnp.nan)

        clone_sizes.append(x_c)

    clone_sizes = jnp.stack(clone_sizes, axis=1)  # T,C

    total_cells = N_w + jnp.nansum(clone_sizes, axis=1)

    h_mut = h_by_clone[clonal_map]

    denom_mut = 1.0 + h_mut[None, :] - 2.0 * vaf
    deterministic_size = 2.0 * vaf * N_w / jnp.maximum(denom_mut, 1e-8)
    deterministic_size = jnp.ceil(deterministic_size)
    deterministic_size = jnp.where(denom_mut > 1e-8, deterministic_size, jnp.nan)

    return deterministic_size, total_cells, clonal_map, leading_mutations, clone_sizes


def compute_global_variables_loh(
    s_vec,
    AO,
    DP,
    total_cells,
    deterministic_size,
    time_points,
    h_vec,
    clonal_map,
    resolution=300,
):
    """
    Compute global variables with clone-level LOH fraction grid.

    Uses constant N_w=1e5, matching the biological model:

        V = (1+h)x / [2(N_w+x)]

    Important numerical choices:
        - x_vec is always finite.
        - invalid h/VAF combinations get zero observation likelihood.
        - N_w is not total_cells - deterministic_size.

    NOTE:
        This function is intentionally not jitted because `resolution` is used
        as an array shape in jax.random.beta.
    """
    N_w = 1e5
    resolution = int(resolution)

    n_mutations = AO.shape[1]
    n_h = h_vec.shape[0]

    delta_t = jnp.diff(time_points)

    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(
        exp_term_vec_s,
        (*exp_term_vec_s.shape, 1, 1),
    )

    beta_p_rvs_vec = jrnd.beta(
        key=key,
        a=(AO + 1)[:, :, None],
        b=DP[:, :, None] - AO[:, :, None] + 1,
        shape=(AO.shape[0], AO.shape[1], resolution),
    )

    beta_p_rvs_vec = jnp.sort(beta_p_rvs_vec)

    # Use constant N_w background, as in:
    # V = (1+h)x / [2(N_w+x)]
    N_w_cond_vec = jnp.ones_like(AO) * N_w
    N_w_cond_vec = N_w_cond_vec[:, :, None]

    # Add h axis.
    vaf = beta_p_rvs_vec[None, :, :, :]       # H,T,M,R
    Nw = N_w_cond_vec[None, :, :, :]          # H,T,M,R
    h = h_vec[:, None, None, None]            # H,1,1,1

    denom = 1.0 + h - 2.0 * vaf

    # h/VAF combinations below this boundary are impossible.
    valid = denom > 1e-6

    denom_safe = jnp.maximum(denom, 1e-6)

    x_vec = 2.0 * vaf * Nw / denom_safe
    x_vec = jnp.ceil(x_vec)

    # Keep finite values, otherwise BD transition can become all NaN.
    x_vec = jnp.nan_to_num(
        x_vec,
        nan=1.0,
        posinf=1e12,
        neginf=1.0,
    )

    x_vec = jnp.clip(x_vec, 1.0, 1e12)

    true_vaf_vec = (1.0 + h) * x_vec / (2.0 * (Nw + x_vec))
    true_vaf_vec = jnp.clip(true_vaf_vec, 1e-9, 1.0 - 1e-9)

    p_y_cond_x_vec = jsp_stats.binom.pmf(
        AO[None, :, :, None],
        n=DP[None, :, :, None],
        p=true_vaf_vec,
    )

    # Invalid h,V combinations contribute zero likelihood.
    p_y_cond_x_vec = jnp.where(valid, p_y_cond_x_vec, 0.0)

    p_y_cond_x_vec = jnp.nan_to_num(
        p_y_cond_x_vec,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    recursive_term_vec = p_y_cond_x_vec[:, 0, :, :] * (1.0 / resolution)

    return (
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        n_mutations,
        n_h,
    )



@jit
def BD_process_dynamics_loh(s, x_vec, exp_term_vec):
    """
    Birth-death dynamics with LOH h axis.

    x_vec shape: H,T,M,R
    exp_term_vec shape after indexing s: T-1,1,1
    """
    lamb = 1.3

    exp_term = exp_term_vec[None, :, :, :]  # 1,T-1,1,1

    mean_vec = x_vec[:, :-1, :, :] * exp_term

    variance_vec = (
        x_vec[:, :-1, :, :]
        * (2.0 * lamb + s)
        * exp_term
        * (exp_term - 1.0)
        / s
    )

    mean_vec = jnp.nan_to_num(
        mean_vec,
        nan=1.0,
        posinf=1e12,
        neginf=1.0,
    )

    variance_vec = jnp.nan_to_num(
        variance_vec,
        nan=2.0,
        posinf=1e14,
        neginf=2.0,
    )

    mean_vec = jnp.maximum(mean_vec, 1e-6)

    # Negative-binomial parameterisation needs variance > mean.
    variance_vec = jnp.maximum(variance_vec, mean_vec + 1e-6)

    p_vec = mean_vec / variance_vec
    p_vec = jnp.clip(p_vec, 1e-8, 1.0 - 1e-8)

    n_vec = jnp.power(mean_vec, 2) / jnp.maximum(
        variance_vec - mean_vec,
        1e-6,
    )
    n_vec = jnp.maximum(n_vec, 1e-6)

    p_vec = jnp.nan_to_num(
        p_vec,
        nan=1e-8,
        posinf=1.0 - 1e-8,
        neginf=1e-8,
    )

    n_vec = jnp.nan_to_num(
        n_vec,
        nan=1e-6,
        posinf=1e12,
        neginf=1e-6,
    )

    return p_vec, n_vec



def mutation_specific_ll_loh(
    h_idx,
    i,
    recursive_term_vec,
    x_vec,
    p_vec,
    n_vec,
    p_y_cond_x_vec,
    n_tps,
):
    """
    Mutation likelihood conditional on h index.

    recursive_term_vec shape: H,M,R
    x_vec shape: H,T,M,R
    p_vec, n_vec shape: H,T-1,M,R
    p_y_cond_x_vec shape: H,T,M,R
    """
    recursive_term_i = recursive_term_vec[h_idx, i]
    x_i = x_vec[h_idx, :, i]
    p_i = p_vec[h_idx, :, i]
    n_i = n_vec[h_idx, :, i]
    p_y_cond_x_i = p_y_cond_x_vec[h_idx, :, i]

    for j in range(1, n_tps):
        recursive_term_i = recursive_term_update(
            j,
            recursive_term_i,
            x_i,
            p_i,
            n_i,
            p_y_cond_x_i,
        )

    likelihood = jsp.integrate.trapezoid(x=x_i[-1], y=recursive_term_i)

    likelihood = jnp.nan_to_num(
        likelihood,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return likelihood


def fitness_specific_computations_loh(
    s_idx,
    s_vec,
    x_vec,
    exp_term_vec_s,
    recursive_term_vec,
    p_y_cond_x_vec,
    time_points,
    n_mutations,
    n_h,
    cs,
    h_prior_by_clone,
):
    """
    Compute clone likelihood at one fitness-grid index, marginalising
    clone-level h.

    mutations in the same clone share the same h.
    """
    s = s_vec[s_idx]
    exp_term_vec = exp_term_vec_s[s_idx]

    p_vec, n_vec = BD_process_dynamics_loh(s, x_vec, exp_term_vec)

    def ll_for_mutation(i):
        return jax.vmap(
            lambda h_idx: mutation_specific_ll_loh(
                h_idx,
                i,
                recursive_term_vec,
                x_vec,
                p_vec,
                n_vec,
                p_y_cond_x_vec,
                time_points.shape[0],
            )
        )(jnp.arange(n_h))

    mutation_likelihood_h = jax.vmap(ll_for_mutation)(
        jnp.arange(n_mutations, dtype=int)
    )
    # shape: M,H

    clonal_likelihood = jnp.zeros(len(cs))

    for c, c_idx in enumerate(cs):
        clone_likelihood_h = jnp.prod(
            mutation_likelihood_h[jnp.array(c_idx), :],
            axis=0,
        )
        # shape: H

        clone_likelihood_marginal_h = jnp.sum(
            clone_likelihood_h * h_prior_by_clone[c]
        )

        clonal_likelihood = clonal_likelihood.at[c].set(
            clone_likelihood_marginal_h
        )

    return clonal_likelihood, mutation_likelihood_h


def jax_cs_hmm_ll_vec_loh(
    s_vec,
    AO,
    DP,
    time_points,
    cs,
    deterministic_size,
    total_cells,
    h_vec,
    h_prior_by_clone,
    clonal_map,
    resolution=300,
):
    """
    Top-level vectorised likelihood with clone-level mixed LOH.

    Returns:
        clonal_likelihood shape: S,C
        mutation_likelihood_h shape: S,M,H
    """
    global_variables = compute_global_variables_loh(
        s_vec,
        AO,
        DP,
        total_cells,
        deterministic_size,
        time_points,
        h_vec,
        clonal_map,
        resolution=resolution,
    )

    (
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        n_mutations,
        n_h,
    ) = global_variables

    s_idx = jnp.arange(s_vec.shape[0])

    clonal_likelihood, mutation_likelihood_h = jax.vmap(
        fitness_specific_computations_loh,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None),
    )(
        s_idx,
        s_vec,
        x_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        time_points,
        n_mutations,
        n_h,
        cs,
        h_prior_by_clone,
    )

    clonal_likelihood = jnp.nan_to_num(
        clonal_likelihood,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mutation_likelihood_h = jnp.nan_to_num(
        mutation_likelihood_h,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return clonal_likelihood, mutation_likelihood_h



def compute_clone_h_posterior(
    mutation_likelihood_h,
    cs,
    s_vec,
    h_prior_by_clone,
):
    """
    Compute clone-level posterior over h.

    mutation_likelihood_h shape: S,M,H
    h_prior_by_clone shape: C,H

    returns:
        clone_h_posterior shape: C,H
    """
    clone_h_posteriors = []

    s_range_size = s_vec.max() - s_vec.min()
    s_prior = 1.0 / s_range_size

    for c, c_idx in enumerate(cs):
        clone_likelihood_h_s = jnp.prod(
            mutation_likelihood_h[:, jnp.array(c_idx), :],
            axis=1,
        )
        # shape: S,H

        marginal_h = s_prior * jnp.trapezoid(
            clone_likelihood_h_s,
            x=s_vec,
            axis=0,
        )
        # shape: H

        posterior_h = marginal_h * h_prior_by_clone[c]
        posterior_sum = posterior_h.sum()

        posterior_h = jnp.where(
            posterior_sum > 0,
            posterior_h / posterior_sum,
            h_prior_by_clone[c],
        )

        clone_h_posteriors.append(posterior_h)

    return jnp.stack(clone_h_posteriors)


def compute_clonal_models_prob_vec_loh(
    part,
    s_resolution=50,
    h_resolution=25,
    min_s=0.01,
    max_s=1.0,
    filter_invalid=True,
    disable_progressbar=False,
    use_leading_only_h_min=True,
    prefer_low_h=True,
    low_h_strength=3.0,
    reject_vaf_sum_gt_one=True,
    resolution=300,
    h_margin=0.02,
):
    """
    Compute clonal model probabilities under clone-level mixed LOH.

    h is treated as a clone-level nuisance parameter and marginalised.
    """
    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var.time_points)

    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    n_mutations = part.shape[0]

    part.uns["model_dict"] = {}

    cs_list = find_valid_clonal_structures(
        part,
        filter_invalid=filter_invalid,
    )

    part.uns["warning"] = None

    if len(cs_list) > 100:
        part.uns["warning"] = "Too many possible structures"
        cs_list = [[[i] for i in range(n_mutations)]]

    model_counter = 0

    for _, cs in tqdm(
        enumerate(cs_list),
        disable=disable_progressbar,
        total=len(cs_list),
    ):
        if reject_vaf_sum_gt_one:
            if not competing_vaf_sum_valid(AO, DP, cs, threshold=1.0):
                continue

        h_min_by_clone = compute_clone_h_min(
            AO,
            DP,
            cs,
            use_leading_only=use_leading_only_h_min,
            h_margin=h_margin,
        )

        h_prior_by_clone = make_h_prior_by_clone(
            h_vec,
            h_min_by_clone,
            prefer_low=prefer_low_h,
            low_h_strength=low_h_strength,
        )

        h_for_deterministic = h_min_by_clone

        (
            deterministic_size,
            total_cells,
            clonal_map,
            leading_mutations,
            clone_sizes,
        ) = compute_deterministic_size_loh(
            cs,
            AO,
            DP,
            AO.shape[1],
            h_for_deterministic,
        )

        output, mutation_likelihood_h = jax_cs_hmm_ll_vec_loh(
            s_vec,
            AO,
            DP,
            time_points,
            cs,
            deterministic_size,
            total_cells,
            h_vec,
            h_prior_by_clone,
            clonal_map,
            resolution=resolution,
        )

        model_prob = compute_model_likelihood(output, cs, s_vec)

        model_prob = float(
            np.nan_to_num(
                model_prob,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )

        part.uns["model_dict"][f"model_{model_counter}"] = (
            cs,
            model_prob,
            {
                "h_min_by_clone": h_min_by_clone,
                "h_prior_by_clone": h_prior_by_clone,
                "leading_mutations": leading_mutations,
                "resolution": resolution,
            },
        )

        model_counter += 1

    if len(part.uns["model_dict"]) == 0:
        part.uns["warning"] = "No valid LOH clonal structures after filtering"
        return part

    part.uns["model_dict"] = {
        k: v
        for k, v in sorted(
            part.uns["model_dict"].items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    }

    return part


def refine_optimal_model_posterior_vec_loh(
    part,
    s_resolution=100,
    h_resolution=50,
    use_leading_only_h_min=True,
    prefer_low_h=True,
    low_h_strength=3.0,
    min_s=0.01,
    max_s=1.0,
    resolution=300,
    h_margin=0.02,
):
    """
    Refine posterior for optimal clonal model under clone-level LOH.

    Adds to part.obs:
        fitness
        fitness_5
        fitness_95
        clonal_index
        loh_fraction
        loh_fraction_5
        loh_fraction_95
        clonal_structure
    """
    cs = list(part.uns["model_dict"].values())[0][0]

    AO = jnp.array(part.layers["AO"].T)
    DP = jnp.array(part.layers["DP"].T)
    time_points = jnp.array(part.var.time_points)

    s_vec = jnp.linspace(min_s, max_s, s_resolution)
    h_vec = jnp.linspace(0.0, 1.0, h_resolution)

    h_min_by_clone = compute_clone_h_min(
        AO,
        DP,
        cs,
        use_leading_only=use_leading_only_h_min,
        h_margin=h_margin,
    )

    h_prior_by_clone = make_h_prior_by_clone(
        h_vec,
        h_min_by_clone,
        prefer_low=prefer_low_h,
        low_h_strength=low_h_strength,
    )

    (
        deterministic_size,
        total_cells,
        clonal_map,
        leading_mutations,
        clone_sizes,
    ) = compute_deterministic_size_loh(
        cs,
        AO,
        DP,
        AO.shape[1],
        h_min_by_clone,
    )

    output, mutation_likelihood_h = jax_cs_hmm_ll_vec_loh(
        s_vec,
        AO,
        DP,
        time_points,
        cs,
        deterministic_size,
        total_cells,
        h_vec,
        h_prior_by_clone,
        clonal_map,
        resolution=resolution,
    )

    output = jnp.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mutation_likelihood_h = jnp.nan_to_num(
        mutation_likelihood_h,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    clone_h_posterior = compute_clone_h_posterior(
        mutation_likelihood_h,
        cs,
        s_vec,
        h_prior_by_clone,
    )

    clone_h_posterior = jnp.nan_to_num(
        clone_h_posterior,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    part.uns["optimal_model"] = {
        "clonal_structure": cs,
        "mutation_structure": [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        "posterior": output,
        "s_range": s_vec,
        "h_range": h_vec,
        "h_min_by_clone": h_min_by_clone,
        "h_prior_by_clone": h_prior_by_clone,
        "h_posterior": clone_h_posterior,
        "leading_mutations": leading_mutations,
        "clone_sizes": clone_sizes,
        "resolution": resolution,
    }

    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    loh_fraction = np.zeros(part.shape[0])
    loh_fraction_5 = np.zeros(part.shape[0])
    loh_fraction_95 = np.zeros(part.shape[0])

    for c, c_idx in enumerate(cs):
        p = np.array(output[:, c])
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        if p.sum() == 0:
            part.uns["warning"] = "Zero posterior"
            return part

        p = p / p.sum()

        fitness_map = float(s_vec[np.argmax(p)])

        sample_s = np.random.choice(
            np.asarray(s_vec),
            p=p,
            size=1_000,
        )

        s_ci = np.quantile(sample_s, [0.05, 0.95])

        h_post = np.array(clone_h_posterior[c])
        h_post = np.nan_to_num(h_post, nan=0.0, posinf=0.0, neginf=0.0)

        if h_post.sum() == 0:
            h_post = np.array(h_prior_by_clone[c])

        h_post = h_post / h_post.sum()

        h_map = float(h_vec[np.argmax(h_post)])

        sample_h = np.random.choice(
            np.asarray(h_vec),
            p=h_post,
            size=1_000,
        )

        h_ci = np.quantile(sample_h, [0.05, 0.95])

        fitness[c_idx] = fitness_map
        fitness_5[c_idx] = s_ci[0]
        fitness_95[c_idx] = s_ci[1]
        clonal_index[c_idx] = c

        loh_fraction[c_idx] = h_map
        loh_fraction_5[c_idx] = h_ci[0]
        loh_fraction_95[c_idx] = h_ci[1]

    part.obs["fitness"] = fitness
    part.obs["fitness_5"] = fitness_5
    part.obs["fitness_95"] = fitness_95
    part.obs["clonal_index"] = clonal_index

    part.obs["loh_fraction"] = loh_fraction
    part.obs["loh_fraction_5"] = loh_fraction_5
    part.obs["loh_fraction_95"] = loh_fraction_95

    mut_structure = part.uns["optimal_model"]["mutation_structure"]

    clonal_structure_list = []

    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)
                break

    part.obs["clonal_structure"] = clonal_structure_list

    return part



def plot_optimal_model_loh(part):
    """
    Plot fitness posterior and clone-level LOH posterior for the optimal model.
    """
    if part.uns.get("warning") is not None:
        print("WARNING: " + str(part.uns["warning"]))

    model = part.uns["optimal_model"]

    output = model["posterior"]
    cs = model["clonal_structure"]
    s_range = model["s_range"]
    h_range = model["h_range"]
    h_posterior = model["h_posterior"]

    norm_max = np.max(output, axis=0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
        constrained_layout=True,
    )

    for c in range(len(cs)):
        label = ""

        for k, j in enumerate(cs[c]):
            if "p_key" in part.obs:
                mut_label = part[j].obs.p_key.values[0]
            else:
                mut_label = str(part.obs.index[j])

            if k == 0:
                label += f"{mut_label}"
            else:
                label += f"\n{mut_label}"

        axes[0].plot(
            np.asarray(s_range),
            np.asarray(output[:, c]) / norm_max[c],
            label=label,
        )

        axes[1].plot(
            np.asarray(h_range),
            np.asarray(h_posterior[c]),
            label=label,
        )

    axes[0].set_xlabel("Fitness s")
    axes[0].set_ylabel("Normalised posterior")
    axes[0].set_title("Fitness posterior")

    axes[1].set_xlabel("LOH fraction h")
    axes[1].set_ylabel("Posterior probability")
    axes[1].set_title("Clone-level LOH fraction posterior")

    axes[0].legend()
    axes[1].legend()

    return fig, axes


# =============================================================================
# Clonal-structure filtering utilities
# =============================================================================

def compute_invalid_combinations(part, pearson_distance_threshold=0.5):
    """
    Compute invalid mutation combinations using difference in correlation
    with time.
    """
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

    part.uns["invalid_combinations"] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """
    Find all valid clonal structures using Pearson correlation analysis.
    """
    n_mutations = part.shape[0]

    if n_mutations == 1:
        valid_cs = [[[0]]]
        return valid_cs

    else:
        if filter_invalid is True:
            compute_invalid_combinations(
                part,
                pearson_distance_threshold=p_distance_threshold,
            )

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
                        [
                            comb
                            for comb in mut_comb
                            if list(comb) in part.uns["invalid_combinations"]
                        ]
                    )

                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                if invalid_combinations_in_cs == 0:
                    valid_cs.append(cs)

            return valid_cs
