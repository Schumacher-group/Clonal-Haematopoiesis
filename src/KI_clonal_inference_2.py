from itertools import combinations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats

from scipy.stats import binom
from scipy.special import logsumexp
from tqdm import tqdm


# NumPy 2 compatibility.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


DEFAULT_MASTER_KEY_SEED = 758493
DEFAULT_WILD_TYPE_POPULATION = 1e5
DEFAULT_BIRTH_RATE = 1.3

EPS = 1e-8
LIKELIHOOD_FLOOR = 1e-300


# =============================================================================
# Basic VAF model
# =============================================================================

def vaf_from_xsh(x_tot, s_unused, h, N_w=DEFAULT_WILD_TYPE_POPULATION):
    """
    VAF model.

    h = homozygous/LOH fraction of the mutant population.

    x_het = (1-h) x_tot
    x_hom = h x_tot

    VAF = (x_het + 2*x_hom) / (2*(N_w+x_tot))
        = x_tot*(1+h) / (2*(N_w+x_tot))
    """
    return x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))


def x0_tot_from_vaf(vaf0, h, N_w=DEFAULT_WILD_TYPE_POPULATION):
    """
    Correct inversion:

        v0 = x0*(1+h)/(2*(N_w+x0))

    so:

        x0 = 2*v0*N_w / ((1+h) - 2*v0)
    """
    denom = (1.0 + h) - 2.0 * vaf0
    denom = np.maximum(denom, EPS)

    return 2.0 * vaf0 * N_w / denom


def project_clone_vaf(
    time_points,
    initial_mutant_cells,
    s,
    h,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
):
    """
    Deterministic VAF trajectory.
    """
    time_points = np.asarray(time_points, dtype=float)

    x_tot = initial_mutant_cells * np.exp(
        s * (time_points - time_points[0])
    )

    projected_vaf = x_tot * (1.0 + h) / (2.0 * (N_w + x_tot))
    projected_vaf = np.clip(projected_vaf, EPS, (1.0 + h) / 2.0 - EPS)

    return projected_vaf


# =============================================================================
# Slope utilities and valid structures
# =============================================================================

def _compute_slope_and_se(part):
    """
    Compute VAF/year slope and propagated binomial SE from first/last timepoint.
    """
    AO = np.asarray(part.layers["AO"], dtype=float)
    DP = np.asarray(part.layers["DP"], dtype=float)

    vaf = AO / np.maximum(DP, 1.0)
    time_points = np.asarray(part.var.time_points, dtype=float)

    t_range = time_points[-1] - time_points[0]

    if t_range <= EPS:
        slopes = np.zeros(part.shape[0])
        se_slope = np.ones(part.shape[0]) * np.inf
        return slopes, se_slope, t_range

    slopes = (vaf[:, -1] - vaf[:, 0]) / t_range

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


def compute_invalid_combinations(part, z_threshold=1.5):
    """
    Flag pairs whose VAF slopes differ significantly.
    """
    slopes, se_slope, t_range = _compute_slope_and_se(part)
    n = part.shape[0]

    invalid_pairs = []

    if t_range <= EPS:
        part.uns["invalid_combinations"] = []
        return

    for i in range(n):
        for j in range(i + 1, n):
            pooled_se = np.sqrt(se_slope[i] ** 2 + se_slope[j] ** 2)

            if pooled_se < 1e-10:
                is_invalid = abs(slopes[i] - slopes[j]) >= 0.01
            else:
                z = abs(slopes[i] - slopes[j]) / pooled_se
                is_invalid = z >= z_threshold

            if is_invalid:
                invalid_pairs.append([i, j])

    part.uns["invalid_combinations"] = invalid_pairs


def partition(collection):
    """
    Generate all set partitions.
    """
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]

    for smaller in partition(collection[1:]):
        for subset_index, subset in enumerate(smaller):
            yield (
                smaller[:subset_index]
                + [[first] + subset]
                + smaller[subset_index + 1:]
            )

        yield [[first]] + smaller


def find_valid_clonal_structures(
    part,
    z_threshold=1.5,
    filter_invalid=True,
    max_models=None,
):
    """
    Find valid flat clonal structures.

    Uses slope-based invalid-pair filtering.

    Important:
        This does NOT hard-filter by summed VAF > 1, because high summed VAF
        can be evidence for LOH/h=1 rather than biological impossibility.
    """
    n_mutations = part.shape[0]

    if n_mutations == 1:
        return [[[0]]]

    if filter_invalid:
        compute_invalid_combinations(part, z_threshold=z_threshold)

    cs_list = list(partition(list(range(n_mutations))))

    if not filter_invalid:
        valid = cs_list
    else:
        valid = []

        for cs in cs_list:
            bad = 0

            for clone in cs:
                pairs = list(combinations(clone, 2))

                bad += len(
                    [
                        pair
                        for pair in pairs
                        if list(pair) in part.uns["invalid_combinations"]
                    ]
                )

            if bad == 0:
                valid.append(cs)

    if max_models is not None:
        valid = valid[:max_models]

    return valid


def _slope_derived_structure(part, z_threshold=1.5):
    """
    Direct slope-derived structure using union-find.
    """
    slopes, se_slope, t_range = _compute_slope_and_se(part)
    n = part.shape[0]

    if n == 1 or t_range <= EPS:
        return [list(range(n))]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    for i in range(n):
        for j in range(i + 1, n):
            pooled_se = np.sqrt(se_slope[i] ** 2 + se_slope[j] ** 2)

            if pooled_se < 1e-10:
                merge = abs(slopes[i] - slopes[j]) < 0.01
            else:
                z = abs(slopes[i] - slopes[j]) / pooled_se
                merge = z < z_threshold

            if merge:
                parent[find(i)] = find(j)

    groups = {}

    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return sorted(
        groups.values(),
        key=lambda g: -float(np.mean(slopes[g])),
    )


# =============================================================================
# HMM deterministic initialisation
# =============================================================================

def compute_deterministic_size_mixed(
    cs,
    AO,
    DP,
    n_mutations,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
):
    """
    Compute deterministic sizes used to initialise the HMM.

    Uses correct VAF ceiling:

        v_max(h) = (1+h)/2

    and correct inversion:

        x_tot = N_w * v / ((1+h)/2 - v)
    """
    AO = jnp.asarray(AO, dtype=float)
    DP = jnp.asarray(DP, dtype=float)

    vaf_ratio = AO / jnp.maximum(DP, 1.0)

    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)

    for clone_index, clone_mutations in enumerate(cs):
        clone_vafs = vaf_ratio[:, clone_mutations]
        lead_idx_within_clone = int(jnp.argmax(clone_vafs.sum(axis=0)))
        lead_mutation = clone_mutations[lead_idx_within_clone]

        lm.append(lead_mutation)

        clonal_map = clonal_map.at[jnp.array(clone_mutations)].set(
            clone_index
        )

    leading_vaf_sum = jnp.sum(vaf_ratio[:, lm], axis=1)
    v_peak = jnp.max(leading_vaf_sum)

    # Choose h floor high enough that denominator stays away from zero.
    min_gap = 0.15
    h_min = jnp.maximum(0.0, 2.0 * (v_peak + min_gap) - 1.0)
    h_min = jnp.minimum(h_min, 1.0)

    denom = (1.0 + h_min) / 2.0 - leading_vaf_sum
    denom = jnp.where(jnp.abs(denom) < EPS, EPS, denom)

    deterministic_clone_size = jnp.ceil(N_w * leading_vaf_sum / denom)
    deterministic_clone_size = jnp.maximum(deterministic_clone_size, 100.0)
    deterministic_clone_size = jnp.minimum(deterministic_clone_size, N_w * 100.0)

    total_cells = N_w + deterministic_clone_size

    deterministic_size = (
        vaf_ratio
        * 2.0
        * total_cells[:, None]
        / (1.0 + h_min)
    )

    max_total_per_mutation = jnp.maximum(
        jnp.max(deterministic_size, axis=0) * 3.0,
        100.0,
    )

    max_total_per_mutation = jnp.minimum(
        max_total_per_mutation,
        N_w * 100.0,
    )

    return (
        deterministic_size.astype(jnp.float32),
        total_cells.astype(jnp.float32),
        max_total_per_mutation.astype(jnp.float32),
        clonal_map,
    )


# =============================================================================
# HMM likelihood
# =============================================================================

def compute_global_variables_mixed(
    s_vec,
    AO,
    DP,
    total_cells,
    deterministic_size,
    max_total_per_mutation,
    time_points,
    key,
    resolution=600,
):
    """
    HMM global variables.

    Uses beta posterior samples over VAF, transformed to x_total for each h
    implicitly through sampled homozygous fraction.
    """
    AO = jnp.asarray(AO, dtype=float)
    DP = jnp.asarray(DP, dtype=float)

    n_tps, n_mut = AO.shape

    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = exp_term_vec_s.reshape(
        (*exp_term_vec_s.shape, 1, 1)
    )

    key_beta, key_h = jrnd.split(key, 2)

    # Ordered beta quantile-like random sample.
    beta_p_rvs = jrnd.beta(
        key=key_beta,
        a=(AO + 1.0)[:, :, None],
        b=(DP - AO + 1.0)[:, :, None],
        shape=(n_tps, n_mut, resolution),
    )

    beta_p_rvs = jnp.clip(beta_p_rvs, EPS, 1.0 - EPS)

    # Sample h fraction on the latent grid.
    h_frac = jrnd.uniform(
        key_h,
        shape=(n_tps, n_mut, resolution),
        minval=0.0,
        maxval=1.0,
    )

    h_frac = jnp.clip(h_frac, EPS, 1.0 - EPS)

    N_w_cond = (total_cells[:, None] - deterministic_size)[:, :, None]
    N_w_cond = jnp.maximum(N_w_cond, EPS)

    # Invert sampled VAF using sampled h.
    denom = (1.0 + h_frac) / 2.0 - beta_p_rvs
    denom = jnp.where(jnp.abs(denom) < EPS, EPS, denom)

    x_total = N_w_cond * beta_p_rvs / denom
    x_total = jnp.clip(
        x_total,
        EPS,
        max_total_per_mutation[None, :, None],
    )

    x_hom = x_total * h_frac
    x_het = x_total * (1.0 - h_frac)

    true_vaf = x_total * (1.0 + h_frac) / (
        2.0 * (N_w_cond + x_total)
    )

    true_vaf = jnp.clip(true_vaf, EPS, 1.0 - EPS)

    log_p_y_cond_x = jsp_stats.binom.logpmf(
        AO[:, :, None],
        n=DP[:, :, None],
        p=true_vaf,
    )

    p_y_cond_x = jnp.exp(jnp.maximum(log_p_y_cond_x, -300.0))

    # Sort by total x along integration axis.
    sort_idx = jnp.argsort(x_total, axis=-1)

    x_total = jnp.take_along_axis(x_total, sort_idx, axis=-1)
    x_het = jnp.take_along_axis(x_het, sort_idx, axis=-1)
    x_hom = jnp.take_along_axis(x_hom, sort_idx, axis=-1)
    p_y_cond_x = jnp.take_along_axis(p_y_cond_x, sort_idx, axis=-1)

    recursive_term_vec = p_y_cond_x[0] * (1.0 / resolution)

    return (
        x_het,
        x_hom,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x,
        n_mut,
    )


def BD_process_dynamics_mixed(
    s,
    x_total_vec,
    exp_term_vec,
    lamb=DEFAULT_BIRTH_RATE,
):
    """
    Birth-death dynamics approximation with negative binomial transition.
    """
    s_safe = jnp.maximum(jnp.abs(s), EPS)

    mean_vec = x_total_vec[:-1] * exp_term_vec

    variance_vec = (
        x_total_vec[:-1]
        * (2.0 * lamb + s)
        * exp_term_vec
        * (exp_term_vec - 1.0)
        / s_safe
    )

    min_variance = mean_vec * 1.2 + 1e-6
    variance_vec = jnp.maximum(variance_vec, min_variance)

    p_vec = mean_vec / variance_vec
    p_vec = jnp.clip(p_vec, EPS, 1.0 - EPS)

    n_vec = mean_vec**2 / jnp.maximum(variance_vec - mean_vec, EPS)
    n_vec = jnp.maximum(n_vec, EPS)

    return p_vec, n_vec


def mutation_specific_ll_mixed_grid(
    i,
    recursive_term_vec,
    x_het_vec,
    x_hom_vec,
    p_vec,
    n_vec,
    p_y_cond_x_vec,
    n_tps,
):
    """
    Mutation-specific HMM likelihood.
    """
    recursive_term_i = recursive_term_vec[i]

    x_het_i = x_het_vec[:, i, :]
    x_hom_i = x_hom_vec[:, i, :]

    x_total_i = x_het_i + x_hom_i

    p_i = p_vec[:, i, :]
    n_i = n_vec[:, i, :]
    p_y_i = p_y_cond_x_vec[:, i, :]

    for j in range(1, n_tps):
        init_total = x_total_i[j - 1]
        next_total = x_total_i[j]

        log_bd_pmf = jsp_stats.nbinom.logpmf(
            next_total[:, None],
            p=p_i[j - 1][None, :],
            n=n_i[j - 1][None, :],
        )

        bd_pmf = jnp.exp(jnp.maximum(log_bd_pmf, -300.0))

        inner_sum = bd_pmf * recursive_term_i[None, :]

        inner_integrated = jsp.integrate.trapezoid(
            x=init_total,
            y=inner_sum,
            axis=1,
        )

        inner_integrated = jnp.maximum(
            inner_integrated,
            LIKELIHOOD_FLOOR,
        )

        recursive_term_i = (
            jnp.maximum(p_y_i[j], LIKELIHOOD_FLOOR)
            * inner_integrated
        )

        recursive_term_i = jnp.maximum(
            recursive_term_i,
            LIKELIHOOD_FLOOR,
        )

    final_like = jsp.integrate.trapezoid(
        x=x_total_i[-1],
        y=recursive_term_i,
    )

    final_like = jnp.maximum(final_like, LIKELIHOOD_FLOOR)

    return final_like


def jax_cs_hmm_ll_vec_mixed(
    s_vec,
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
    """
    HMM likelihood over s for each clone.
    """
    (
        x_het_vec,
        x_hom_vec,
        exp_term_vec_s,
        recursive_term_vec,
        p_y_cond_x_vec,
        n_mut,
    ) = compute_global_variables_mixed(
        s_vec,
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

    def for_one_s(si):
        s = s_vec[si]
        exp_term_vec = exp_term_vec_s[si]

        x_total_vec = x_het_vec + x_hom_vec

        p_vec, n_vec = BD_process_dynamics_mixed(
            s,
            x_total_vec,
            exp_term_vec,
        )

        mutation_likelihood = jax.vmap(
            lambda ii: mutation_specific_ll_mixed_grid(
                ii,
                recursive_term_vec,
                x_het_vec,
                x_hom_vec,
                p_vec,
                n_vec,
                p_y_cond_x_vec,
                time_points.shape[0],
            )
        )(jnp.arange(n_mut))

        clonal_likelihood = jnp.zeros(len(cs))

        for clone_index, clone_mutations in enumerate(cs):
            clone_mutations = jnp.array(clone_mutations)

            log_mut_liks = jnp.log(
                jnp.maximum(
                    mutation_likelihood[clone_mutations],
                    LIKELIHOOD_FLOOR,
                )
            )

            clonal_likelihood = clonal_likelihood.at[clone_index].set(
                jnp.exp(jnp.maximum(jnp.sum(log_mut_liks), -700.0))
            )

        return clonal_likelihood

    return jax.vmap(for_one_s)(s_idx)


def compute_model_log_likelihood(output, cs, s_range):
    """
    Compute model evidence in log-space.
    """
    s_range = np.asarray(s_range, dtype=float)
    ds = float(np.mean(np.diff(s_range)))

    log_s_prior = -np.log(float(s_range.max() - s_range.min()))

    total_log_like = 0.0

    for clone_index in range(len(cs)):
        grid = np.asarray(output[:, clone_index], dtype=float)

        grid = np.nan_to_num(
            grid,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        grid = np.maximum(grid, 0.0)

        log_grid = np.log(np.maximum(grid, LIKELIHOOD_FLOOR))

        clone_log_like = (
            logsumexp(log_grid)
            + np.log(ds)
            + log_s_prior
        )

        total_log_like += clone_log_like

    return float(total_log_like)


def compute_clonal_models_prob_vec_mixed(
    part,
    s_resolution=50,
    min_s=0.01,
    max_s=3.0,
    filter_invalid=True,
    disable_progressbar=False,
    resolution=600,
    master_key_seed=DEFAULT_MASTER_KEY_SEED,
    z_threshold=1.5,
    max_models=None,
):
    """
    Compute posterior probability over flat clonal structures.

    Stores model entries as:

        (clonal_structure, log_evidence, posterior_probability)
    """
    AO = jnp.asarray(part.layers["AO"].T, dtype=float)
    DP = jnp.asarray(part.layers["DP"].T, dtype=float)
    time_points = jnp.asarray(part.var.time_points, dtype=float)

    s_vec = jnp.linspace(min_s, max_s, s_resolution)

    n_mutations = part.shape[0]

    part.uns["model_dict"] = {}
    part.uns["warning"] = None

    cs_list = find_valid_clonal_structures(
        part,
        z_threshold=z_threshold,
        filter_invalid=filter_invalid,
        max_models=max_models,
    )

    if len(cs_list) == 0:
        part.uns["warning"] = "No valid clonal structures found"
        return part

    master_key = jrnd.PRNGKey(master_key_seed)
    keys = jrnd.split(master_key, len(cs_list))

    iterator = enumerate(cs_list)

    if not disable_progressbar:
        iterator = tqdm(
            iterator,
            total=len(cs_list),
            desc="Evaluating clonal structures",
        )

    log_evidences = []

    for model_index, cs in iterator:
        try:
            (
                deterministic_size,
                total_cells,
                max_total_per_mutation,
                _,
            ) = compute_deterministic_size_mixed(
                cs,
                AO,
                DP,
                n_mutations,
            )

            output = jax_cs_hmm_ll_vec_mixed(
                s_vec,
                AO,
                DP,
                time_points,
                cs,
                deterministic_size,
                total_cells,
                max_total_per_mutation,
                keys[model_index],
                resolution=resolution,
            )

            log_evidence = compute_model_log_likelihood(
                output,
                cs,
                np.asarray(s_vec),
            )

            if not np.isfinite(log_evidence):
                log_evidence = -np.inf

        except Exception as exc:
            log_evidence = -np.inf
            part.uns.setdefault("model_errors", {})[
                f"model_{model_index}"
            ] = repr(exc)

        log_evidences.append(log_evidence)

        part.uns["model_dict"][f"model_{model_index}"] = (
            cs,
            log_evidence,
            None,
        )

    log_evidences = np.asarray(log_evidences, dtype=float)

    finite_mask = np.isfinite(log_evidences)
    posterior_probs = np.zeros_like(log_evidences, dtype=float)

    if finite_mask.any():
        log_norm = logsumexp(log_evidences[finite_mask])
        posterior_probs[finite_mask] = np.exp(
            log_evidences[finite_mask] - log_norm
        )
    else:
        part.uns["warning"] = (
            "All HMM evidences non-finite. "
            "Using uniform posterior over valid structures."
        )
        posterior_probs[:] = 1.0 / len(posterior_probs)

    for idx, (model_name, entry) in enumerate(
        list(part.uns["model_dict"].items())
    ):
        cs, log_evidence, _ = entry

        part.uns["model_dict"][model_name] = (
            cs,
            log_evidence,
            posterior_probs[idx],
        )

    part.uns["model_dict"] = {
        k: v
        for k, v in sorted(
            part.uns["model_dict"].items(),
            key=lambda item: item[1][2],
            reverse=True,
        )
    }

    return part


# =============================================================================
# Deterministic posterior refinement for s/h
# =============================================================================

def _leading_mutation_for_clone(vaf_ratio, clone_mutations):
    clone_vafs = vaf_ratio[:, clone_mutations]
    lead_idx = int(np.argmax(clone_vafs.sum(axis=0)))

    return clone_mutations[lead_idx]


def _credible_interval(prob, grid, lo=0.05, hi=0.95):
    prob = np.asarray(prob, dtype=float)
    grid = np.asarray(grid, dtype=float)

    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.maximum(prob, 0.0)

    prob = prob / np.maximum(prob.sum(), LIKELIHOOD_FLOOR)

    cdf = np.cumsum(prob)

    return (
        float(np.interp(lo, cdf, grid)),
        float(np.interp(hi, cdf, grid)),
    )


def infer_sh_jointly_from_dynamics(
    cs,
    AO,
    DP,
    time_points,
    s_resolution=60,
    h_resolution=80,
    min_s=0.01,
    max_s=3.0,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
):
    """
    Correct deterministic joint inference over (s,h).

    Uses leading mutation per clone.
    """
    s_range = np.linspace(min_s, max_s, s_resolution)
    h_range = np.linspace(0.0, 1.0, h_resolution)

    AO = np.asarray(AO, dtype=float)
    DP = np.asarray(DP, dtype=float)
    time_points = np.asarray(time_points, dtype=float)

    vaf_ratio = AO / np.maximum(DP, 1.0)

    results = []

    for clone_index, clone_mutations in enumerate(cs):
        lead_mutation = _leading_mutation_for_clone(
            vaf_ratio,
            clone_mutations,
        )

        ao_clone = AO[:, lead_mutation]
        dp_clone = DP[:, lead_mutation]

        valid_mask = dp_clone > 0

        ao_valid = ao_clone[valid_mask]
        dp_valid = dp_clone[valid_mask]
        t_valid = time_points[valid_mask]

        observed_vaf = ao_valid / np.maximum(dp_valid, 1.0)
        observed_vaf = np.clip(observed_vaf, EPS, 1.0 - EPS)

        joint_log_likelihood = np.full(
            (s_resolution, h_resolution),
            -np.inf,
        )

        expected_grid = np.zeros(
            (s_resolution, h_resolution, len(t_valid)),
            dtype=float,
        )

        for s_idx, s in enumerate(s_range):
            for h_idx, h in enumerate(h_range):
                v0 = observed_vaf[0]

                denom = (1.0 + h) - 2.0 * v0

                if denom <= EPS:
                    continue

                x0_tot = 2.0 * v0 * N_w / denom
                x0_tot = max(x0_tot, 100.0)

                projected_vaf = project_clone_vaf(
                    t_valid,
                    x0_tot,
                    s,
                    h,
                    N_w=N_w,
                )

                expected_grid[s_idx, h_idx, :] = projected_vaf

                log_lik = 0.0

                for t_idx, p in enumerate(projected_vaf):
                    log_lik += binom.logpmf(
                        int(ao_valid[t_idx]),
                        int(dp_valid[t_idx]),
                        float(p),
                    )

                joint_log_likelihood[s_idx, h_idx] = log_lik

        finite = np.isfinite(joint_log_likelihood)

        if finite.any():
            joint_likelihood = np.zeros_like(joint_log_likelihood)
            max_ll = joint_log_likelihood[finite].max()

            joint_likelihood[finite] = np.exp(
                np.clip(joint_log_likelihood[finite] - max_ll, -700, 0)
            )

            joint_posterior = joint_likelihood / np.maximum(
                joint_likelihood.sum(),
                LIKELIHOOD_FLOOR,
            )
        else:
            joint_posterior = np.ones_like(joint_log_likelihood)
            joint_posterior = joint_posterior / joint_posterior.sum()

        s_posterior = joint_posterior.sum(axis=1)
        h_posterior = joint_posterior.sum(axis=0)

        s_posterior = s_posterior / np.maximum(
            s_posterior.sum(),
            LIKELIHOOD_FLOOR,
        )
        h_posterior = h_posterior / np.maximum(
            h_posterior.sum(),
            LIKELIHOOD_FLOOR,
        )

        map_flat = np.argmax(joint_posterior)
        s_map_idx, h_map_idx = np.unravel_index(
            map_flat,
            joint_posterior.shape,
        )

        s_map_joint = float(s_range[s_map_idx])
        h_map_joint = float(h_range[h_map_idx])

        # Marginal MAPs for reporting.
        s_map = float(s_range[np.argmax(s_posterior)])
        h_map = float(h_range[np.argmax(h_posterior)])

        s_ci = _credible_interval(s_posterior, s_range)
        h_ci = _credible_interval(h_posterior, h_range)

        projected_vaf = expected_grid[s_map_idx, h_map_idx, :]

        results.append(
            {
                "s_map": s_map,
                "h_map": h_map,
                "s_map_joint": s_map_joint,
                "h_map_joint": h_map_joint,
                "s_posterior": s_posterior,
                "h_posterior": h_posterior,
                "joint_posterior": joint_posterior,
                "s_range": s_range,
                "h_range": h_range,
                "s_ci": s_ci,
                "h_ci": h_ci,
                "leading_mutation_index": lead_mutation,
                "valid_mask": valid_mask,
                "time_points_valid": t_valid,
                "observed_vaf_valid": observed_vaf,
                "projected_vaf_valid": projected_vaf,
                "expected_grid": expected_grid,
                "used_sum_constraint": False,
            }
        )

    return results


def infer_sh_jointly_from_dynamics_with_sum_constraint(
    cs,
    AO,
    DP,
    time_points,
    s_resolution=60,
    h_resolution=80,
    min_s=0.01,
    max_s=3.0,
    N_w=DEFAULT_WILD_TYPE_POPULATION,
    sum_weight=0.25,
):
    """
    Joint (s,h) inference with an additional summed-VAF likelihood.

    Purpose:
        - s is still primarily inferred from individual clone dynamics.
        - h is additionally informed by the total/summed VAF trajectory.

    Warning:
        The summed VAF likelihood is not independent of individual mutation
        read likelihoods, so use a small sum_weight, e.g. 0.25 or 0.5.
    """
    base_results = infer_sh_jointly_from_dynamics(
        cs,
        AO,
        DP,
        time_points,
        s_resolution=s_resolution,
        h_resolution=h_resolution,
        min_s=min_s,
        max_s=max_s,
        N_w=N_w,
    )

    AO = np.asarray(AO, dtype=float)
    DP = np.asarray(DP, dtype=float)

    observed_vaf = AO / np.maximum(DP, 1.0)
    obs_sum_vaf = np.sum(observed_vaf, axis=1)
    obs_sum_vaf = np.clip(obs_sum_vaf, EPS, 1.0 - EPS)

    dp_sum = np.mean(DP, axis=1)
    ao_sum = np.round(obs_sum_vaf * dp_sum)

    refined = []

    # Individual MAP trajectories for other clones.
    map_trajectories = [
        r["projected_vaf_valid"]
        for r in base_results
    ]

    for clone_index, result in enumerate(base_results):
        joint_base = result["joint_posterior"]
        expected_grid = result["expected_grid"]

        # Convert posterior back to pseudo-log-likelihood up to constant.
        joint_log = np.log(
            np.maximum(joint_base, LIKELIHOOD_FLOOR)
        )

        valid_mask = result["valid_mask"]

        ao_sum_valid = ao_sum[valid_mask]
        dp_sum_valid = dp_sum[valid_mask]

        other_sum = np.zeros_like(result["projected_vaf_valid"])

        for j, traj in enumerate(map_trajectories):
            if j == clone_index:
                continue

            other_sum += traj

        s_resolution_i, h_resolution_i = joint_base.shape

        for s_idx in range(s_resolution_i):
            for h_idx in range(h_resolution_i):
                candidate_sum = other_sum + expected_grid[s_idx, h_idx, :]
                candidate_sum = np.clip(candidate_sum, EPS, 1.0 - EPS)

                sum_log_lik = 0.0

                for t_idx, p_sum in enumerate(candidate_sum):
                    sum_log_lik += binom.logpmf(
                        int(ao_sum_valid[t_idx]),
                        int(dp_sum_valid[t_idx]),
                        float(p_sum),
                    )

                joint_log[s_idx, h_idx] += sum_weight * sum_log_lik

        finite = np.isfinite(joint_log)

        if finite.any():
            joint_likelihood = np.zeros_like(joint_log)
            max_ll = joint_log[finite].max()

            joint_likelihood[finite] = np.exp(
                np.clip(joint_log[finite] - max_ll, -700, 0)
            )

            joint_posterior = joint_likelihood / np.maximum(
                joint_likelihood.sum(),
                LIKELIHOOD_FLOOR,
            )
        else:
            joint_posterior = joint_base

        s_posterior = joint_posterior.sum(axis=1)
        h_posterior = joint_posterior.sum(axis=0)

        s_posterior = s_posterior / np.maximum(
            s_posterior.sum(),
            LIKELIHOOD_FLOOR,
        )
        h_posterior = h_posterior / np.maximum(
            h_posterior.sum(),
            LIKELIHOOD_FLOOR,
        )

        s_range = result["s_range"]
        h_range = result["h_range"]

        map_flat = np.argmax(joint_posterior)
        s_map_idx, h_map_idx = np.unravel_index(
            map_flat,
            joint_posterior.shape,
        )

        projected_vaf = expected_grid[s_map_idx, h_map_idx, :]

        new_result = dict(result)
        new_result.update(
            {
                "s_map": float(s_range[np.argmax(s_posterior)]),
                "h_map": float(h_range[np.argmax(h_posterior)]),
                "s_map_joint": float(s_range[s_map_idx]),
                "h_map_joint": float(h_range[h_map_idx]),
                "s_posterior": s_posterior,
                "h_posterior": h_posterior,
                "joint_posterior": joint_posterior,
                "s_ci": _credible_interval(s_posterior, s_range),
                "h_ci": _credible_interval(h_posterior, h_range),
                "projected_vaf_valid": projected_vaf,
                "used_sum_constraint": True,
                "sum_weight": sum_weight,
            }
        )

        refined.append(new_result)

    return refined


def refine_optimal_model_posterior_vec(
    part,
    s_resolution=60,
    h_resolution=80,
    min_s=0.01,
    max_s=3.0,
    use_sum_constraint=True,
    sum_weight=0.25,
):
    """
    Refine optimal model posterior using deterministic joint (s,h) inference.

    Stores:
        part.obs['fitness']
        part.obs['zygosity']
        part.obs['clonal_index']
    """
    if "model_dict" not in part.uns or len(part.uns["model_dict"]) == 0:
        part.uns["warning"] = "No model_dict available"
        return part

    cs = list(part.uns["model_dict"].values())[0][0]

    AO = np.asarray(part.layers["AO"].T, dtype=float)
    DP = np.asarray(part.layers["DP"].T, dtype=float)
    time_points = np.asarray(part.var.time_points, dtype=float)

    if use_sum_constraint and len(cs) > 1:
        joint_results = infer_sh_jointly_from_dynamics_with_sum_constraint(
            cs,
            AO,
            DP,
            time_points,
            s_resolution=s_resolution,
            h_resolution=h_resolution,
            min_s=min_s,
            max_s=max_s,
            sum_weight=sum_weight,
        )
    else:
        joint_results = infer_sh_jointly_from_dynamics(
            cs,
            AO,
            DP,
            time_points,
            s_resolution=s_resolution,
            h_resolution=h_resolution,
            min_s=min_s,
            max_s=max_s,
        )

    s_range = joint_results[0]["s_range"]
    h_range = joint_results[0]["h_range"]

    posterior_2d = np.stack(
        [r["joint_posterior"] for r in joint_results],
        axis=2,
    )

    part.uns["optimal_model"] = {
        "clonal_structure": cs,
        "mutation_structure": [
            list(part.obs.iloc[clone_mutations].index)
            for clone_mutations in cs
        ],
        "posterior_2d": posterior_2d,
        "s_range": s_range,
        "h_range": h_range,
        "joint_inference": joint_results,
        "used_sum_constraint": use_sum_constraint,
        "sum_weight": sum_weight if use_sum_constraint else 0.0,
    }

    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])

    zygosity = np.zeros(part.shape[0])
    zygosity_5 = np.zeros(part.shape[0])
    zygosity_95 = np.zeros(part.shape[0])

    clonal_index = np.zeros(part.shape[0])

    for clone_index, clone_mutations in enumerate(cs):
        result = joint_results[clone_index]

        fitness[clone_mutations] = result["s_map"]
        fitness_5[clone_mutations] = result["s_ci"][0]
        fitness_95[clone_mutations] = result["s_ci"][1]

        zygosity[clone_mutations] = result["h_map"]
        zygosity_5[clone_mutations] = result["h_ci"][0]
        zygosity_95[clone_mutations] = result["h_ci"][1]

        clonal_index[clone_mutations] = clone_index

    part.obs["fitness"] = fitness
    part.obs["fitness_5"] = fitness_5
    part.obs["fitness_95"] = fitness_95

    part.obs["zygosity"] = zygosity
    part.obs["zygosity_5"] = zygosity_5
    part.obs["zygosity_95"] = zygosity_95

    part.obs["clonal_index"] = clonal_index

    mutation_structure = part.uns["optimal_model"]["mutation_structure"]

    part.obs["clonal_structure"] = [
        next(
            structure
            for structure in mutation_structure
            if mutation in structure
        )
        for mutation in part.obs.index
    ]

    return part


# =============================================================================
# Plotting
# =============================================================================

def plot_optimal_model(part):
    """
    Plot marginal s and h posteriors from optimal model.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if part.uns.get("warning") is not None:
        print("WARNING: " + str(part.uns["warning"]))

    model = part.uns["optimal_model"]
    results = model["joint_inference"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for clone_index, result in enumerate(results):
        clone_muts = model["clonal_structure"][clone_index]

        label = "\n".join(
            str(part.obs.iloc[i].name)
            for i in clone_muts
        )

        s_post = result["s_posterior"]
        h_post = result["h_posterior"]

        s_post = s_post / np.maximum(s_post.max(), EPS)
        h_post = h_post / np.maximum(h_post.max(), EPS)

        sns.lineplot(
            x=result["s_range"],
            y=s_post,
            ax=axes[0],
            label=label,
        )

        sns.lineplot(
            x=result["h_range"],
            y=h_post,
            ax=axes[1],
            label=label,
        )

        axes[0].axvline(result["s_map"], ls="--", alpha=0.5)
        axes[1].axvline(result["h_map"], ls="--", alpha=0.5)

    axes[0].set_title("Fitness posterior")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("Normalised posterior")

    axes[1].set_title("Zygosity / LOH posterior")
    axes[1].set_xlabel("h")
    axes[1].set_ylabel("Normalised posterior")

    plt.tight_layout()
    plt.show()
