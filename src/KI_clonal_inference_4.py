import sys
sys.path.append("..")   # fix to import modules from root
from scripts.debug_patient2 import AO, DP
from src.general_imports import *

import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.stats as jsp_stats
import jax.random as jrnd
from itertools import combinations
from scipy.stats import binom  # For h inference

key = jrnd.PRNGKey(758493)  # Random seed is explicit in JAX

#region FIXED: Non vectorised functions with DEBUG
# Non vectorised 
def compute_deterministic_size_mixed(cs, AO, DP, n_mutations):
    N_w = 1e5

    lm = []
    clonal_map = jnp.zeros(n_mutations, dtype=int)
    for i, cs_idx in enumerate(cs):
        max_idx = jnp.argmax((AO/DP)[:, cs_idx].sum(axis=0))
        lm.append(cs_idx[max_idx])
        clonal_map = clonal_map.at[jnp.array(cs_idx)].set(jnp.repeat(i, len(cs_idx)))

    deterministic_clone_size = jnp.array(-N_w * jnp.sum((AO/DP)[:, lm], axis=1)
                                        / (jnp.sum((AO/DP)[:, lm], axis=1) - 0.5))
    deterministic_clone_size = jnp.ceil(deterministic_clone_size)
    total_cells = N_w + deterministic_clone_size

    deterministic_size = AO/DP * 2 * total_cells[:, None]

    return deterministic_size, total_cells, clonal_map
#endregion

#region FIXED: Vectorised and jit optimised functions

def jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                            deterministic_size, total_cells, 
                            key, resolution=600):
    """
    FIXED VERSION: Improved numerical stability throughout.
    """
    
    # Compute global variables (samples) using the provided key
    x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut = \
        compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size, 
                                    time_points, key, resolution=resolution)

    # s index vector
    s_idx = jnp.arange(s_vec.shape[0])

    # Fitness_specific_computations mapped over s index
    def fitness_specific_computations_mapped(s_idx):
        s = s_vec[s_idx]
        exp_term_vec = exp_term_vec_s[s_idx]
        
        # BD dynamics on total
        p_vec, n_vec = BD_process_dynamics_mixed(s, x_het_vec + x_hom_vec, exp_term_vec)

        # Compute mutation-likelihoods across all mutations (vectorised)
        mutation_likelihood = jax.vmap(
            lambda ii: mutation_specific_ll_mixed_grid(
                ii, recursive_term_vec[:, :],
                x_het_vec, x_hom_vec,
                p_vec, n_vec, p_y_cond_x_vec,
                time_points.shape[0]
            )
        )(jnp.arange(n_mut))
        
        # Compute clonal likelihoods as product over mutations in each clone
        clonal_likelihood = jnp.zeros(len(cs))
        for idx_c, c_idx in enumerate(cs):
            clone_mutations = jnp.array(c_idx)
            # Use log-space for numerical stability
            log_mut_liks = jnp.log(jnp.maximum(mutation_likelihood[clone_mutations], 1e-300))
            clone_likelihood = jnp.exp(jnp.sum(log_mut_liks))
            clonal_likelihood = clonal_likelihood.at[idx_c].set(clone_likelihood)
            
        return clonal_likelihood

    # Vmap over s
    clonal_likelihood = jax.vmap(fitness_specific_computations_mapped)(s_idx)
    return clonal_likelihood  # shape (n_s, n_clones)


def compute_global_variables_mixed(s_vec, AO, DP, total_cells, deterministic_size,
                                    time_points, key, resolution=600):
    
    n_tps, n_mut = AO.shape
    
    # BD exponential term for pmf
    delta_t = jnp.diff(time_points)
    exp_term_vec_s = jnp.exp(delta_t * s_vec[:, None])
    exp_term_vec_s = jnp.reshape(exp_term_vec_s, (*exp_term_vec_s.shape, 1, 1))

    key_het, key_hom = jrnd.split(key, 2)

    # Sample VAFs from Beta posterior (script 1 approach)
    beta_p_rvs = jrnd.beta(key=key_het, a=AO[:, :, None] + 1,
                            b=DP[:, :, None] - AO[:, :, None] + 1,
                            shape=(n_tps, n_mut, resolution))
    beta_p_rvs = jnp.sort(beta_p_rvs)

    # Back-transform to total clone size
    N_w_cond_vec = (total_cells[:, None] - deterministic_size)[:, :, None]
    N_w_cond_vec = jnp.maximum(N_w_cond_vec, 1.0)
    x_total_vec = jnp.ceil(-N_w_cond_vec * beta_p_rvs / (beta_p_rvs - 0.5))

    # Sample het/hom split
    frac_hom = jrnd.uniform(key_hom, shape=(n_tps, n_mut, resolution))
    x_hom_vec = x_total_vec * frac_hom
    x_het_vec = x_total_vec - x_hom_vec

    # VAF with het/hom correction
    true_vaf_vec = (x_het_vec + 2.0 * x_hom_vec) / (2.0 * (N_w_cond_vec + x_total_vec))
    true_vaf_vec = jnp.clip(true_vaf_vec, 1e-6, 1.0 - 1e-6)

    # Observation probabilities
    log_p_y_cond_x_vec = jsp_stats.binom.logpmf(AO[:, :, None], n=DP[:, :, None], p=true_vaf_vec)
    p_y_cond_x_vec = jnp.exp(jnp.maximum(log_p_y_cond_x_vec, -300.0))

    # Initial recursive term
    recursive_term_vec = p_y_cond_x_vec[0, :, :] * (1.0 / resolution)

    return x_het_vec, x_hom_vec, exp_term_vec_s, recursive_term_vec, p_y_cond_x_vec, n_mut


def BD_process_dynamics_mixed(s, x_total_vec, exp_term_vec):
    """
    FIXED VERSION: Robust negative binomial parameter calculation.
    """
    
    lamb = 1.3
    
    # x_total_vec shape: (n_tps, n_mut, resolution)
    mean_vec = x_total_vec[:-1, :, :] * exp_term_vec  # broadcasting
    
    # FIXED: Robust variance calculation
    s_safe = jnp.maximum(jnp.abs(s), 1e-8)
    variance_vec = x_total_vec[:-1, :, :] * (2.0 * lamb + s) * exp_term_vec * (exp_term_vec - 1.0) / s_safe
    
    # CRITICAL FIX: Ensure variance > mean always (required for negative binomial)
    min_variance_ratio = 1.2  # Variance must be at least 20% larger than mean
    min_variance = mean_vec * min_variance_ratio + 1e-6
    variance_vec = jnp.maximum(variance_vec, min_variance)
    
    # FIXED: Ensure valid parameters with strict bounds
    p_vec = mean_vec / variance_vec
    p_vec = jnp.clip(p_vec, 1e-8, 1.0 - 1e-8)  # Must be in (0, 1)
    
    n_vec = jnp.power(mean_vec, 2) / jnp.maximum(variance_vec - mean_vec, 1e-8)
    n_vec = jnp.maximum(n_vec, 1e-8)  # Must be positive
    
    return p_vec, n_vec


def mutation_specific_ll_mixed_grid(i, recursive_term_vec, x_het_vec, x_hom_vec, 
                                    p_vec, n_vec, p_y_cond_x_vec, n_tps):
    """
    FIXED VERSION: Improved numerical stability in recursive integration.
    """
    
    recursive_term_i = recursive_term_vec[i]  # shape (resolution,)
    x_het_i = x_het_vec[:, i, :]  # (n_tps, resolution)
    x_hom_i = x_hom_vec[:, i, :]  # (n_tps, resolution)
    p_i = p_vec[:, i, :]  # (n_intervals, resolution)
    n_i = n_vec[:, i, :]  # (n_intervals, resolution)
    p_y_cond_x_i = p_y_cond_x_vec[:, i, :]  # (n_tps, resolution)

    # Iterate through timepoints
    for j in range(1, n_tps):
        # Total sizes at previous and current tp
        init_total = x_het_i[j-1] + x_hom_i[j-1]  # (resolution,)
        next_total = x_het_i[j] + x_hom_i[j]  # (resolution,)

        # FIXED: Log-space computation for BD PMF to avoid underflow
        log_bd_pmf = jsp_stats.nbinom.logpmf(next_total[:, None], p=p_i[j-1][None, :], n=n_i[j-1][None, :])
        log_bd_pmf = jnp.maximum(log_bd_pmf, -300.0)  # Floor
        bd_pmf = jnp.exp(log_bd_pmf)
        
        # Inner sum: integrate over previous-res grid weighted by recursive_term_i
        inner_sum = bd_pmf * recursive_term_i  # (next_res, init_res)
        
        # Integrate over init axis using trapezoid
        inner_integrated = jsp.integrate.trapezoid(x=init_total, y=inner_sum, axis=1)  # (next_res,)
        inner_integrated = jnp.maximum(inner_integrated, 1e-300)  # Floor to prevent log(0)
        
        # FIXED: Log-space multiplication for numerical stability
        log_p_y = jnp.log(jnp.maximum(p_y_cond_x_i[j], 1e-300))
        log_inner = jnp.log(inner_integrated)
        log_recursive_term_i = log_p_y + log_inner
        
        # Convert back to linear space
        recursive_term_i = jnp.exp(jnp.maximum(log_recursive_term_i, -300.0))

    # Final integration over the last grid
    total_x_final = x_het_i[-1] + x_hom_i[-1]  # (resolution,)
    final_like = jsp.integrate.trapezoid(x=total_x_final, y=recursive_term_i)
    final_like = jnp.maximum(final_like, 1e-300)  # Ensure non-negative
    
    return final_like


def compute_clonal_models_prob_vec_mixed(part, s_resolution=50, min_s=0.01, max_s=3,
                                        filter_invalid=True, disable_progressbar=False,
                                        resolution=600, master_key_seed=758493):
    
    print("="*60)
    print("COMPUTING CLONAL MODELS PROBABILITIES (VECTORIZED - FIXED)")
    print("="*60)
    
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

    master_key = jrnd.PRNGKey(master_key_seed)
    keys = jrnd.split(master_key, len(cs_list))

    # --- Step 1: compute raw probability for every model, store as 2-tuple ---
    for i, cs in enumerate(cs_list):
        print(f"\nProcessing model {i}/{len(cs_list)}: {cs}")
        
        deterministic_size, total_cells, clonal_map = \
            compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

        key_i = keys[i]
        output = jax_cs_hmm_ll_vec_mixed(s_vec, AO, DP, time_points, cs,
                                        deterministic_size, total_cells, 
                                        key_i, resolution=resolution)

        model_prob = compute_model_likelihood(output, cs, s_vec)
        part.uns['model_dict'][f'model_{i}'] = (cs, model_prob)
        print(f"Model {i} probability: {model_prob:.3e}")

    # --- Step 2: normalise to get posterior probabilities ---
    total_prob = sum(v[1] for v in part.uns['model_dict'].values())

    if total_prob == 0 or not np.isfinite(total_prob):
        print("\n⚠️  WARNING: All model probabilities are zero. Cannot normalise.")
        normalised = {k: 0.0 for k in part.uns['model_dict']}
    else:
        normalised = {k: v[1] / total_prob for k, v in part.uns['model_dict'].items()}

    # --- Step 3: upgrade to 3-tuple (cs, raw_prob, normalised_prob) and sort ---
    part.uns['model_dict'] = {
        k: (v[0], v[1], normalised[k])
        for k, v in part.uns['model_dict'].items()
    }

    part.uns['model_dict'] = {
        k: v for k, v in sorted(
            part.uns['model_dict'].items(),
            key=lambda item: item[1][2],  # sort by normalised prob
            reverse=True
        )
    }

    # --- Step 4: print ranking ---
    print("\n" + "="*60)
    print("MODEL RANKING (sorted by posterior probability):")
    print("="*60)
    for i, (k, v) in enumerate(part.uns['model_dict'].items()):
        cs, raw, norm = v
        print(f"Rank {i}: {k} - Raw: {raw:.3e} | Posterior: {norm*100:.2f}% | Structure: {cs}")

    print(f"\nTop 5 models:")
    for i, (k, v) in enumerate(list(part.uns['model_dict'].items())[:5]):
        cs, raw, norm = v
        print(f"  {i+1}. {k}: posterior={norm*100:.2f}%, raw={raw:.3e}, structure={cs}")

    return part


def infer_sh_jointly_from_dynamics(cs, AO, DP, time_points, 
                                   deterministic_size, total_cells, 
                                   s_resolution=30, h_resolution=20, lamb=1.3, N_w=1e5):
    """
    Joint inference of (s, h) using temporal VAF dynamics.
    
    Key insight: The VAF trajectory shape reveals both fitness and zygosity:
    - Growth rate → fitness (s)
    - Saturation level → zygosity (h)
    
    Parameters:
    -----------
    cs : list of lists
        Clonal structure
    AO, DP : arrays (n_timepoints, n_mutations)
        Observed data
    time_points : array
        Observation timepoints
    s_resolution, h_resolution : int
        Grid resolution
    lamb : float
        Birth rate
    N_w : float
        Wild-type population size
        
    Returns:
    --------
    results : list of dicts
        MAP estimates and posteriors for each clone
    """
    
    print("\n" + "="*70)
    print("JOINT (s, h) INFERENCE FROM TEMPORAL DYNAMICS")
    print("="*70)
    
    s_range = np.linspace(0.01, 1.0, s_resolution)
    h_range = np.linspace(0, 1, h_resolution)
    
    results = []
    
    for clone_idx, clone_muts in enumerate(cs):
        print(f"\nClone {clone_idx}: {clone_muts}")
        print("-" * 70)
        
        # Select leading mutation (highest summed VAF across timepoints) —
        # consistent with compute_deterministic_size_mixed
        vaf_ratio = AO / np.maximum(DP, 1.0)
        clone_vafs = vaf_ratio[:, clone_muts]
        vaf_sums = clone_vafs.sum(axis=0)
        mut_idx = clone_muts[int(np.argmax(vaf_sums))]

        AO_clone = np.array(AO[:, mut_idx])
        DP_clone = np.array(DP[:, mut_idx])

        # Mask out zero-depth timepoints (ND / not in panel)
        valid_mask = DP_clone > 0
        AO_clone = AO_clone[valid_mask]
        DP_clone = DP_clone[valid_mask]
        time_points_valid = np.array(time_points)[valid_mask]
        observed_vaf = AO_clone / np.maximum(DP_clone, 1.0)
        
        print(f"  Leading mutation index: {mut_idx}")
        print(f"  Valid timepoints: {valid_mask.sum()} / {len(valid_mask)}")
        print(f"  VAF trajectory: {' → '.join(f'{v:.3f}' for v in observed_vaf)}")
        
        # 2D likelihood grid over (s, h)
        joint_log_likelihood = np.zeros((s_resolution, h_resolution))
        
        print(f"  Computing ({s_resolution} × {h_resolution}) likelihood grid...")
        
        for s_idx, s in enumerate(s_range):
            for h_idx, h in enumerate(h_range):
                
                # Simulate clone expansion under (s, h)
                log_lik = 0
                
                # Initial size estimate
                N_mut_init = observed_vaf[0] * 2 * N_w / (1 + h)
                N_mut_init = max(N_mut_init, 100)
                
                N_mut = N_mut_init
                
                # ── FIXED: iterate over valid timepoints only ──────────────
                for tp_idx in range(len(time_points_valid)):
                    if tp_idx > 0:
                        dt = time_points_valid[tp_idx] - time_points_valid[tp_idx - 1]
                        N_mut = N_mut * np.exp(s * dt)
                    
                    # Cap at wildtype population (can't exceed total)
                    N_mut = min(N_mut, N_w * 0.95)
                    
                    # Split into het/hom based on h
                    N_hom = N_mut * h
                    N_het = N_mut * (1 - h)
                    
                    # Expected VAF
                    vaf_expected = (N_het + 2 * N_hom) / (2 * N_w)
                    vaf_expected = np.clip(vaf_expected, 1e-8, 1.0 - 1e-8)
                    
                    # Binomial log likelihood
                    ao = int(AO_clone[tp_idx])
                    dp = int(DP_clone[tp_idx])
                    log_lik += binom.logpmf(ao, dp, vaf_expected)
                # ────────────────────────────────────────────────────────────
                
                joint_log_likelihood[s_idx, h_idx] = log_lik
        
        # Convert to probability
        max_log_lik = joint_log_likelihood.max()
        joint_log_likelihood = joint_log_likelihood - max_log_lik
        joint_likelihood = np.exp(np.clip(joint_log_likelihood, -700, 0))
        
        # Uniform prior
        prior_s = np.ones_like(s_range) / s_range.shape[0]
        prior_h = np.ones_like(h_range) / h_range.shape[0]
        prior_joint = prior_s[:, None] * prior_h[None, :]
        
        # Posterior
        joint_posterior = joint_likelihood * prior_joint
        Z = joint_posterior.sum()
        
        if Z == 0 or not np.isfinite(Z):
            print(f"  ⚠️  WARNING: Posterior normalization failed")
            joint_posterior = prior_joint
        else:
            joint_posterior = joint_posterior / Z
        
        # Marginalize
        s_posterior = joint_posterior.sum(axis=1)
        h_posterior = joint_posterior.sum(axis=0)
        
        s_posterior = s_posterior / (s_posterior.sum() + 1e-300)
        h_posterior = h_posterior / (h_posterior.sum() + 1e-300)
        
        # MAP estimates
        s_map_idx, h_map_idx = np.unravel_index(
            joint_posterior.argmax(), joint_posterior.shape
        )
        s_map = s_range[s_map_idx]
        h_map = h_range[h_map_idx]
        
        # Marginal MAP
        s_map_marginal = s_range[np.argmax(s_posterior)]
        h_map_marginal = h_range[np.argmax(h_posterior)]
        
        # Credible intervals
        s_cumsum = np.cumsum(s_posterior)
        h_cumsum = np.cumsum(h_posterior)
        
        s_ci_low = s_range[np.searchsorted(s_cumsum, 0.05)]
        s_ci_high = s_range[np.searchsorted(s_cumsum, 0.95)]
        h_ci_low = h_range[np.searchsorted(h_cumsum, 0.05)]
        h_ci_high = h_range[np.searchsorted(h_cumsum, 0.95)]
        
        print(f"\n  Results:")
        print(f"    s = {s_map_marginal:.3f}  [90% CI: {s_ci_low:.3f} - {s_ci_high:.3f}]")
        print(f"    h = {h_map_marginal:.3f}  [90% CI: {h_ci_low:.3f} - {h_ci_high:.3f}]")
        
        results.append({
            's_map': s_map_marginal,
            'h_map': h_map_marginal,
            's_posterior': s_posterior,
            'h_posterior': h_posterior,
            'joint_posterior': joint_posterior,
            's_range': s_range,
            'h_range': h_range,
            's_ci': (s_ci_low, s_ci_high),
            'h_ci': (h_ci_low, h_ci_high),
        })
    
    print("\n" + "="*70)
    return results


def refine_optimal_model_posterior_vec(part, s_resolution=30, h_resolution=20):
    """
    IMPROVED VERSION: Joint (s, h) inference from temporal dynamics.
    """
    
    # Retrieve optimal clonal structure
    cs = list(part.uns['model_dict'].values())[0][0]

    # Extract participant features
    AO = jnp.array(part.layers['AO'].T)
    DP = jnp.array(part.layers['DP'].T)
    time_points = jnp.array(part.var.time_points)

    # Compute deterministic clone sizes (for bounds only)
    deterministic_size, total_cells, clonal_map = \
        compute_deterministic_size_mixed(cs, AO, DP, AO.shape[1])

    # Joint (s, h) inference from temporal dynamics
    joint_results = infer_sh_jointly_from_dynamics(
        cs, AO, DP, time_points, 
        deterministic_size, total_cells,
        s_resolution=s_resolution, 
        h_resolution=h_resolution
    )

    # Extract results
    h_vec = np.array([result['h_map'] for result in joint_results])
    h_posterior_list = [result['h_posterior'] for result in joint_results]
    s_vec = joint_results[0]['s_range']
    
    # Construct fitness posterior from joint results (for compatibility)
    fitness_posterior = np.column_stack([result['s_posterior'] for result in joint_results])

    part.uns['optimal_model'] = {
        'clonal_structure': cs,
        'mutation_structure': [list(part.obs.iloc[cs_idx].index) for cs_idx in cs],
        'joint_inference': joint_results,
        'posterior': fitness_posterior,  # For compatibility
        's_range': s_vec,
        'h_vec': h_vec,
        'h_posterior': h_posterior_list
    }

    # Append optimal model information to dataset observations
    fitness = np.zeros(part.shape[0])
    fitness_5 = np.zeros(part.shape[0])
    fitness_95 = np.zeros(part.shape[0])
    clonal_index = np.zeros(part.shape[0])

    for i, c_idx in enumerate(cs):
        result = joint_results[i]
        
        # Fitness from joint inference
        fitness[c_idx] = result['s_map']
        fitness_5[c_idx] = result['s_ci'][0]
        fitness_95[c_idx] = result['s_ci'][1]
        clonal_index[c_idx] = i

    part.obs['fitness'] = fitness
    part.obs['fitness_5'] = fitness_5
    part.obs['fitness_95'] = fitness_95
    part.obs['clonal_index'] = clonal_index

    # Append mutational structure to each mutation
    mut_structure = part.uns['optimal_model']['mutation_structure']
    clonal_structure_list = []
    for mut in part.obs.index:
        for structure in mut_structure:
            if mut in structure:
                clonal_structure_list.append(structure)
                break

    part.obs['clonal_structure'] = clonal_structure_list

    return part


#endregion

#region Utility functions (unchanged but included for completeness)

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


def compute_model_likelihood(output, cs, s_range):
    """Compute model likelihood from clonal posteriors"""
    # Initialize clonal probability
    clonal_prob = np.zeros(len(cs))

    s_range_size = s_range.max() - s_range.min()
    s_prior = 1/s_range_size
    
    # Marginalise fitness for every clone to get clonal probability
    for i, out in enumerate(output.T):
        # Convert to numpy and guard against NaNs/Infs
        out = np.array(out, copy=False)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.maximum(out, 0.0)  # Ensure non-negative
        
        integral = np.trapz(x=s_range, y=out)
        clonal_prob[i] = s_prior * max(integral, 0.0)

    # Model probability as product of clonal probabilities (independence assumption)
    model_probability = np.prod(clonal_prob)

    return model_probability


def compute_invalid_combinations(part, pearson_distance_threshold=0.7):
    """Find mutation pairs with very different temporal correlation"""
    # Compute pearson of each mutation with time
    correlation_matrix = np.corrcoef(
        np.vstack([part.X, part.var.time_points]))
    correlation_vec = correlation_matrix[-1, :-1]

    # Compute distance between pearsonr's
    distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])

    # Label invalid combinations if pearson correlation is too different
    not_valid_comb = np.argwhere(distance_matrix > pearson_distance_threshold)
    
    # Extract unique tuples from list (Order Irrespective)
    res = []
    for i in not_valid_comb:
        if [i[0], i[1]] and [i[1], i[0]] not in res:
            res.append(i.tolist()) 
    
    part.uns['invalid_combinations'] = res


def find_valid_clonal_structures(part, p_distance_threshold=1, filter_invalid=True):
    """
    Find all valid clonal structures using pearson correlation analysis
    and a VAF sum feasibility check (sum of leading mutation VAFs must be ≤ 1
    at every timepoint, otherwise the structure is biologically impossible).
    """
    
    n_mutations = part.shape[0]

    # AO/DP layers are (n_mutations, n_timepoints); transpose to (n_tps, n_muts)
    AO = part.layers['AO'].T
    DP = part.layers['DP'].T
    vaf_ratio = AO / np.maximum(DP, 1.0)  # (n_tps, n_muts)

    if n_mutations == 1:
        valid_cs = [[[0]]]
        return valid_cs

    else:
        if filter_invalid is True:
            compute_invalid_combinations(part, pearson_distance_threshold=p_distance_threshold)
            
        a = list(partition(list(range(n_mutations))))
        cs_list = [cs for cs in a]

        if filter_invalid is False:
            return cs_list
        
        else:
            valid_cs = []

            for cs in cs_list:
                # ── existing correlation filter ────────────────────────────
                invalid_combinations_in_cs = 0
                for clone in cs:
                    mut_comb = list(combinations(clone, 2))
                    n_invalid_comb_in_clone = len(
                        [comb for comb in mut_comb 
                            if list(comb) in part.uns['invalid_combinations']])
                    invalid_combinations_in_cs += n_invalid_comb_in_clone

                if invalid_combinations_in_cs > 0:
                    continue

                # ── NEW: VAF sum feasibility filter ───────────────────────
                # For each clone, find the leading mutation (highest summed VAF)
                # then check that the sum of leading VAFs across clones ≤ 1
                # at every timepoint. If it ever exceeds 1 the structure is
                # biologically impossible in a diploid model.
                leading_vafs = []
                for clone in cs:
                    clone_vafs = vaf_ratio[:, clone]          # (n_tps, n_muts_in_clone)
                    vaf_sums   = clone_vafs.sum(axis=0)       # (n_muts_in_clone,)
                    lead_idx   = int(np.argmax(vaf_sums))
                    leading_vafs.append(clone_vafs[:, lead_idx])  # (n_tps,)

                leading_vafs = np.stack(leading_vafs, axis=1)  # (n_tps, n_clones)
                if np.any(leading_vafs.sum(axis=1) > 1.0):
                    continue
                # ──────────────────────────────────────────────────────────

                valid_cs.append(cs)
                    
            return valid_cs


def plot_optimal_model(part):
    """Plot the posterior distributions for the optimal model"""
    if part.uns.get('warning') is not None:
        print('WARNING: ' + part.uns['warning'])
        
    model = part.uns['optimal_model']
    output = model['posterior']
    cs = model['clonal_structure']
    ms = model['mutation_structure']
    s_range = model['s_range']

    # Normalisation constant
    norm_max = np.max(output, axis=0)
    norm_max = np.where(norm_max == 0, 1.0, norm_max)  # Prevent division by zero

    # Plot
    for i in range(len(cs)):
        p_key_str = ''
        for k, j in enumerate(cs[i]):
            if k == 0:
                p_key_str += f'{part[j].obs.p_key.values[0]}'
            if k > 0:
                p_key_str += f'\n{part[j].obs.p_key.values[0]}'

        sns.lineplot(x=s_range,
                    y=output[:, i] / norm_max[i],
                    label=p_key_str)
        

#endregion