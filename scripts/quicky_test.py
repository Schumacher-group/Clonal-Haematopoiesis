import sys
sys.path.append("..")

import numpy as np
import jax.numpy as jnp
import jax.scipy.stats as jsp_stats
from src.KI_2 import (
    compute_deterministic_size,
    compute_global_variables,
    BD_process_dynamics,
    log_trapezoid,
)

# --- config (match the benchmark) ---
N_W         = 1e5
LAMB        = 1.3
DEPTH       = 2000
TIME_POINTS = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0])
INITIAL_VAF = 0.05
EPS         = 1e-8
CS          = [[0]]
S_VEC = jnp.linspace(0.01, 1.5, 40)
TPS   = jnp.asarray(TIME_POINTS)


# --- forward model (h=0 truth) ---
def initial_x_from_vaf(v0, h, N_w=N_W):
    pole = (1.0 + h) / 2.0
    return max(N_w * v0 / max(pole - v0, EPS), 10.0)

def bd_step(x, s, dt, rng):
    if x <= 0: return 0.0
    e = np.exp(s*dt); mean = x*e
    var = x*(2*LAMB+s)*e*(e-1)/s
    if var <= mean: return mean
    p = mean/var; n = mean**2/(var-mean)
    return float(rng.negative_binomial(n, p))

def simulate(s, h, seed):
    rng = np.random.default_rng(seed)
    x = initial_x_from_vaf(INITIAL_VAF, h); sizes = [x]
    for i in range(1, len(TIME_POINTS)):
        x = bd_step(x, s, TIME_POINTS[i]-TIME_POINTS[i-1], rng); sizes.append(x)
    sizes = np.asarray(sizes)
    vaf = (1.0+h)*sizes/(2.0*(N_W+sizes))
    vaf = np.clip(vaf, EPS, (1.0+h)/2.0 - EPS)
    AO = rng.binomial(DEPTH, vaf).astype(float)
    DP = np.full(len(TIME_POINTS), DEPTH, dtype=float)
    return AO[:, None], DP[:, None], vaf


# --- hom (pole 1.0 == KI_2 at h=1) max log-likelihood over s-grid ---
def hom_max_loglik(AO, DP):
    AOj, DPj = jnp.asarray(AO), jnp.asarray(DP)
    det, tot = compute_deterministic_size(CS, AOj, DPj, AOj.shape[1], 1.0)  # h=1
    x_vec, exp_s, rec, pyx, _ = compute_global_variables(
        S_VEC, AOj, DPj, tot, det, TPS, 1.0)
    best = -np.inf
    for si in range(S_VEC.shape[0]):
        p_v, n_v = BD_process_dynamics(float(S_VEC[si]), x_vec, exp_s[si])
        la = jnp.log(rec[:, 0])
        xi, pi, ni, py = x_vec[:, 0], p_v[:, 0], n_v[:, 0], pyx[:, 0]
        for j in range(1, TPS.shape[0]):
            lb = jsp_stats.nbinom.logpmf(xi[j][:, None], p=pi[j-1], n=ni[j-1])
            la = jnp.log(py[j]) + log_trapezoid(lb + la[None, :], xi[j-1], axis=1)
        best = max(best, float(log_trapezoid(la, xi[-1])))
    return best


if __name__ == "__main__":
    AO, DP, vaf = simulate(0.5, 0.0, 42)   # h=0 data (hom's worst case)
    print("observed VAF trajectory:", np.round(vaf, 3))
    ll = hom_max_loglik(AO, DP)
    print(f"hom max log-lik at h=0 data: {ll:.2f}")
    print(f"exp(that) in linear space : {np.exp(ll):.3e}")

