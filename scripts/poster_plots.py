import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# ============================================================
# Displacement model
# ============================================================

def displacement_rhs(t, y, s, E, B):
    x, Nw, F = y

    dxdt = s * x * (1 - (x + Nw) / E)

    denom = F + B * Nw

    if denom < 1e-12:
        dNwdt = 0.0
        dFdt = 0.0
    else:
        dNwdt = -dxdt * (B * Nw) / denom
        dFdt = -dxdt * F / denom

    return [dxdt, dNwdt, dFdt]


def simulate_displacement(
    s=0.15,
    B=1.0,
    E=100000,
    x0=100,
    Nw0=80000,
    F0=None,
    tmax=30,
):
    if F0 is None:
        F0 = E - x0 - Nw0

    t_eval = np.linspace(0, tmax, 200)

    sol = solve_ivp(
        displacement_rhs,
        [0, tmax],
        [x0, Nw0, F0],
        args=(s, E, B),
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    x = sol.y[0]
    Nw = sol.y[1]

    vaf = x / (2 * (x + Nw))

    return sol.t, x, Nw, vaf


# ============================================================
# Exponential CHIP model
# ============================================================

def vaf_exponential(t, x0, s, Nw):
    x = x0 * np.exp(s * t)
    return x / (2 * (Nw + x))


# ============================================================
# Generate synthetic data
# ============================================================

TRUE_S = 0.15
TRUE_B = 3.0

t_dense, x, Nw, vaf_dense = simulate_displacement(
    s=TRUE_S,
    B=TRUE_B,
)

# synthetic observation times
t_obs = np.arange(0, 31, 3)

_, _, _, vaf_obs = simulate_displacement(
    s=TRUE_S,
    B=TRUE_B,
)

vaf_obs = np.interp(t_obs, t_dense, vaf_dense)

# add sequencing noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.005, size=len(vaf_obs))
vaf_noisy = np.clip(vaf_obs + noise, 0, 1)

# ============================================================
# Fit exponential model
# ============================================================

popt_exp, _ = curve_fit(
    lambda t, x0, s: vaf_exponential(t, x0, s, 80000),
    t_obs,
    vaf_noisy,
    p0=[100, 0.1],
    bounds=([1, 0], [10000, 1]),
)

x0_hat, s_hat_exp = popt_exp

t_fit = np.linspace(0, 30, 300)

vaf_fit_exp = vaf_exponential(
    t_fit,
    x0_hat,
    s_hat_exp,
    80000,
)

# ============================================================
# Fit displacement model (known B for demo)
# ============================================================

def displacement_vaf_fit(t, x0, s):

    t_sim, _, _, vaf = simulate_displacement(
        s=s,
        B=TRUE_B,
        x0=x0,
    )

    return np.interp(t, t_sim, vaf)

popt_disp, _ = curve_fit(
    displacement_vaf_fit,
    t_obs,
    vaf_noisy,
    p0=[100, 0.1],
    bounds=([1, 0], [10000, 1]),
)

x0_hat_disp, s_hat_disp = popt_disp

vaf_fit_disp = displacement_vaf_fit(
    t_fit,
    x0_hat_disp,
    s_hat_disp,
)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    t_obs,
    vaf_noisy,
    s=80,
    label="Synthetic observations",
)

ax.plot(
    t_dense,
    vaf_dense,
    linewidth=3,
    label=f"True displacement model\ns={TRUE_S:.2f}, B={TRUE_B}",
)

ax.plot(
    t_fit,
    vaf_fit_exp,
    "--",
    linewidth=3,
    label=f"Exponential fit\ns={s_hat_exp:.3f}",
)

ax.plot(
    t_fit,
    vaf_fit_disp,
    linewidth=3,
    label=f"Displacement fit\ns={s_hat_disp:.3f}",
)

ax.set_xlabel("Years")
ax.set_ylabel("VAF")
ax.set_title("Model Mismatch Biases Fitness Inference")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()