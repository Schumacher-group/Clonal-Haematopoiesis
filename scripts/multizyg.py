import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ── PARAMETERS ───────────────────────────────────────────────────────────────
S_TRUE     = 0.5
T_MAX      = 35
N_OBS      = 3
VAF_THRESH = 0.02
K          = 1e5

C_HET  = "#2171b5"
C_HOM  = "#e6550d"
C_MIX  = "#009696"
C_FIT  = "#333333"

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── MODELS ───────────────────────────────────────────────────────────────────
def vaf_het(t, s):
    x = np.exp(s * t)
    return x / (2 * (K + x))

def vaf_hom(t, s):
    x = np.exp(s * t)
    return x / (K + x)

def vaf_mixed(t, s):
    x = np.exp(s * t)
    return 1.5 * x / (2 * (K + x))

def fit_het_model(t_obs, v_obs):
    def loss(s):
        return np.sum((v_obs - vaf_het(t_obs, s))**2)
    return minimize_scalar(loss, bounds=(0.01, 2.0), method='bounded').x

# ── OBSERVATIONS ─────────────────────────────────────────────────────────────
def generate_het_observations(s_true, seed=1):
    rng   = np.random.default_rng(seed)
    t_all = np.linspace(0, T_MAX, 2000)
    v_all = vaf_het(t_all, s_true)
    idx0  = np.where(v_all >= VAF_THRESH)[0][0]
    t0    = t_all[idx0]
    t_obs = np.linspace(t0, t0 + 4, N_OBS)
    v_obs = np.interp(t_obs, t_all, v_all)
    noise = rng.normal(0, 0.05 * v_obs)
    return t_obs, np.clip(v_obs + noise, 1e-4, 0.5)

def generate_observations(vaf_func, s_true, t_obs_het, seed=0):
    rng   = np.random.default_rng(seed)
    t_all = np.linspace(0, T_MAX, 2000)
    v_all = vaf_func(t_all, s_true)
    v_obs = np.interp(t_obs_het, t_all, v_all)
    noise = rng.normal(0, 0.05 * v_obs)
    return np.clip(v_obs + noise, 1e-4, 0.99)

t_het, v_het = generate_het_observations(S_TRUE, seed=1)
v_hom = generate_observations(vaf_hom,   S_TRUE, t_het, seed=2)
v_mix = generate_observations(vaf_mixed, S_TRUE, t_het, seed=3)

s_hat_hom = fit_het_model(t_het, v_hom)
s_hat_mix = fit_het_model(t_het, v_mix)

t_full = np.linspace(0, T_MAX, 600)

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.4],
                      left=0.08, right=0.97,
                      top=0.94, bottom=0.07,
                      wspace=0.25, hspace=0.18)

# ── TOP: all three true trajectories ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])

ax0.plot(t_full, vaf_het(t_full, S_TRUE),   color=C_HET, lw=2.5, label="Heterozygous")
ax0.plot(t_full, vaf_mixed(t_full, S_TRUE), color=C_MIX, lw=2.5, label="Mixed (50/50)")
ax0.plot(t_full, vaf_hom(t_full, S_TRUE),   color=C_HOM, lw=2.5, label="Homozygous")

ax0.axhline(0.5, color='#aaaaaa', lw=0.8, ls='--')
ax0.text(T_MAX - 0.3, 0.515, 'VAF = 0.5', fontsize=9, color='#888888', ha='right')

ax0.axvspan(t_het[0] - 0.3, t_het[-1] + 0.3, alpha=0.08, color='grey')
ax0.axvline(t_het[0], color='grey', lw=1, ls='--', alpha=0.4)

ax0.scatter(t_het, v_het, color=C_HET, s=60, zorder=5, edgecolors='white', lw=1)
ax0.scatter(t_het, v_mix, color=C_MIX, s=60, zorder=5, edgecolors='white', lw=1)
ax0.scatter(t_het, v_hom, color=C_HOM, s=60, zorder=5, edgecolors='white', lw=1)

# equations
ax0.text(0.02, 0.97,
    r"$V_{\mathrm{het}}(t)=\dfrac{x(t)}{2(N_w+x(t))}$",
    transform=ax0.transAxes, fontsize=20, color=C_HET, va='top')
ax0.text(0.02, 0.62,
    r"$V_{\mathrm{mixed}}(t)=\dfrac{x_{\mathrm{het}}(t)+2x_{\mathrm{hom}}(t)}{2(N_w+x_{\mathrm{het}}(t)+x_{\mathrm{hom}}(t))}$",
    transform=ax0.transAxes, fontsize=20, color=C_MIX, va='top')
ax0.text(0.02, 0.27,
    r"$V_{\mathrm{hom}}(t)=\dfrac{x(t)}{N_w+x(t)}$",
    transform=ax0.transAxes, fontsize=20, color=C_HOM, va='top')

ax0.set_xlim(0, T_MAX)
ax0.set_ylim(0, 1.05)
ax0.set_xlabel('Time (years)', fontsize=12)
ax0.set_ylabel('VAF', fontsize=12)
ax0.set_title('True VAF trajectories for different zygosities', fontsize=13,
              fontweight='bold', loc='left')
ax0.legend(fontsize=10, loc='center right', frameon=True, framealpha=0.9)

# ── BOTTOM: het model misapplied ──────────────────────────────────────────────
panels = [
    (gs[1, 0], vaf_hom,   v_hom, s_hat_hom, "Homozygous data",    C_HOM),
    (gs[1, 1], vaf_mixed, v_mix, s_hat_mix, "Mixed (50/50) data", C_MIX),
]

for spec, vaf_func, v_obs, s_hat, title, color in panels:
    ax = fig.add_subplot(spec)

    ax.axvspan(t_het[0] - 0.3, t_het[-1] + 0.3, alpha=0.08, color='grey')

    # true trajectory
    ax.plot(t_full, vaf_func(t_full, S_TRUE),
            color=color, lw=2.5, label=f'True trajectory ($s={S_TRUE}$)')

    # het reference (faint)
    ax.plot(t_full, vaf_het(t_full, S_TRUE),
            color=C_HET, lw=1.5, ls=':', alpha=0.45,
            label=f'Het reference ($s={S_TRUE}$)')

    # observations
    ax.scatter(t_het, v_obs, color=color, s=100, zorder=5,
               edgecolors='white', lw=1.5, label='Observed VAF')

    # wrong fit projected
    ax.plot(t_full, vaf_het(t_full, s_hat),
            color=C_FIT, lw=2.2, ls='--',
            label=rf'Het model fit ($\hat{{s}}={s_hat:.2f}$)')

    ax.axhline(0.5, color='#aaaaaa', lw=0.8, ls='--')
    ax.text(T_MAX - 0.3, 0.515, 'VAF = 0.5', fontsize=9,
            color='#888888', ha='right')

    bias  = s_hat - S_TRUE
    btext = (f"$s_{{\\mathrm{{true}}}}={S_TRUE}$\n"
             f"$\\hat{{s}}={s_hat:.2f}$\n"
             f"bias $={bias:+.2f}$")
    ax.text(0.97, 0.35, btext,
            transform=ax.transAxes, fontsize=13,
            va='top', ha='right', color='#C62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFEBEE',
                      edgecolor='#C62828', lw=1.4))

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('VAF', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=8)
    ax.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9)

fig.suptitle(
    "Fitness Inference Assuming Heterozygosity — Applied to Different Zygosity Data",
    fontsize=14, fontweight='bold'
)

plt.savefig('zygosity_vaf_plot.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved.")