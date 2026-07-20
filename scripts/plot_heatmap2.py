import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

def plot_assumption2_talk(results, E_real=6e5, B_real=2.512,
                          save_path="exports/assumption2_talk.png"):
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16,
                         "axes.labelsize": 14, "axes.spines.top": False,
                         "axes.spines.right": False})
    E_arr = np.array(sorted(set(r["E"] for r in results)))
    B_arr = np.array(sorted(set(r["B"] for r in results)))
    B_real = min(B_arr, key=lambda b: abs(b - B_real))

    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.28,
                  left=0.06, right=0.97, top=0.84, bottom=0.14)

    # ---- LEFT: bounded truth vs exponential fit ----
    axL = fig.add_subplot(gs[0])
    axL.yaxis.grid(True, alpha=0.3); axL.set_axisbelow(True)
    slice_r = sorted([r for r in results
                      if np.isclose(r["E"], E_real) and np.isclose(r["B"], B_real)],
                     key=lambda r: r["s_true"])
    s_vals = [r["s_true"] for r in slice_r]
    norm = plt.Normalize(min(s_vals), max(s_vals)); cmap = plt.cm.viridis
    for r in slice_r[::2]:                       # thin out for legibility
        col = cmap(norm(r["s_true"]))
        t, x, Nw, _ = run_true_model_cached(r["s_true"], E_real, B_real)
        d = 2.*(Nw + x)
        axL.plot(t, np.where(d > 0, x/d, 0), color=col, lw=2.5, zorder=3)
        data = generate_synth(r["s_true"], E_real, B_real, seed=42+int(r["s_true"]*1000))
        if data:
            axL.scatter(data["t"], data["VAF"], color=col, s=30,
                        edgecolor="white", lw=0.6, zorder=5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axL, fraction=0.04, pad=0.02); cb.set_label(r"$s_{\rm true}$")
    axL.plot([], [], color="grey", lw=2.5, label="Bounded (true) VAF")
    axL.scatter([], [], color="grey", s=30, edgecolor="white", label="Observed samples")
    axL.legend(loc="upper left", frameon=False, fontsize=11)
    axL.set_xlim(0, T_MAX); axL.set_ylim(0, 1.0)
    axL.set_xlabel("Time (yr)"); axL.set_ylabel("VAF")
    axL.set_title("Growth is bounded — VAF saturates as the\nmutant displaces wild-type",
                  fontweight="bold")

    # ---- RIGHT: recovery heatmap ----
    axR = fig.add_subplot(gs[1])
    grid = np.full((len(E_arr), len(B_arr)), np.nan)
    for i, E in enumerate(E_arr):
        for j, B in enumerate(B_arr):
            v = [abs(r["rel_bias"]) < 10 for r in results
                 if np.isclose(r["E"], E) and np.isclose(r["B"], B)]
            if v: grid[i, j] = 100*np.mean(v)
    im = axR.imshow(grid, origin="lower", aspect="auto", cmap="plasma",
                    vmin=0, vmax=100, interpolation="bicubic")
    cb2 = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.03)
    cb2.set_label("Fitness recovered within 10% (%)")

    # box the biologically realistic region
    ei = [i for i, E in enumerate(E_arr) if E_REALISTIC[0] <= E <= E_REALISTIC[1]]
    bj = [j for j, B in enumerate(B_arr) if B_REALISTIC[0] <= B <= B_REALISTIC[1]]
    if ei and bj:
        axR.add_patch(Rectangle((min(bj)-0.5, min(ei)-0.5),
                                len(bj), len(ei), fill=False,
                                edgecolor="white", lw=3.5, zorder=6))
        axR.annotate("real patients\nlive here", xytext=(0.5, 1.02),
                     xy=((min(bj)+max(bj))/2, max(ei)+0.5),
                     textcoords="axes fraction", ha="center",
                     fontsize=12, fontweight="bold", color="black",
                     arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    axR.set_xticks(range(len(B_arr)))
    axR.set_xticklabels([f"{b:.2g}" for b in B_arr], fontsize=9)
    axR.set_yticks(range(len(E_arr)))
    axR.set_yticklabels([_format_E(E) for E in E_arr], fontsize=9)
    axR.set_xlabel(r"Wild-type displacement  $B$")
    axR.set_ylabel(r"Niche size  $E$")
    axR.set_title("Simple model recovers fitness in the\nrealistic regime — fails only at extremes",
                  fontweight="bold")

    fig.suptitle("Exponential growth is an approximation — but a good one where it matters",
                 fontsize=18, fontweight="bold", y=0.98)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
