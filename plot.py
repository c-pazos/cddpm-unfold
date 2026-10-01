import os
import numpy as np
import matplotlib.pyplot as plt
import configs

if not os.path.exists(configs.plots_path):
    os.makedirs(configs.plots_path)

# load test truth, test reco, and unfolded predictions
truth = np.load(configs.x_test_file)
reco = np.load(configs.y_test_file)
unfolded = np.load(configs.output_path + "pred_x_" + configs.project + ".npy")

# unfold.py can drop leftover events, so match lengths
n = len(unfolded)
truth = truth[:n]
reco = reco[:n]

n_bins = 50

for i in range(configs.n_dims):

    # common binning covering all three distributions
    lo = min(truth[:, i].min(), reco[:, i].min(), unfolded[:, i].min())
    hi = max(truth[:, i].max(), reco[:, i].max(), unfolded[:, i].max())
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    h_truth, _ = np.histogram(truth[:, i], bins=bins)
    h_reco, _ = np.histogram(reco[:, i], bins=bins)
    h_unf, _ = np.histogram(unfolded[:, i], bins=bins)

    # ratio and its statistical uncertainty (both histograms treated as Poisson)
    mask = (h_truth > 0) & (h_unf > 0)
    ratio = np.full(n_bins, np.nan)
    ratio_err = np.full(n_bins, np.nan)
    ratio[mask] = h_unf[mask] / h_truth[mask]
    ratio_err[mask] = ratio[mask] * np.sqrt(1 / h_unf[mask] + 1 / h_truth[mask])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # top panel: distributions
    ax_top.stairs(h_truth, bins, label="truth", color="black", linewidth=1.5)
    ax_top.stairs(h_reco, bins, label="reco", color="tab:blue", linestyle="--")
    ax_top.stairs(h_unf, bins, label="unfolded", color="tab:red")
    ax_top.set_ylabel("events")
    ax_top.legend()

    # bottom panel: unfolded / truth
    ax_bot.errorbar(centers, ratio, yerr=ratio_err, fmt="o", markersize=3, color="tab:red")
    ax_bot.axhline(1, color="black", linewidth=1)
    ax_bot.set_ylim(0.8, 1.2)
    ax_bot.set_ylabel("unfolded / truth")
    ax_bot.set_xlabel(f"component {i}")

    plt.savefig(configs.plots_path + f"unfolded_{configs.project}_{i}.png", bbox_inches="tight")
    plt.close()

print("done!")