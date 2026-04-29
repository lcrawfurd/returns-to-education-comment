#!/usr/bin/env python3
"""
cn_simulation.py

Simulation: show CN's truncated-normal procedure recovers a near-zero mean
even when the true effect is 6% and there is no publication bias.

Left panel: distribution of recovered mu across 200 simulated datasets
Right panel: CN's Figure 7 reproduced — truncated-normal fit (using CN's
             reported mu=0.2, sigma=12.1) vs the unconstrained MLE normal
             (mu=8.0, sigma=6.3) which does not assume truncation.

Outputs:
  output/figures/fig6_simulation.pdf  — two-panel figure
  output/cn_simulation_results.json  — key numbers for the paper
"""

import warnings
import json
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "Returns_to_education.dta"
FIG_OUT   = ROOT / "output" / "figures" / "fig6_simulation.pdf"
JSON_OUT  = ROOT / "output" / "cn_simulation_results.json"

np.random.seed(20260417)
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_stata(DATA_PATH)
df = df[df["ind_est"] == 1].dropna(subset=["effect_percentage", "se_percentage"])
effects = df["effect_percentage"].values

# CN's reported Figure 7 parameters (their published result)
CN_MU    = 0.2
CN_SIGMA = 12.1
# MLE normal fit to the full data (from Figure 5)
MLE_MU    = 8.0
MLE_SIGMA = 6.3

BIN_WIDTH = 2.5

def make_bins(eff):
    pos = eff[eff > 0]
    max_val = min(pos.max(), 40.0)
    edges = np.arange(0, max_val + BIN_WIDTH, BIN_WIDTH)
    counts, _ = np.histogram(pos, bins=edges)
    return edges, counts, len(pos)

# ---------------------------------------------------------------------------
# CN's truncated-normal grid search (for simulation)
# ---------------------------------------------------------------------------
def cn_grid_fit(eff_arr, step=0.5):
    mu_grid    = np.arange(-15, 20, step)
    sigma_grid = np.arange(3, 25, step)
    edges, obs, n_pos = make_bins(eff_arr)
    best_sse = np.inf; best_mu = best_sigma = np.nan
    for mu in mu_grid:
        for sigma in sigma_grid:
            try:
                a = -mu / sigma
                tn = stats.truncnorm(a=a, b=np.inf, loc=mu, scale=sigma)
                sse = float(np.sum((obs - np.diff(tn.cdf(edges)) * n_pos) ** 2))
                if sse < best_sse:
                    best_sse = sse; best_mu = mu; best_sigma = sigma
            except Exception:
                continue
    return float(best_mu), float(best_sigma)

# ---------------------------------------------------------------------------
# Simulation: true mean=6%, SD=6%, log-normal, no publication bias
# ---------------------------------------------------------------------------
sigma_ln_sim = float(np.sqrt(np.log(2)))       # SD/mean=1 => sigma_ln=sqrt(ln2)
mu_ln_sim    = float(np.log(6) - sigma_ln_sim**2 / 2)

N_SIM   = 200
n_draws = 71
sim_mus = []

print(f"Running {N_SIM} simulations (n={n_draws}, log-normal: mean=6%, SD=6%)...")
for i in range(N_SIM):
    if i % 50 == 0:
        print(f"  iteration {i}/{N_SIM}")
    sim_eff = np.random.lognormal(mean=mu_ln_sim, sigma=sigma_ln_sim, size=n_draws)
    mu_hat, _ = cn_grid_fit(sim_eff)
    sim_mus.append(mu_hat)

sim_mus     = np.array(sim_mus)
sim_median  = float(np.median(sim_mus))
sim_p10     = float(np.percentile(sim_mus, 10))
sim_p90     = float(np.percentile(sim_mus, 90))
pct_below_3 = float(np.mean(sim_mus < 3.0) * 100)
pct_below_0 = float(np.mean(sim_mus < 0.0) * 100)

print(f"  Median recovered: {sim_median:.2f}%  [10th-90th: {sim_p10:.1f}% to {sim_p90:.1f}%]")
print(f"  < 3%: {pct_below_3:.0f}%,  < 0%: {pct_below_0:.0f}%")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: CN's Figure 7 — observed histogram + CN fit + MLE fit
ax2 = axes[0]
edges, obs_counts, n_pos = make_bins(effects)
bin_centres = (edges[:-1] + edges[1:]) / 2

# CN's truncated-normal fit (their reported values)
tn_cn = stats.truncnorm(a=-CN_MU/CN_SIGMA, b=np.inf, loc=CN_MU, scale=CN_SIGMA)
exp_cn = np.diff(tn_cn.cdf(edges)) * n_pos

# MLE normal fit to full data (no truncation assumption)
norm_mle = stats.norm(loc=MLE_MU, scale=MLE_SIGMA)
# Scale so total area under the plotted range = n_pos
mle_probs = np.diff(norm_mle.cdf(edges))
exp_mle   = mle_probs / mle_probs.sum() * n_pos

ax2.bar(bin_centres, obs_counts, width=2.2, color="#AAAAAA", alpha=0.6, label="Observed")
ax2.plot(bin_centres, exp_cn,  "o-", color="crimson",  linewidth=2, markersize=4,
         label=f"CN fit: truncated normal ($\\hat{{\\mu}}=0.2\\%$)")
ax2.plot(bin_centres, exp_mle, "s-", color="#2ca02c",  linewidth=2, markersize=4,
         label=f"MLE normal (no truncation, $\\hat{{\\mu}}=8.0\\%$)")
ax2.set_xlabel("Return to schooling (%)", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("CN Figure 7: two interpretations of the same data", fontsize=11)
ax2.legend(fontsize=9, loc="upper right")
ax2.spines[["top", "right"]].set_visible(False)

# Right: simulation histogram
ax = axes[1]
ax.hist(sim_mus, bins=25, color="#4C72B0", alpha=0.72, edgecolor="white", linewidth=0.4)
ax.axvline(sim_median, color="crimson", linestyle="-",  linewidth=1.8,
           label=f"Median: {sim_median:.1f}%")
ax.axvline(6.0,        color="black",   linestyle="--", linewidth=1.8,
           label="True mean: 6%")
ax.axvline(CN_MU,      color="#888888", linestyle=":",  linewidth=1.4,
           label=f"CN: {CN_MU}%")
ax.set_xlabel("CN procedure: recovered $\\hat{\\mu}$ (%)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"Simulation ({N_SIM} draws, $n={n_draws}$ each)", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=2.0)
fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figure saved to {FIG_OUT}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "cn_normal_fit_actual":  {"mu_pct": CN_MU, "sigma_pct": CN_SIGMA, "note": "CN published values"},
    "mle_normal_fit_actual": {"mu_pct": MLE_MU, "sigma_pct": MLE_SIGMA, "note": "MLE without truncation"},
    "simulation": {
        "n_sim": N_SIM, "n_per_draw": n_draws,
        "true_mean_pct": 6.0, "true_sd_pct": 6.0, "distribution": "lognormal",
        "median_recovered_pct": round(sim_median, 2),
        "p10_recovered_pct":    round(sim_p10, 2),
        "p90_recovered_pct":    round(sim_p90, 2),
        "pct_below_3pct":       round(pct_below_3, 1),
        "pct_below_0pct":       round(pct_below_0, 1)
    }
}
with open(JSON_OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
