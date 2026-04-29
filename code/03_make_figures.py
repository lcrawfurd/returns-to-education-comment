"""Generate standalone PDF figures for the Clark & Nielsen comment.

Outputs (in figures/):
  fig1_funnel.pdf       — funnel plot
  fig2_petpeese.pdf     — PET-PEESE fits
  fig4_tstats.pdf       — distribution of t-statistics
  fig5_normality.pdf    — effect-size distribution with fitted normals
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).parent.parent
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Load data ---
ind = pd.read_stata(ROOT / "data" / "Returns_to_education.dta")
ind = ind[ind["ind_est"] == 1].dropna(subset=["effect_percentage", "se_percentage"])
effect = ind["effect_percentage"].to_numpy()
se = ind["se_percentage"].to_numpy()
tstat = effect / se

# --- Figure 1: funnel plot ---
mask = se <= 20
w = 1.0 / se**2
mean_w = np.sum(w * effect) / np.sum(w)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
se_range = np.linspace(0.01, max(se[mask]) * 1.1, 100)
for z, ls in [(1.96, "--"), (2.58, ":")]:
    ax.plot(mean_w + z * se_range, se_range, color="gray", ls=ls, alpha=0.6)
    ax.plot(mean_w - z * se_range, se_range, color="gray", ls=ls, alpha=0.6)
ax.scatter(effect[mask], se[mask], s=28, alpha=0.75, color="steelblue",
           edgecolors="white", linewidth=0.5, zorder=5)
ax.axvline(mean_w, color="black", lw=1, alpha=0.5,
           label=f"Weighted mean ({mean_w:.1f}\\%)")
ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
ax.set_xlabel("Estimated return to a year of schooling (\\%)")
ax.set_ylabel("Standard error (\\%)")
ax.invert_yaxis()
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig1_funnel.pdf")
plt.close(fig)

# --- Figure 2: PET-PEESE ---
X_pet = sm.add_constant(se)
pet = sm.WLS(effect, X_pet, weights=1.0 / se**2).fit(cov_type="HC1")
X_peese = sm.add_constant(se**2)
peese = sm.WLS(effect, X_peese, weights=1.0 / se**2).fit(cov_type="HC1")

fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.scatter(se, effect, s=28, alpha=0.75, color="steelblue",
           edgecolors="white", linewidth=0.5)
xs = np.linspace(0, max(se) * 1.05, 100)
ax.plot(xs, pet.params[0] + pet.params[1] * xs, color="firebrick", ls="--",
        lw=2, label=f"PET: {pet.params[0]:.1f}\\%")
ax.plot(xs, peese.params[0] + peese.params[1] * xs**2, color="darkgreen",
        lw=2, label=f"PEESE: {peese.params[0]:.1f}\\%")
ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
ax.set_xlabel("Standard error (\\%)")
ax.set_ylabel("Estimated return (\\%)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_petpeese.pdf")
plt.close(fig)

# --- Figure 4: t-statistic distribution ---
fig, ax = plt.subplots(figsize=(5.5, 3.6))
valid = tstat[(tstat > -5) & (tstat < 12)]
ax.hist(valid, bins=np.arange(-5, 12, 0.5), color="steelblue", alpha=0.75,
        edgecolor="white")
ax.axvline(1.96, color="firebrick", ls="--", lw=2, label="$t=1.96$")
ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
ax.set_xlabel("$t$-statistic")
ax.set_ylabel("Frequency")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig4_tstats.pdf")
plt.close(fig)

# --- Figure 5: effect-size distribution + fitted normals ---
fig, ax = plt.subplots(figsize=(5.5, 3.8))
eplot = effect[effect < 40]
ax.hist(eplot, bins=np.arange(-15, 35, 2.5), color="steelblue", alpha=0.75,
        edgecolor="white", density=True)
mu, sigma = stats.norm.fit(eplot)
xs = np.linspace(-15, 35, 200)
ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="firebrick", lw=2,
        label=f"Fitted normal ($\\mu={mu:.1f}$, $\\sigma={sigma:.1f}$)")
ax.plot(xs, stats.norm.pdf(xs, 0, sigma), color="gray", ls="--", lw=1.5,
        label=f"Normal at 0 ($\\sigma={sigma:.1f}$)")
ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
ax.set_xlabel("Estimated return to a year of schooling (\\%)")
ax.set_ylabel("Density")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig5_normality.pdf")
plt.close(fig)

print(f"Wrote 4 PDFs to {OUT}")
