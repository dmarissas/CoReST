# Part 2 ROBUSTNESS figure (from consensus_robustness.py results).
#
# Panel A: "run once" (20 single-seed ARIs) vs "run 5 + consensus" (60 pools) —
#          shows the lift (+0.037), the tighter spread (~1.4x), and determinism.
# Panel B: the margins — consensus beats a TYPICAL run almost always, but only
#          TIES the best-of-pool (the honest "coin flip vs best seed").
#
# Reads results/consensus_robustness_pools.csv (the 60 pools) and recomputes the
# 20 single-seed ARIs from the cached embeddings (cheap; same recipe as the study,
# no SEDR training). Style matches the other viz scripts (Agg, dpi=150).

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from utils_func import cluster_latent, refine_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
N_REFINE_NEIGH = 6

# Must match consensus_robustness.py (the 20 seeds it trained/used).
SEEDS_ALL = [42, 123, 456, 789, 1234, 7, 88, 314, 2024, 51966,
             11, 222, 333, 444, 555, 1212, 2323, 3434, 4545, 5656]

print("=" * 60)
print("Part 2 robustness figure — consensus_robustness.png")
print("=" * 60)

# ── Load pools + recompute single-seed ARIs ─────────────────────────
ppath = os.path.join(RESULT_DIR, "consensus_robustness_pools.csv")
if not os.path.exists(ppath):
    print(f"MISSING {ppath} — run consensus_robustness.py first.")
    sys.exit(1)
pools = pd.read_csv(ppath)
cons   = pools["consensus"].values
m_mean = pools["margin_vs_mean"].values
m_best = pools["margin_vs_best"].values

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
gold = labels.loc[barcodes, "fine_annot_type"].values
xy   = coords.loc[barcodes, ["x", "y"]].values
K = len(np.unique(gold))

print("Recomputing 20 single-seed ARIs (GMM, refined) from cached embeddings ...")
singles = []
for s in SEEDS_ALL:
    z = np.load(os.path.join(RESULT_DIR, f"embeddings_consensus_gene_seed{s}_fine.npy"))
    lab = refine_labels(cluster_latent(z, K, s, method="gmm"), xy, N_REFINE_NEIGH)
    singles.append(adjusted_rand_score(gold, lab))
singles = np.array(singles)

s_mean, s_std = singles.mean(), singles.std()
c_mean, c_std = cons.mean(), cons.std()
var_ratio = s_std / c_std if c_std > 1e-9 else float("inf")
n_vs_mean = int((m_mean > 0).sum())
n_vs_best = int((m_best >= -1e-9).sum())
print(f"  single {s_mean:.4f}±{s_std:.4f}   consensus {c_mean:.4f}±{c_std:.4f}   "
      f"var ratio {var_ratio:.2f}x   vs_mean>{0}: {n_vs_mean}/{len(m_mean)}   "
      f"vs_best>=0: {n_vs_best}/{len(m_best)}")

# ── Figure ──────────────────────────────────────────────────────────
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
rng = np.random.RandomState(0)

# Panel A — the two clouds
for xc, vals, color, lab in [
        (0, singles, "#95a5a6", f"run once — single seed (n={len(singles)}):  {singles.mean():.3f} ± {singles.std():.3f}"),
        (1, cons, "#27ae60", f"run 5 + consensus (n={len(cons)} pools):  {cons.mean():.3f} ± {cons.std():.3f}")]:
    jit = rng.uniform(-0.10, 0.10, size=len(vals))
    axA.scatter(np.full(len(vals), xc) + jit, vals, s=45, color=color, alpha=0.7,
                edgecolor="white", linewidth=0.4, zorder=3, label=lab)
    axA.errorbar(xc, vals.mean(), yerr=vals.std(), fmt="_", color="black", markersize=46,
                 elinewidth=2, capsize=10, capthick=2, zorder=4)
axA.annotate("", xy=(1, c_mean), xytext=(0, s_mean),
             arrowprops=dict(arrowstyle="->", color="#1e8449", lw=2))
axA.text(0.5, (s_mean + c_mean) / 2 - 0.02, f"lift +{c_mean - s_mean:.3f}",
         ha="center", color="#1e8449", fontsize=11, fontweight="bold")
axA.set_xlim(-0.5, 1.5); axA.set_xticks([0, 1])
axA.set_xticklabels(["Run once\n(single seed)", "Run 5 + consensus"])
axA.set_ylim(singles.min() - 0.03, cons.max() + 0.03)
axA.set_ylabel("Fine ARI (k=20, refined)")
axA.set_title(f"Run once vs run-5-and-consensus\n"
              f"consensus is higher (+{c_mean - s_mean:.3f}), tighter ({var_ratio:.1f}×), and deterministic")
axA.grid(axis="y", alpha=0.3); axA.legend(fontsize=8, loc="lower right")

# Panel B — the margins (vs typical run, vs best-of-pool)
lo = min(m_best.min(), m_mean.min()) - 0.005
hi = max(m_best.max(), m_mean.max()) + 0.005
bins = np.linspace(lo, hi, 22)
axB.hist(m_mean, bins=bins, color="#27ae60", alpha=0.65,
         label=f"vs a TYPICAL run  ({n_vs_mean}/{len(m_mean)} above 0)")
axB.hist(m_best, bins=bins, color="#e67e22", alpha=0.65,
         label=f"vs the BEST-of-pool  ({n_vs_best}/{len(m_best)} ≥ 0 — a coin flip)")
axB.axvline(0, color="black", ls="--", lw=1.5)
axB.axvline(m_mean.mean(), color="#1e8449", lw=2, label=f"mean vs typical = +{m_mean.mean():.3f}")
axB.axvline(m_best.mean(), color="#b9770e", lw=2,
            label=f"mean vs best = {m_best.mean():+.3f}")
axB.set_xlabel("consensus ARI − single-seed ARI")
axB.set_ylabel("number of pools (of 60)")
axB.set_title("Consensus beats a TYPICAL run, but TIES the best-of-pool\n"
              "(you can't pick the best seed without labels — consensus ≈ matches it, label-free)")
axB.legend(fontsize=8, loc="upper left"); axB.grid(axis="y", alpha=0.3)

plt.suptitle("Consensus robustness — 20 seeds, 60 sampled 5-seed pools (HBRC, fine k=20)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(FIG_DIR, "consensus_robustness.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {os.path.join('figures', 'consensus_robustness.png')}")
