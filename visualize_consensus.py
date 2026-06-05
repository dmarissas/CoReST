# Figures for Part 2 (Consensus / Robustness).
#
# Reads ONLY saved outputs of run_consensus.py — NO retraining, NO GPU:
#   results/consensus_ari.csv                       (all approaches: mean/std/best/worst)
#   results/consensus_perseed.csv                   (the 15 single-seed ARIs — the "lottery")
#   results/clusters_consensus_plain_k20_refined.npy (the headline consensus labels)
#   results/stability_consensus.npy                 (per-spot [confidence, stability])
#   processed/{barcodes,coords,labels}_final.csv
#
# Figures written to data/.../figures/:
#   consensus_vs_lottery.png  — single-seed scatter vs the ONE deterministic consensus
#   consensus_gain.png        — variants bar + gain decomposition (cross-seed is the driver)
#   consensus_spatial.png     — gold | consensus prediction | per-spot confidence map
#
# Style matches visualize.py (Agg, s=6 spatial scatter, tab20 for 20 domains, dpi=150).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("Part 2 figures — Consensus / Robustness")
print("=" * 60)

# ── Load (only saved artifacts) ─────────────────────────────────────
ari      = pd.read_csv(os.path.join(RESULT_DIR, "consensus_ari.csv"))
perseed  = pd.read_csv(os.path.join(RESULT_DIR, "consensus_perseed.csv"))
barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
xy = coords.loc[barcodes, ["x", "y"]].values
label_fine = labels.loc[barcodes, "fine_annot_type"].values


def val(approach, col="ARI_mean", refine="refined"):
    """Fetch one number from consensus_ari.csv, or None if the row is absent."""
    r = ari[(ari["approach"] == approach) & (ari["refine"] == refine)]
    return float(r.iloc[0][col]) if len(r) else None


best_single = max(v for v in [val("single_seed_kmeans", "ARI_best"),
                              val("single_seed_gmm", "ARI_best"),
                              val("single_seed_leiden", "ARI_best")] if v is not None)
plain    = val("consensus_plain")
gmm_mean = val("single_seed_gmm")

# ── Figure 1: the seed lottery vs the deterministic consensus ───────
print("\n[1] consensus_vs_lottery.png")
methods  = ["kmeans", "gmm", "leiden"]
colors_m = {"kmeans": "#95a5a6", "gmm": "#2980b9", "leiden": "#8e44ad"}
rng = np.random.RandomState(0)
fig, ax = plt.subplots(figsize=(9, 6))
for i, m in enumerate(methods):
    vals = perseed[perseed["method"] == m]["ARI_refined"].values
    if len(vals) == 0:
        continue
    jit = rng.uniform(-0.07, 0.07, size=len(vals))
    ax.scatter(np.full(len(vals), i) + jit, vals, s=80, color=colors_m[m],
               alpha=0.8, edgecolor="white", zorder=3, label=f"{m} — 5 seeds")
    ax.hlines(vals.mean(), i - 0.18, i + 0.18, color=colors_m[m], lw=2.5, zorder=4)

lo_w, lo_b = val("consensus_loso", "ARI_worst"), val("consensus_loso", "ARI_best")
if lo_w is not None and lo_b is not None:
    ax.axhspan(lo_w, lo_b, color="#27ae60", alpha=0.10, zorder=0,
               label=f"LOSO 4-seed consensus range [{lo_w:.3f}, {lo_b:.3f}]")
ax.axhline(plain, color="#27ae60", lw=2.6, zorder=2,
           label=f"plain consensus = {plain:.4f}  (deterministic)")
ax.axhline(best_single, color="#e74c3c", ls="--", lw=1.8, zorder=2,
           label=f"best single seed = {best_single:.4f}")
ax.axhline(gmm_mean, color="#7f8c8d", ls=":", lw=1.6, zorder=1,
           label=f"GMM mean (typical run) = {gmm_mean:.4f}")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels([m.upper() for m in methods])
ax.set_ylabel("Fine ARI (k=20, refined)")
ax.set_title("Removing the seed lottery (single seed pool)\n"
             "consensus = ONE deterministic answer, reliably at/above the pool's seeds (seed-insurance)\n"
             "0.6504 here is a lucky pool — typical ~0.61 across seed pools (see seed-pool replication)",
             fontsize=11)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "consensus_vs_lottery.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: consensus_vs_lottery.png")

# ── Figure 2: variants bar + gain decomposition ─────────────────────
print("\n[2] consensus_gain.png")
fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))

# Panel A — single-seed clusterers vs the 4 consensus variants
labelsA = ["KMeans\n(seed mean)", "GMM\n(seed mean)", "Leiden\n(seed mean)",
           "Consensus\nplain", "Consensus\nweighted", "Consensus\nspatial", "Consensus\nheadline"]
meansA = [val("single_seed_kmeans"), val("single_seed_gmm"), val("single_seed_leiden"),
          val("consensus_plain"), val("consensus_weighted"), val("consensus_spatial"),
          val("consensus_headline")]
stdsA = [val("single_seed_kmeans", "ARI_std"), val("single_seed_gmm", "ARI_std"),
         val("single_seed_leiden", "ARI_std"), 0, 0, 0, 0]
colsA = ["#95a5a6", "#2980b9", "#8e44ad", "#27ae60", "#16a085", "#16a085", "#16a085"]
xA = np.arange(len(labelsA))
barsA = axL.bar(xA, meansA, yerr=stdsA, capsize=4, color=colsA, alpha=0.9,
                error_kw={"linewidth": 1.3})
axL.axhline(best_single, color="#e74c3c", ls="--", lw=1.6,
            label=f"best single seed = {best_single:.4f}")
for b, m, sd in zip(barsA, meansA, stdsA):
    axL.text(b.get_x() + b.get_width() / 2, b.get_height() + (sd or 0) + 0.008, f"{m:.3f}",
             ha="center", va="bottom", fontsize=8)
axL.set_xticks(xA); axL.set_xticklabels(labelsA, fontsize=8)
axL.set_ylabel("Fine ARI (k=20, refined)"); axL.set_ylim(0, 0.72)
axL.set_title("Consensus vs single-seed clusterers (pool A)\n"
              "consensus lands at/above the best single seed — margin within seed noise (seed-insurance); twists < plain")
axL.legend(fontsize=8, loc="lower left"); axL.grid(axis="y", alpha=0.3)

# Panel B — gain decomposition
labelsB = ["Ward agg.\n(1 embed)", "GMM mean\n(typical)", "+ cross-method\n(1 seed, 3 styles)",
           "+ cross-seed\n(5 seeds, GMM)", "Full plain\nconsensus"]
meansB = [val("ablation_agg_single_embed"), val("single_seed_gmm"),
          val("ablation_singleseed_xmethod"), val("ablation_xseed_gmm"), val("consensus_plain")]
colsB = ["#bdc3c7", "#2980b9", "#f39c12", "#e67e22", "#27ae60"]
xB = np.arange(len(labelsB))
barsB = axR.bar(xB, meansB, color=colsB, alpha=0.9)
for b, m in zip(barsB, meansB):
    axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{m:.3f}",
             ha="center", va="bottom", fontsize=8)
if meansB[1] is not None and meansB[3] is not None:
    axR.annotate(f"+{meansB[3]-meansB[1]:.3f}\ncross-seed", xy=(3, meansB[3]),
                 xytext=(3, meansB[3] + 0.045), ha="center", fontsize=9,
                 color="#d35400", fontweight="bold")
axR.set_xticks(xB); axR.set_xticklabels(labelsB, fontsize=8)
axR.set_ylabel("Fine ARI (k=20, refined)"); axR.set_ylim(0, 0.72)
axR.set_title("Where the lift comes from (within a pool)\n"
              "cross-SEED pooling is the primary driver; cross-method adds the rest")
axR.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "consensus_gain.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: consensus_gain.png")

# ── Figure 3: gold | consensus prediction | per-spot confidence ─────
print("\n[3] consensus_spatial.png")
unique_fine = sorted(np.unique(label_fine))
fine_to_int = {l: i for i, l in enumerate(unique_fine)}
fine_ints = np.array([fine_to_int[l] for l in label_fine])
cmap20 = plt.cm.get_cmap("tab20", len(unique_fine))

fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
axes[0].scatter(xy[:, 0], xy[:, 1], c=fine_ints, cmap=cmap20, s=6,
                vmin=0, vmax=len(unique_fine) - 1, alpha=0.9)
axes[0].set_title("Gold standard (fine, 20 domains)", fontsize=12)

cpath = os.path.join(RESULT_DIR, "clusters_consensus_plain_k20_refined.npy")
if os.path.exists(cpath):
    pred = np.load(cpath)
    axes[1].scatter(xy[:, 0], xy[:, 1], c=pred, cmap=plt.cm.get_cmap("tab20", 20),
                    s=6, alpha=0.9)
    axes[1].set_title(f"Plain consensus prediction\n(refined, ARI = {plain:.4f})", fontsize=12)
else:
    axes[1].text(0.5, 0.5, "No data\n(run run_consensus.py)", ha="center", va="center",
                 transform=axes[1].transAxes)
    axes[1].set_title("Plain consensus prediction", fontsize=12)

spath = os.path.join(RESULT_DIR, "stability_consensus.npy")
if os.path.exists(spath):
    conf = np.load(spath)[:, 0]
    sc = axes[2].scatter(xy[:, 0], xy[:, 1], c=conf, cmap="magma", s=6, alpha=0.95,
                         vmin=0, vmax=1)
    fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04, label="co-association confidence")
    axes[2].set_title(f"Per-spot consensus confidence\n(label-free; mean = {conf.mean():.3f})",
                      fontsize=12)
else:
    axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Per-spot confidence", fontsize=12)

for a in axes:
    a.set_aspect("equal"); a.axis("off")
plt.suptitle("Consensus: gold standard vs prediction vs label-free per-spot confidence", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "consensus_spatial.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: consensus_spatial.png")

print(f"\nAll Part-2 figures saved to: {FIG_DIR}")
