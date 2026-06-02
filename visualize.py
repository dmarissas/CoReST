# Generate all figures for the final presentation/paper.
#
# Reads the long-format results/ari_results.csv
#   (condition, graph_mode, model, resolution, cluster_method, refine, ARI_mean, ...)
# and the clusters_<cond>_{k20,k4}[_refined].npy produced by cluster_regeneration.py.
#
# Figures:
#   1. results_barplot.png        — ARI per condition (SEDR/KMeans, raw vs refined)
#   2. spatial_clusters_<k>.png   — predicted clusters per condition in space
#   3. gold_standard_map.png      — gold standard (fine, 20 domains)
#   4. comparison_map.png         — gold vs image-gated-graph prediction (the win)
#   5. tsne_embeddings.png        — t-SNE of SEDR embeddings, colored by gold standard

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
# ──────────────────────────────────────────────────────────────────

PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 60)
print("STEP 5: Visualization")
print("=" * 60)

# ── Load data ──────────────────────────────────────────────────────
barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
df_ari   = pd.read_csv(os.path.join(RESULT_DIR, "ari_results.csv"))

xy = coords.loc[barcodes, ["x", "y"]].values
label_coarse = labels.loc[barcodes, "annot_type"].values
label_fine   = labels.loc[barcodes, "fine_annot_type"].values
unique_fine = sorted(np.unique(label_fine))
fine_to_int = {l: i for i, l in enumerate(unique_fine)}
fine_ints   = np.array([fine_to_int[l] for l in label_fine])
cmap20      = plt.cm.get_cmap("tab20", len(unique_fine))

conditions = ["gene_only", "image_only", "concat_fused", "gated_fused", "gene_imagegraph"]
conditions_display = {
    "gene_only":       "Gene only\n(spatial)",
    "image_only":      "Image only\n(spatial)",
    "concat_fused":    "Concat\n(spatial)",
    "gated_fused":     "Feature gate\n(spatial)",
    "gene_imagegraph": "Image-gated\ngraph",
}

PUBLISHED = {"Seurat": 0.4612, "SEDR": 0.3668, "STAGATE": 0.4944, "TGR-NMF": 0.5286}


def get_ari(cond, resolution, cluster_method="kmeans", refine="raw", model="SEDR"):
    r = df_ari[(df_ari["condition"] == cond) & (df_ari["resolution"] == resolution) &
               (df_ari["cluster_method"] == cluster_method) & (df_ari["refine"] == refine) &
               (df_ari["model"] == model)]
    if len(r) == 0:
        return None, None
    return float(r["ARI_mean"].values[0]), float(r["ARI_std"].values[0])


# ── Figure 1: ARI bar chart (refined; KMeans vs GMM — shows the clusterer flip) ──
print("\n[1] Plotting ARI bar chart...")
x = np.arange(len(conditions))
width = 0.38
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

for ax_idx, (res, title, ylim) in enumerate([
    ("fine",   "Fine-grained ARI (k=20)", 0.65),
    ("coarse", "Coarse ARI (k=4)",        0.40),
]):
    ax = axes[ax_idx]
    # Show both clusterers (refined) so the gene-vs-graph ranking flip is visible.
    for bi, (method, color) in enumerate([("kmeans", "#95a5a6"), ("gmm", "#2980b9")]):
        means = [get_ari(c, res, method, "refined")[0] or 0 for c in conditions]
        stds  = [get_ari(c, res, method, "refined")[1] or 0 for c in conditions]
        bars = ax.bar(x + (bi - 0.5) * width, means, width, yerr=stds, capsize=4,
                      label=f"SEDR + {method.upper()} (refined)", color=color, alpha=0.9,
                      error_kw={"linewidth": 1.3})
        for bar, m in zip(bars, means):
            if m > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

    if res == "fine":
        for (nm, val), col in zip(PUBLISHED.items(),
                                  ["#e74c3c", "#e67e22", "#9b59b6", "#27ae60"]):
            ax.axhline(val, color=col, linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"{nm}={val:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([conditions_display[c] for c in conditions], fontsize=8.5)
    ax.set_ylabel("ARI (refined)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="upper left", ncol=1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ylim)

plt.suptitle("Refined ARI across feature/graph conditions, KMeans vs GMM — ±1 std over 5 seeds\n"
             "Feature fusion (concat/gate) < gene-only under both; image-gated graph ties gene-only "
             "(wins KMeans, loses GMM)", fontsize=10.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "results_barplot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: results_barplot.png")

# ── Figure 2: Spatial cluster maps (refined, all conditions) ───────
print("\n[2] Plotting spatial cluster maps...")
for lt in ["k20", "k4"]:
    n_clusters = 20 if lt == "k20" else 4
    cmap = plt.cm.get_cmap("tab20" if n_clusters > 10 else "Set1", n_clusters)
    fig, axes = plt.subplots(1, len(conditions), figsize=(5*len(conditions), 6))
    for ci, cond in enumerate(conditions):
        # prefer refined clusters; fall back to raw
        path = os.path.join(RESULT_DIR, f"clusters_{cond}_{lt}_refined.npy")
        if not os.path.exists(path):
            path = os.path.join(RESULT_DIR, f"clusters_{cond}_{lt}.npy")
        if not os.path.exists(path):
            axes[ci].text(0.5, 0.5, "No data", ha="center", va="center",
                          transform=axes[ci].transAxes)
            axes[ci].set_title(conditions_display.get(cond, cond)); axes[ci].axis("off")
            continue
        pred = np.load(path)
        axes[ci].scatter(xy[:, 0], xy[:, 1], c=pred, cmap=cmap, s=6,
                         vmin=0, vmax=n_clusters-1, alpha=0.9)
        axes[ci].set_title(conditions_display.get(cond, cond), fontsize=11)
        axes[ci].set_aspect("equal"); axes[ci].axis("off")
    plt.suptitle(f"Predicted Spatial Clusters ({lt}, refined) — SEDR", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"spatial_clusters_{lt}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: spatial_clusters_{lt}.png")

# ── Figure 3: Gold standard fine map ───────────────────────────────
print("\n[3] Plotting gold standard labels...")
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(xy[:, 0], xy[:, 1], c=fine_ints, cmap=cmap20, s=6,
           vmin=0, vmax=len(unique_fine)-1, alpha=0.9)
patches = [mpatches.Patch(color=cmap20(i), label=l) for i, l in enumerate(unique_fine)]
ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2, bbox_to_anchor=(1.35, 1.0))
ax.set_title("Gold Standard: fine_annot_type (Xu et al., 20 domains)", fontsize=12)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gold_standard_map.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: gold_standard_map.png")

# ── Figure 4: Gold vs image-gated-graph prediction (the win) ───────
print("\n[4] Plotting image-gated-graph vs gold standard comparison...")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

unique_labels = sorted(np.unique(label_coarse))
label_ints = np.array([unique_labels.index(l) for l in label_coarse])
cmap4 = plt.cm.get_cmap("Set2", len(unique_labels))
axes[0].scatter(xy[:, 0], xy[:, 1], c=label_ints, cmap=cmap4, s=6,
                vmin=0, vmax=len(unique_labels)-1, alpha=0.9)
axes[0].legend(handles=[mpatches.Patch(color=cmap4(i), label=l)
                        for i, l in enumerate(unique_labels)], fontsize=9, loc="upper right")
axes[0].set_title("Gold Standard\n(coarse, 4 classes)", fontsize=12)
axes[0].set_aspect("equal"); axes[0].axis("off")

axes[1].scatter(xy[:, 0], xy[:, 1], c=fine_ints, cmap=cmap20, s=6,
                vmin=0, vmax=len(unique_fine)-1, alpha=0.9)
axes[1].set_title("Gold Standard\n(fine, 20 domains)", fontsize=12)
axes[1].set_aspect("equal"); axes[1].axis("off")

# winner: gene features on the image-gated graph (refined clusters), dynamic ARI label
win = "gene_imagegraph"
wpath = os.path.join(RESULT_DIR, f"clusters_{win}_k20_refined.npy")
if not os.path.exists(wpath):
    wpath = os.path.join(RESULT_DIR, f"clusters_{win}_k20.npy")
pred = np.load(wpath)
w_mean, w_std = get_ari(win, "fine", "kmeans", "refined")
lbl = f"\n(k=20, refined ARI={w_mean:.4f} ± {w_std:.4f})" if w_mean else ""
axes[2].scatter(xy[:, 0], xy[:, 1], c=pred, cmap=plt.cm.get_cmap("tab20", 20), s=6,
                vmin=0, vmax=19, alpha=0.9)
axes[2].set_title(f"Image-gated graph SEDR{lbl}", fontsize=12)
axes[2].set_aspect("equal"); axes[2].axis("off")

plt.suptitle("Gold Standard vs Image-gated-graph Prediction", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "comparison_map.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: comparison_map.png")

# ── Figure 5: t-SNE of SEDR embeddings ─────────────────────────────
print("\n[5] Plotting t-SNE of SEDR embeddings...")
fig, axes = plt.subplots(1, len(conditions), figsize=(5*len(conditions), 6))
for ci, cond in enumerate(conditions):
    emb_path = os.path.join(RESULT_DIR, f"embeddings_{cond}.npy")
    if not os.path.exists(emb_path):
        axes[ci].text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=axes[ci].transAxes); axes[ci].axis("off")
        continue
    emb = np.load(emb_path)
    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb)
    axes[ci].scatter(proj[:, 0], proj[:, 1], c=fine_ints, cmap=cmap20, s=4,
                     vmin=0, vmax=len(unique_fine)-1, alpha=0.7)
    axes[ci].set_title(conditions_display.get(cond, cond), fontsize=11)
    axes[ci].axis("off")
plt.suptitle("t-SNE of SEDR Embeddings — colored by gold standard fine domains", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tsne_embeddings.png"), dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: tsne_embeddings.png")

print(f"\nAll figures saved to: {FIG_DIR}")
