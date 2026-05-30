# Generate all figures for the final presentation/paper.
#
# Figures produced:
#   1. results_barplot.png       — ARI comparison bar chart (all conditions)
#   2. spatial_clusters_<cond>.png — spatial map of predicted clusters per condition
#   3. gate_map.png              — spatial map of gate values (gated fusion only)
#   4. umap_embeddings.png       — UMAP of SEDR embeddings colored by gold standard

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
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
barcodes  = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
coords    = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels    = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
df_ari    = pd.read_csv(os.path.join(RESULT_DIR, "ari_results.csv"))
gates     = np.load(os.path.join(PROC_DIR, "gates.npy"))

xy = coords.loc[barcodes, ["x","y"]].values
label_coarse = labels.loc[barcodes, "annot_type"].values
label_fine  = labels.loc[barcodes, "fine_annot_type"].values
unique_fine = sorted(np.unique(label_fine))
fine_to_int = {l: i for i, l in enumerate(unique_fine)}
fine_ints   = np.array([fine_to_int[l] for l in label_fine])
cmap20      = plt.cm.get_cmap("tab20", len(unique_fine))

# ── Figure 1: ARI bar chart ───────────────────────────────────────
print("\n[1] Plotting ARI bar chart...")

sedr_rows = df_ari[df_ari["model"] == "SEDR"]
km_rows   = df_ari[df_ari["model"] == "KMeans"]

conditions_display = {
    "gene_only":    "Gene only\n(200d)",
    "image_only":   "Image only\n(256d)",
    "concat_fused": "Concat\n(456d)",
    "gated_fused":  "Gated\n(456d)",
}

x      = np.arange(len(conditions_display))
width  = 0.35
colors = {"KMeans": "#95a5a6", "SEDR": "#2980b9"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (label_type, title) in enumerate([
    ("fine",   "Fine-grained ARI (k=20)"),
    ("coarse", "Coarse ARI (k=4)"),
]):
    ax = axes[ax_idx]
    for mi, (model, rows_df) in enumerate([("KMeans", km_rows), ("SEDR", sedr_rows)]):
        means, stds = [], []
        for cond in conditions_display:
            row = rows_df[rows_df["condition"].str.contains(cond)]
            if len(row) == 0:
                means.append(0); stds.append(0)
                continue
            means.append(row[f"ARI_{label_type}_mean"].values[0])
            stds.append(row[f"ARI_{label_type}_std"].values[0])

        offset = (mi - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      capsize=4, label=model, color=colors[model],
                      alpha=0.85, error_kw={"linewidth":1.5})

        for bar, m in zip(bars, means):
            if m > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    # Published baselines
    published = {"Seurat": 0.4612, "SEDR_pub": 0.3668,
                 "STAGATE": 0.4944, "TGR-NMF": 0.5286}
    colors_pub = ["#e74c3c", "#e67e22", "#9b59b6", "#27ae60"]
    if label_type == "fine":
        for (name, val), col in zip(published.items(), colors_pub):
            ax.axhline(val, color=col, linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"{name}={val:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(list(conditions_display.values()), fontsize=9)
    ax.set_ylabel("ARI")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    # Change it to set different limits per subplot:
    if label_type == "fine":
        ax.set_ylim(0, 0.65)
    else:
        ax.set_ylim(0, 0.40)  # coarse ARI max is ~0.28, no need to go to 0.65

plt.suptitle("ARI Comparison: KMeans vs SEDR across Feature Conditions\n"
             "Error bars = ±1 std over 5 seeds", fontsize=11)
plt.tight_layout()
out1 = os.path.join(FIG_DIR, "results_barplot.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: results_barplot.png")

# ── Figure 2: Spatial cluster maps ───────────────────────────────
print("\n[2] Plotting spatial cluster maps...")

conditions = ["gene_only", "image_only", "concat_fused", "gated_fused"]
label_types = ["k20", "k4"]

for lt in label_types:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    n_clusters = 20 if lt == "k20" else 4
    cmap = plt.cm.get_cmap("tab20" if n_clusters > 10 else "Set1", n_clusters)

    for ci, cond in enumerate(conditions):
        cluster_path = os.path.join(RESULT_DIR, f"clusters_{cond}_{lt}.npy")
        if not os.path.exists(cluster_path):
            axes[ci].text(0.5, 0.5, "No data", ha="center", va="center",
                          transform=axes[ci].transAxes)
            axes[ci].set_title(conditions_display.get(cond, cond))
            continue

        pred = np.load(cluster_path)
        sc_plot = axes[ci].scatter(
            xy[:, 0], xy[:, 1], c=pred, cmap=cmap,
            s=6, vmin=0, vmax=n_clusters-1, alpha=0.9
        )
        axes[ci].set_title(conditions_display.get(cond, cond), fontsize=11)
        axes[ci].set_aspect("equal")
        axes[ci].axis("off")

    plt.suptitle(f"Predicted Spatial Clusters ({lt}) — SEDR", fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, f"spatial_clusters_{lt}.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: spatial_clusters_{lt}.png")

# ── Figure 3: Gold standard label map ────────────────────────────
print("\n[3] Plotting gold standard labels...")

# Fine 20-class map — matches ARI evaluation
fig, ax = plt.subplots(figsize=(8, 7))
sc_plot = ax.scatter(xy[:,0], xy[:,1], c=fine_ints, cmap=cmap20,
                     s=6, vmin=0, vmax=len(unique_fine)-1, alpha=0.9)
patches = [mpatches.Patch(color=cmap20(i), label=l)
           for i, l in enumerate(unique_fine)]
ax.legend(handles=patches, fontsize=7, loc="upper right",
          ncol=2, bbox_to_anchor=(1.35, 1.0))
ax.set_title("Gold Standard: fine_annot_type (Xu et al., 20 domains)", fontsize=12)
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()
out3 = os.path.join(FIG_DIR, "gold_standard_map.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: gold_standard_map.png")

# ── Figure 4: Gated clusters vs Gold standard comparison ──────────
print("\n[4] Plotting gated vs gold standard comparison...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Panel 1: gold standard coarse (4 class) — visual reference
unique_labels = sorted(np.unique(label_coarse))
label_to_int  = {l: i for i, l in enumerate(unique_labels)}
label_ints    = np.array([label_to_int[l] for l in label_coarse])
cmap4 = plt.cm.get_cmap("Set2", len(unique_labels))

axes[0].scatter(xy[:,0], xy[:,1], c=label_ints, cmap=cmap4,
                s=6, vmin=0, vmax=len(unique_labels)-1, alpha=0.9)
patches = [mpatches.Patch(color=cmap4(i), label=l)
           for i, l in enumerate(unique_labels)]
axes[0].legend(handles=patches, fontsize=9, loc="upper right")
axes[0].set_title("Gold Standard\n(coarse, 4 classes)", fontsize=12)
axes[0].set_aspect("equal")
axes[0].axis("off")

# Panel 2: gold standard fine (20 class) — matches ARI
axes[1].scatter(xy[:,0], xy[:,1], c=fine_ints, cmap=cmap20,
                s=6, vmin=0, vmax=len(unique_fine)-1, alpha=0.9)
axes[1].set_title("Gold Standard\n(fine, 20 domains)", fontsize=12)
axes[1].set_aspect("equal")
axes[1].axis("off")

# Panel 3: gated SEDR k=20 — matches ARI evaluation
cluster_path = os.path.join(RESULT_DIR, "clusters_gated_fused_k20.npy")
pred = np.load(cluster_path)
axes[2].scatter(xy[:,0], xy[:,1], c=pred,
                cmap=plt.cm.get_cmap("tab20", 20),
                s=6, vmin=0, vmax=19, alpha=0.9)
axes[2].set_title(f"Gated Fusion SEDR\n(k=20, ARI=0.4951)", fontsize=12)
axes[2].set_aspect("equal")
axes[2].axis("off")

plt.suptitle("Gold Standard vs Gated Fusion Prediction", fontsize=13)
plt.tight_layout()
out4 = os.path.join(FIG_DIR, "comparison_map.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: comparison_map.png")

# ── Figure 5: UMAP of SEDR embeddings ────────────────────────────
print("\n[5] Plotting UMAP of SEDR embeddings...")
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for ci, cond in enumerate(conditions):
    emb_path = os.path.join(RESULT_DIR, f"embeddings_{cond}.npy")
    if not os.path.exists(emb_path):
        axes[ci].text(0.5, 0.5, "No data", ha="center", va="center",
                      transform=axes[ci].transAxes)
        continue
    emb = np.load(emb_path)

    # t-SNE (faster than UMAP, no extra dependency)
    tsne  = TSNE(n_components=2, random_state=42, perplexity=30)
    proj  = tsne.fit_transform(emb)

    axes[ci].scatter(proj[:,0], proj[:,1], c=fine_ints,
                    cmap=cmap20, s=4, vmin=0, vmax=len(unique_fine)-1,
                    alpha=0.7)
    axes[ci].set_title(conditions_display.get(cond, cond), fontsize=11)
    axes[ci].axis("off")

plt.suptitle("t-SNE of SEDR Embeddings — colored by gold standard annot_type",
             fontsize=13)

patches = [mpatches.Patch(color=cmap20(i), label=l)
           for i, l in enumerate(unique_fine)]
fig.legend(handles=patches, fontsize=6, loc="lower center",
           ncol=5, bbox_to_anchor=(0.5, -0.15))
plt.subplots_adjust(bottom=0.25)

plt.tight_layout()
out5 = os.path.join(FIG_DIR, "tsne_embeddings.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: tsne_embeddings.png")

print(f"\nAll figures saved to: {FIG_DIR}")