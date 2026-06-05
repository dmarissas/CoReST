# Figures for the Part 1 (fusion) ABLATION studies that produce CSVs but no plots.
#
# Reads ONLY saved CSVs (no retraining, no GPU):
#   results/graph_experiment_ari.csv   (from experiment_graph.py)
#   results/image_feature_study.csv    (from image_feature_study.py)
#
# Figures written to data/.../figures/:
#   graph_experiment.png    — image-gated graph vs density-matched & shuffled controls
#   image_feature_study.png — image-only ARI by representation vs the gene-only ceiling
#
# Style matches visualize.py (Agg backend, dpi=150, labelled bars).

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GENE_GMM_REFINED = 0.5746   # gene-only ceiling (GMM, refined) for the image study

print("=" * 60)
print("Part 1 ablation figures — graph experiment + image study")
print("=" * 60)

# ── Figure 1: graph experiment (modes + density-matched + shuffle) ──
gpath = os.path.join(RESULT_DIR, "graph_experiment_ari.csv")
if not os.path.exists(gpath):
    print(f"\n[1] SKIP graph_experiment.png — missing {gpath}\n    (run experiment_graph.py first)")
else:
    print("\n[1] graph_experiment.png")
    g = pd.read_csv(gpath).set_index("graph_mode")

    # display order + role-based colour (baseline / density control / image / shuffle)
    order = ["spatial", "spatial_k3", "spatial_k4", "spatial_k5",
             "blend_a0.75", "intersect", "intersect_SHUFFLE"]
    order = [m for m in order if m in g.index]
    disp = {"spatial": "spatial k6\n(gene baseline)", "spatial_k3": "spatial k3\n(density ctrl)",
            "spatial_k4": "spatial k4\n(density ctrl)", "spatial_k5": "spatial k5\n(density ctrl)",
            "blend_a0.75": "blend α=0.75\n(image)", "intersect": "intersect\n(image-gated)",
            "intersect_SHUFFLE": "intersect\nSHUFFLED"}
    role_color = {"spatial": "#34495e", "spatial_k3": "#aed6f1", "spatial_k4": "#aed6f1",
                  "spatial_k5": "#aed6f1", "blend_a0.75": "#52be80", "intersect": "#1e8449",
                  "intersect_SHUFFLE": "#e74c3c"}

    anchor = float(g.loc["spatial", "ARI_refined_mean"]) if "spatial" in g.index else None
    x = np.arange(len(order))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A — refined ARI with avg-degree annotated (the density-matching axis)
    meansA = [g.loc[m, "ARI_refined_mean"] for m in order]
    stdsA  = [g.loc[m, "ARI_refined_std"] for m in order]
    colsA  = [role_color[m] for m in order]
    barsA = axA.bar(x, meansA, yerr=stdsA, capsize=4, color=colsA, alpha=0.92,
                    error_kw={"linewidth": 1.3})
    if anchor is not None:
        axA.axhline(anchor, color="#34495e", ls="--", lw=1.4,
                    label=f"gene / spatial-k6 anchor = {anchor:.4f}")
    for m, b, sd in zip(order, barsA, stdsA):
        axA.text(b.get_x() + b.get_width() / 2, b.get_height() + sd + 0.010,
                 f"{g.loc[m,'ARI_refined_mean']:.3f}", ha="center", va="bottom", fontsize=8)
        axA.text(b.get_x() + b.get_width() / 2, 0.012,
                 f"deg {g.loc[m,'avg_degree']:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color="white", fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels([disp[m] for m in order], fontsize=8)
    axA.set_ylabel("Fine ARI (k=20, KMeans, refined)")
    axA.set_ylim(0, max(meansA) + 0.14)
    axA.set_title("Image-gated graph vs density-matched controls\n"
                  "intersect (deg ~4.5) beats the same-density spatial-k3 → it's the image, not sparsity")
    axA.legend(fontsize=8, loc="upper left"); axA.grid(axis="y", alpha=0.3)

    # Panel B — silhouette (the falsification: real image vs shuffled)
    silB = [g.loc[m, "silhouette_mean"] for m in order]
    barsB = axB.bar(x, silB, color=colsA, alpha=0.92)
    for m, b in zip(order, barsB):
        axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                 f"{g.loc[m,'silhouette_mean']:.3f}", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels([disp[m] for m in order], fontsize=8)
    axB.set_ylabel("Silhouette (cluster quality, label-free)")
    axB.set_ylim(0, max(silB) + 0.04)
    axB.set_title("Falsification control: real image vs shuffled\n"
                  "shuffling the image collapses cluster quality (intersect 0.288 → 0.237)")
    axB.grid(axis="y", alpha=0.3)
    if "intersect" in g.index and "intersect_SHUFFLE" in g.index:
        xi, xs = order.index("intersect"), order.index("intersect_SHUFFLE")
        axB.annotate("", xy=(xs, g.loc["intersect_SHUFFLE", "silhouette_mean"]),
                     xytext=(xi, g.loc["intersect", "silhouette_mean"]),
                     arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.6))

    plt.suptitle("Graph-level fusion ablation (gene features fixed; only the SEDR graph changes) — "
                 "mean ± std over 5 seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "graph_experiment.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: graph_experiment.png")

# ── Figure 2: image-feature study (representation sweep) ────────────
ipath = os.path.join(RESULT_DIR, "image_feature_study.csv")
if not os.path.exists(ipath):
    print(f"\n[2] SKIP image_feature_study.png — missing {ipath}\n    (run image_feature_study.py first)")
else:
    print("\n[2] image_feature_study.png")
    im = pd.read_csv(ipath)
    disp = {"all_pca256(base)": "all scales\nPCA 256", "all_pca512": "all scales\nPCA 512",
            "all_pca128": "all scales\nPCA 128", "scale1_1x_pca128": "scale 1×\n(cellular)",
            "scale2_2x_pca128": "scale 2×", "scale3_3x_pca128": "scale 3×\n(context)"}
    names = im["representation"].tolist()
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width / 2, im["kmeans_refined"], width, label="KMeans (refined)",
                color="#95a5a6", alpha=0.92)
    b2 = ax.bar(x + width / 2, im["gmm_refined"], width, label="GMM (refined)",
                color="#2980b9", alpha=0.92)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(GENE_GMM_REFINED, color="#e74c3c", ls="--", lw=1.8,
               label=f"gene-only ceiling (GMM, refined) = {GENE_GMM_REFINED:.4f}")
    ax.set_xticks(x); ax.set_xticklabels([disp.get(n, n) for n in names], fontsize=8.5)
    ax.set_ylabel("Image-only fine ARI (k=20, refined)")
    ax.set_ylim(0, 0.62)
    ax.set_title("Is the bottleneck the image FEATURES or the MODALITY?\n"
                 "Every UNI representation caps at ~0.30–0.34 — a ~0.24 gap no extraction choice closes")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    # annotate the gap
    ax.annotate("", xy=(len(names) - 1, GENE_GMM_REFINED),
                xytext=(len(names) - 1, im["gmm_refined"].iloc[-1]),
                arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.3))
    ax.text(len(names) - 1.25, (GENE_GMM_REFINED + im["gmm_refined"].iloc[-1]) / 2,
            "~0.24 gap", ha="right", va="center", fontsize=9, color="#7f8c8d", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "image_feature_study.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: image_feature_study.png")

print(f"\nAblation figures saved to: {FIG_DIR}")
