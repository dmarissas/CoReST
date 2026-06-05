# Cross-tissue GENERALIZATION figure: does the consensus recipe transfer?
#
# Side-by-side "run once vs run-5+consensus" clouds for each dataset that has a
# consensus_robustness_pools.csv (default: HBRC + DLPFC 151673). Each panel's
# subtitle is DATA-DRIVEN (lift, variance ratio tighter/WIDER, beats/ties best-of-
# pool), so the figure honestly shows what generalizes (lift + determinism) and
# what doesn't (variance behaviour).
#
# Recomputes single-seed ARIs from each dataset's cached embeddings (no training).
# Run:  python visualize_generalization.py
# Output: data/dlpfc_151673/figures/generalization_hbrc_vs_dlpfc.png

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

N_REFINE_NEIGH = 6
SEEDS_ALL = [42, 123, 456, 789, 1234, 7, 88, 314, 2024, 51966,
             11, 222, 333, 444, 555, 1212, 2323, 3434, 4545, 5656]
DATASETS = [
    ("HBRC (breast cancer)",  "data/block_a_section1_v110", "fine_annot_type"),
    ("DLPFC 151673 (brain)",  "data/dlpfc_151673",          "layer_guess"),
]


def load_one(base_dir, label_col):
    proc = os.path.join(CODE_DIR, base_dir, "processed")
    res  = os.path.join(CODE_DIR, base_dir, "results")
    pools = pd.read_csv(os.path.join(res, "consensus_robustness_pools.csv"))
    barcodes = pd.read_csv(os.path.join(proc, "barcodes_final.csv"))["barcode"].tolist()
    gold = pd.read_csv(os.path.join(proc, "labels_final.csv"), index_col="barcode").loc[barcodes, label_col].values
    xy   = pd.read_csv(os.path.join(proc, "coords_final.csv"), index_col="barcode").loc[barcodes, ["x", "y"]].values
    K = len(np.unique(gold))
    singles = []
    for s in SEEDS_ALL:
        z = np.load(os.path.join(res, f"embeddings_consensus_gene_seed{s}_fine.npy"))
        singles.append(adjusted_rand_score(
            gold, refine_labels(cluster_latent(z, K, s, method="gmm"), xy, N_REFINE_NEIGH)))
    return {"singles": np.array(singles), "cons": pools["consensus"].values,
            "m_best": pools["margin_vs_best"].values, "K": K}


def main():
    data = []
    for name, bd, lc in DATASETS:
        if not os.path.exists(os.path.join(CODE_DIR, bd, "results", "consensus_robustness_pools.csv")):
            print(f"skip {name}: no consensus_robustness_pools.csv (run consensus_robustness.py for it)")
            continue
        print(f"loading {name} ...")
        data.append((name, load_one(bd, lc)))
    if not data:
        print("no datasets with robustness results found")
        sys.exit(1)

    fig, axes = plt.subplots(1, len(data), figsize=(7.5 * len(data), 6.5))
    if len(data) == 1:
        axes = [axes]
    rng = np.random.RandomState(0)
    for ax, (name, d) in zip(axes, data):
        s, c = d["singles"], d["cons"]
        sm, ss, cm, cc = s.mean(), s.std(), c.mean(), c.std()
        vr = ss / cc if cc > 1e-9 else float("inf")
        N = len(c)
        nbest = int((d["m_best"] >= -1e-9).sum())
        beats = nbest > 0.6 * N
        for xc, vals, color, lab in [(0, s, "#95a5a6", "single seed"),
                                     (1, c, "#27ae60", "5-seed consensus")]:
            jit = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full(len(vals), xc) + jit, vals, s=40, color=color, alpha=0.7,
                       edgecolor="white", linewidth=0.4, zorder=3, label=lab)
            ax.errorbar(xc, vals.mean(), yerr=vals.std(), fmt="_", color="black", markersize=44,
                        elinewidth=2, capsize=10, capthick=2, zorder=4)
        ax.annotate("", xy=(1, cm), xytext=(0, sm), arrowprops=dict(arrowstyle="->", color="#1e8449", lw=2))
        ax.text(0.5, (sm + cm) / 2 - 0.012, f"+{cm - sm:.3f}", ha="center",
                color="#1e8449", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1])
        ax.set_xticklabels(["single\nseed", "5-seed\nconsensus"])
        ax.set_ylabel(f"ARI (k={d['K']}, refined)")
        ax.set_title(f"{name}\nlift +{cm - sm:.3f}   ·   {'tighter' if vr > 1.05 else 'WIDER'} {vr:.1f}×   ·   "
                     f"{'BEATS' if beats else 'ties'} best-of-pool ({nbest}/{N})", fontsize=10)
        ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("Does the consensus recipe generalize? — same untuned pipeline, two tissues\n"
                 "lift + determinism replicate on both; variance behaviour is dataset-dependent",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(CODE_DIR, "data", "dlpfc_151673", "figures", "generalization_hbrc_vs_dlpfc.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
