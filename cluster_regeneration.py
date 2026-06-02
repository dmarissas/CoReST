# Regenerate per-spot cluster assignments from the saved SEDR embeddings, and
# report ARI for every (clustering method x raw/refined) combination.
#
# train_sedr.py saves only embeddings_<cond>.npy + ARI scores; this script turns
# those embeddings into the clusters_<cond>_{k20,k4}.npy files that visualize.py
# reads, AND evaluates the shared eval upgrades (GMM clustering + spatial label
# refinement) on the already-trained embeddings — no retraining needed.
#
# Refinement and clustering are applied IDENTICALLY to every condition, so the
# gene-vs-fusion comparison stays fair.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from utils_func import cluster_latent, refine_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PROC_DIR   = os.path.join(BASE_DIR, "processed")

SEED = 42
N_REFINE_NEIGH = 6   # match the SEDR spatial graph k=6

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels_fine   = labels.loc[barcodes, "fine_annot_type"].values
labels_coarse = labels.loc[barcodes, "annot_type"].values
xy = coords.loc[barcodes, ["x", "y"]].values

conditions = {
    "gene_only":       "embeddings_gene_only.npy",
    "image_only":      "embeddings_image_only.npy",
    "concat_fused":    "embeddings_concat_fused.npy",
    "gated_fused":     "embeddings_gated_fused.npy",
    "gene_imagegraph": "embeddings_gene_imagegraph.npy",   # image-gated graph (the win)
    # end-to-end gate (Mechanism 2), present once train_sedr.py has produced it
    "gated_fused_e2e": "embeddings_gated_fused_e2e.npy",
}

resolutions = [("k20", 20, labels_fine), ("k4", 4, labels_coarse)]
methods     = ["kmeans", "gmm"]

rows = []
for cond, emb_file in conditions.items():
    emb_path = os.path.join(RESULT_DIR, emb_file)
    if not os.path.exists(emb_path):
        print(f"Skipping {cond} (missing {emb_file})")
        continue

    emb = np.load(emb_path)

    for tag, k, gold in resolutions:
        for method in methods:
            pred = cluster_latent(emb, k, SEED, method=method)
            pred_ref = refine_labels(pred, xy, n_neigh=N_REFINE_NEIGH, n_iter=1)

            ari_raw = adjusted_rand_score(gold, pred)
            ari_ref = adjusted_rand_score(gold, pred_ref)

            # Keep the legacy filenames (raw KMeans) that visualize.py expects,
            # and additionally save the refined assignments.
            if method == "kmeans":
                np.save(os.path.join(RESULT_DIR, f"clusters_{cond}_{tag}.npy"), pred)
                np.save(os.path.join(RESULT_DIR, f"clusters_{cond}_{tag}_refined.npy"), pred_ref)

            rows.append({
                "condition": cond, "resolution": tag, "k": k,
                "method": method,
                "ARI_raw": round(ari_raw, 4),
                "ARI_refined": round(ari_ref, 4),
                "delta_refine": round(ari_ref - ari_raw, 4),
            })
            print(f"{cond:<16} {tag:<3} {method:<6}  "
                  f"raw={ari_raw:.4f}  refined={ari_ref:.4f}  "
                  f"(+{ari_ref - ari_raw:.4f})")

df = pd.DataFrame(rows)
out = os.path.join(RESULT_DIR, "cluster_regeneration_ari.csv")
df.to_csv(out, index=False)
print(f"\nSaved per-condition table: {out}")

# Compact pivot: fine-resolution ARI per condition x method x raw/refined
print("\nFine (k=20) ARI summary:")
fine = df[df["resolution"] == "k20"]
print(fine[["condition", "method", "ARI_raw", "ARI_refined", "delta_refine"]]
      .to_string(index=False))
