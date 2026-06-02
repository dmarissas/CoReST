# Diagnostic: how much tissue-domain signal do the UNI histology features carry,
# and are we discarding it in the PCA / scale-aggregation?
#
# Runs SEDR on IMAGE-ONLY features (spatial graph) for several image
# representations built from the raw 3072d multi-scale UNI embeddings:
#   - all scales, PCA to {128, 256(baseline), 512}
#   - each scale alone (1x cellular, 2x, 3x tissue-context), PCA 128
# and reports fine-ARI (KMeans + GMM, refined) vs the gene-only ceiling.
#
# If some image representation scores much higher than the current ~0.30, the
# bottleneck was the representation (worth re-fusing). If all cap near ~0.30, the
# honest conclusion stands: UNI histology lacks domain signal beyond genes here.
#
# Requires: image_features_raw3072.npy (re-run image_features.py once to create).
# Output: results/image_feature_study.csv

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from SEDR_model import Sedr
from graph_func import graph_construction
from utils_func import fix_seed, cluster_latent, refine_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS  = [42, 123, 456, 789, 1234]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_FINE = 20
N_REFINE_NEIGH = 6
GENE_GMM_REFINED = 0.5746   # gene-only ceiling (GMM, refined) for reference

print("=" * 60)
print("IMAGE FEATURE STUDY — how much domain signal in UNI?")
print("=" * 60)

raw_path = os.path.join(PROC_DIR, "image_features_raw3072.npy")
if not os.path.exists(raw_path):
    print("MISSING image_features_raw3072.npy — re-run image_features.py once to create it.")
    sys.exit(1)

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels_fine = labels.loc[barcodes, "fine_annot_type"].values
xy = coords.loc[barcodes, ["x", "y"]].values

# raw3072 rows are in image_features_256d.csv's barcode order; align to barcodes_final
img_order = pd.read_csv(os.path.join(PROC_DIR, "image_features_256d.csv"),
                        index_col="barcode").index
raw = np.load(raw_path)                                  # (n_in_tissue, 3072) pos order
raw_df = pd.DataFrame(raw, index=img_order)
raw_aligned = raw_df.loc[barcodes].values.astype(np.float32)
print(f"Raw UNI aligned: {raw_aligned.shape}")

# spatial graph (same as the main pipeline)
adata_g = sc.AnnData(X=raw_aligned[:, :1])
adata_g.obsm["spatial"] = xy
graph_spatial = graph_construction(adata_g, n=6, mode="KNN")


def make_rep(cols, pca_dims):
    """raw[:, cols] -> L2 -> PCA(pca_dims) -> L2 -> z-score (matches the pipeline)."""
    x = raw_aligned[:, cols].astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    if pca_dims and pca_dims < x.shape[1]:
        x = PCA(n_components=pca_dims, random_state=42).fit_transform(x)
        x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    return StandardScaler().fit_transform(x).astype(np.float32)


REPS = [
    ("all_pca256(base)", slice(0, 3072), 256),
    ("all_pca512",       slice(0, 3072), 512),
    ("all_pca128",       slice(0, 3072), 128),
    ("scale1_1x_pca128", slice(0, 1024), 128),
    ("scale2_2x_pca128", slice(1024, 2048), 128),
    ("scale3_3x_pca128", slice(2048, 3072), 128),
]


def run_sedr(feats, seed):
    fix_seed(seed)
    sedr = Sedr(X=feats, graph_dict=graph_spatial, mode="clustering", device=DEVICE)
    sedr.model.dec_cluster_n = N_FINE
    sedr.model.cluster_layer = torch.nn.Parameter(
        torch.Tensor(N_FINE, sedr.model.latent_dim).to(DEVICE))
    torch.nn.init.xavier_normal_(sedr.model.cluster_layer.data)
    sedr.train_without_dec(epochs=200, lr=0.01, decay=0.01)
    sedr.train_with_dec(epochs=200)
    z, _, _, _ = sedr.process()
    return z


rows = []
for name, cols, pca_dims in REPS:
    feats = make_rep(cols, pca_dims)
    print(f"\n{'='*60}\n{name}  (dim={feats.shape[1]})\n{'='*60}")
    res = {m: {"raw": [], "refined": []} for m in ["kmeans", "gmm"]}
    for s in SEEDS:
        z = run_sedr(feats, s)
        for m in ["kmeans", "gmm"]:
            pred = cluster_latent(z, N_FINE, s, method=m)
            predr = refine_labels(pred, xy, n_neigh=N_REFINE_NEIGH, n_iter=1)
            res[m]["raw"].append(adjusted_rand_score(labels_fine, pred))
            res[m]["refined"].append(adjusted_rand_score(labels_fine, predr))
        print(f"  seed={s}  km_ref={res['kmeans']['refined'][-1]:.4f}  "
              f"gmm_ref={res['gmm']['refined'][-1]:.4f}")
    row = {"representation": name, "dim": feats.shape[1]}
    for m in ["kmeans", "gmm"]:
        for r in ["raw", "refined"]:
            row[f"{m}_{r}"] = round(float(np.mean(res[m][r])), 4)
    rows.append(row)
    print(f"  -> kmeans_refined={row['kmeans_refined']:.4f}  gmm_refined={row['gmm_refined']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "image_feature_study.csv"), index=False)
print(f"\n{'='*60}\nIMAGE-ONLY fine ARI by representation "
      f"(gene-only ceiling GMM refined = {GENE_GMM_REFINED})\n{'='*60}")
print(df.to_string(index=False))
best = df.loc[df["gmm_refined"].idxmax()]
print(f"\nStrongest image rep (gmm refined): {best['representation']} = {best['gmm_refined']:.4f}")
print("If this is far above the current ~0.30, the representation was the bottleneck "
      "→ worth re-fusing. If still ~0.30, histology genuinely lacks domain signal here.")
