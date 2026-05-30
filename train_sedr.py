# Train SEDR for all 4 feature conditions x 2 cluster resolutions.
# Runs 5 seeds per condition to report mean ± std ARI.
#
# Conditions:
#   1. gene_only    — 200d  (published SEDR baseline)
#   2. image_only   — 256d
#   3. concat_fused — 456d  (naive z-score concatenation)
#   4. gated_fused  — 456d  (adaptive gated fusion — novel)
#
# Also runs KMeans-only baseline (no GNN) per condition
# to isolate GNN contribution from feature quality contribution.
#
# Output: results/ari_results.csv
#         results/embeddings_<condition>.npy

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
SEEDS    = [42, 123, 456, 789, 1234]
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
from SEDR_model import Sedr
from graph_func import graph_construction
from utils_func import fix_seed

PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4: SEDR Training — All Conditions")
print("=" * 60)
print(f"Device : {DEVICE}")
print(f"Seeds  : {SEEDS}")

# ── Load data ──────────────────────────────────────────────────────
print("\n[1] Loading aligned data...")
barcodes = pd.read_csv(
    os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(
    os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(
    os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")

labels_fine   = labels.loc[barcodes, "fine_annot_type"].values
labels_coarse = labels.loc[barcodes, "annot_type"].values

features = {
    "gene_only":   np.load(os.path.join(PROC_DIR, "gene_only.npy")),
    "image_only":  np.load(os.path.join(PROC_DIR, "image_only.npy")),
    "concat_fused":np.load(os.path.join(PROC_DIR, "concat_fused.npy")),
    "gated_fused": np.load(os.path.join(PROC_DIR, "gated_fused.npy")),
}

print(f"    Barcodes: {len(barcodes)}")
for k, v in features.items():
    print(f"    {k:<14}: {v.shape}")
print(f"    Fine labels  : {len(np.unique(labels_fine))} classes")
print(f"    Coarse labels: {len(np.unique(labels_coarse))} classes")

# ── Build spatial graph (shared across all conditions) ────────────
print("\n[2] Building spatial graph (k=6 KNN)...")
adata_g = sc.AnnData(X=features["gene_only"])
adata_g.obsm["spatial"] = coords.loc[barcodes, ["x","y"]].values
graph_dict = graph_construction(adata_g, n=6, mode="KNN")
print("    Graph built")

# ── KMeans baseline (no GNN) ──────────────────────────────────────
def run_kmeans(feats, n_clusters, seeds):
    """Run KMeans n_seeds times, return mean ± std ARI."""
    aris = []
    for s in seeds:
        km = KMeans(n_clusters=n_clusters, n_init=n_clusters*2, random_state=s)
        pred = km.fit_predict(feats)
        aris.append(pred)
    return aris

# ── SEDR training function ────────────────────────────────────────
def run_sedr_once(feats, graph_dict, n_clusters, seed):
    fix_seed(seed)
    sedr = Sedr(X=feats, graph_dict=graph_dict, mode="clustering", device=DEVICE)
    sedr.model.dec_cluster_n = n_clusters
    sedr.model.cluster_layer = torch.nn.Parameter(
        torch.Tensor(n_clusters, sedr.model.latent_dim).to(DEVICE)
    )
    torch.nn.init.xavier_normal_(sedr.model.cluster_layer.data)
    sedr.train_without_dec(epochs=200, lr=0.01, decay=0.01)
    sedr.train_with_dec(epochs=200)
    latent_z, q, feat_x, gnn_z = sedr.process()
    km = KMeans(n_clusters=n_clusters, n_init=n_clusters*2, random_state=seed)
    pred = km.fit_predict(latent_z)
    return pred, latent_z

# ── Run all conditions ────────────────────────────────────────────
N_FINE   = len(np.unique(labels_fine))    # 20
N_COARSE = len(np.unique(labels_coarse))  # 4

all_results = []

for cond_name, feats in features.items():
    print(f"\n{'='*60}")
    print(f"Condition: {cond_name}  dim={feats.shape[1]}")
    print(f"{'='*60}")

    # ── KMeans baseline ──────────────────────────────────────────
    print(f"  [KMeans baseline]")
    km_fine_aris, km_coarse_aris = [], []
    for s in SEEDS:
        km = KMeans(n_clusters=N_FINE,   n_init=N_FINE*2,   random_state=s)
        km_fine_aris.append(adjusted_rand_score(labels_fine,   km.fit_predict(feats)))
        km = KMeans(n_clusters=N_COARSE, n_init=N_COARSE*2, random_state=s)
        km_coarse_aris.append(adjusted_rand_score(labels_coarse, km.fit_predict(feats)))

    km_fine_mean   = np.mean(km_fine_aris)
    km_fine_std    = np.std(km_fine_aris)
    km_coarse_mean = np.mean(km_coarse_aris)
    km_coarse_std  = np.std(km_coarse_aris)
    print(f"    KMeans fine   ARI: {km_fine_mean:.4f} ± {km_fine_std:.4f}")
    print(f"    KMeans coarse ARI: {km_coarse_mean:.4f} ± {km_coarse_std:.4f}")

    all_results.append({
        "condition":        f"{cond_name}_kmeans",
        "model":            "KMeans",
        "input_dim":        feats.shape[1],
        "ARI_fine_mean":    round(km_fine_mean,  4),
        "ARI_fine_std":     round(km_fine_std,   4),
        "ARI_coarse_mean":  round(km_coarse_mean,4),
        "ARI_coarse_std":   round(km_coarse_std, 4),
    })

    # ── SEDR fine (20 clusters) ───────────────────────────────────
    print(f"\n  [SEDR fine, k={N_FINE}]")
    sedr_fine_aris   = []
    best_fine_ari    = -1
    best_fine_emb    = None

    for s in SEEDS:
        print(f"    seed={s} ...", end=" ", flush=True)
        pred, emb = run_sedr_once(feats, graph_dict, N_FINE, s)
        ari = adjusted_rand_score(labels_fine, pred)
        sedr_fine_aris.append(ari)
        print(f"ARI={ari:.4f}")
        if ari > best_fine_ari:
            best_fine_ari = ari
            best_fine_emb = emb

    sedr_fine_mean = np.mean(sedr_fine_aris)
    sedr_fine_std  = np.std(sedr_fine_aris)
    print(f"    SEDR fine ARI: {sedr_fine_mean:.4f} ± {sedr_fine_std:.4f}  "
          f"(best={best_fine_ari:.4f})")

    np.save(os.path.join(RESULT_DIR, f"embeddings_{cond_name}.npy"), best_fine_emb)

    # ── SEDR coarse (4 clusters) ──────────────────────────────────
    print(f"\n  [SEDR coarse, k={N_COARSE}]")
    sedr_coarse_aris = []

    for s in SEEDS:
        print(f"    seed={s} ...", end=" ", flush=True)
        pred, _ = run_sedr_once(feats, graph_dict, N_COARSE, s)
        ari = adjusted_rand_score(labels_coarse, pred)
        sedr_coarse_aris.append(ari)
        print(f"ARI={ari:.4f}")

    sedr_coarse_mean = np.mean(sedr_coarse_aris)
    sedr_coarse_std  = np.std(sedr_coarse_aris)
    print(f"    SEDR coarse ARI: {sedr_coarse_mean:.4f} ± {sedr_coarse_std:.4f}")

    all_results.append({
        "condition":        cond_name,
        "model":            "SEDR",
        "input_dim":        feats.shape[1],
        "ARI_fine_mean":    round(sedr_fine_mean,   4),
        "ARI_fine_std":     round(sedr_fine_std,    4),
        "ARI_coarse_mean":  round(sedr_coarse_mean, 4),
        "ARI_coarse_std":   round(sedr_coarse_std,  4),
    })

# ── Results table ─────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
df = pd.DataFrame(all_results)

# Format for display
df["ARI_fine"]   = df.apply(lambda r: f"{r.ARI_fine_mean:.4f} ± {r.ARI_fine_std:.4f}",   axis=1)
df["ARI_coarse"] = df.apply(lambda r: f"{r.ARI_coarse_mean:.4f} ± {r.ARI_coarse_std:.4f}", axis=1)
print(df[["condition","model","input_dim","ARI_fine","ARI_coarse"]].to_string(index=False))

# Published baselines for context
print(f"\nPublished baselines (fine ARI, k=20):")
baselines = [
    ("Seurat",  0.4612), ("SEDR",    0.3668),
    ("STAGATE", 0.4944), ("TGR-NMF", 0.5286),
]
for name, val in baselines:
    print(f"  {name:<10}: {val:.4f}")

# Save
df.to_csv(os.path.join(RESULT_DIR, "ari_results.csv"), index=False)
print(f"\nSaved: results/ari_results.csv")