# Train SEDR across feature/graph conditions x 2 resolutions, 5 seeds each.
#
# Conditions (each = node FEATURES + message-passing GRAPH):
#   1. gene_only        — gene 200d, spatial graph        (published SEDR baseline)
#   2. image_only       — image 256d, spatial graph
#   3. concat_fused     — concat 456d, spatial graph      (naive concat — hurts)
#   4. gated_fused      — gated 128d, spatial graph        (feature-space gate)
#   5. gene_imagegraph  — gene 200d, IMAGE-GATED graph     (graph-level gate — the win)
#
# The graph-level gate (condition 5) keeps a spatial edge only if the two spots
# are ALSO morphologically similar (intersect of spatial-KNN and image-KNN), so
# histology guides smoothing without diluting the gene reconstruction signal.
# A density-matched control (spatial_k3) and a shuffle control live in
# experiment_graph.py; this script produces the headline comparison table.
#
# Every condition is evaluated identically with {KMeans, GMM} x {raw, refined},
# so the gene-vs-fusion comparison stays fair.
#
# Output: results/ari_results.csv      (long format)
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
CLUSTER_METHODS = ["kmeans", "gmm"]   # both reported; KMeans is the comparison column
N_REFINE_NEIGH  = 6                   # spatial label refinement (same for all conditions)
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
from SEDR_model import Sedr
from graph_func import graph_construction, graph_construction_fused
from utils_func import fix_seed, cluster_latent, refine_labels

PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4: SEDR Training — feature/graph conditions")
print("=" * 60)
print(f"Device : {DEVICE}")
print(f"Seeds  : {SEEDS}")

# ── Load aligned data ──────────────────────────────────────────────
print("\n[1] Loading aligned data...")
barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")

labels_fine   = labels.loc[barcodes, "fine_annot_type"].values
labels_coarse = labels.loc[barcodes, "annot_type"].values
xy = coords.loc[barcodes, ["x", "y"]].values

gene_feats   = np.load(os.path.join(PROC_DIR, "gene_only.npy"))
image_feats  = np.load(os.path.join(PROC_DIR, "image_only.npy"))
concat_feats = np.load(os.path.join(PROC_DIR, "concat_fused.npy"))
gated_feats  = np.load(os.path.join(PROC_DIR, "gated_fused.npy"))

print(f"    Barcodes: {len(barcodes)}")
print(f"    Fine labels  : {len(np.unique(labels_fine))} classes")
print(f"    Coarse labels: {len(np.unique(labels_coarse))} classes")

# ── Build graphs ───────────────────────────────────────────────────
print("\n[2] Building graphs...")
adata_g = sc.AnnData(X=gene_feats)
adata_g.obsm["spatial"] = xy
graph_spatial    = graph_construction(adata_g, n=6, mode="KNN")
graph_imagegraph = graph_construction_fused(adata_g, image_feats, n=6, mode_fuse="intersect")
print("    spatial (k=6 KNN) and image-gated (intersect) graphs built")

N_FINE   = len(np.unique(labels_fine))    # 20
N_COARSE = len(np.unique(labels_coarse))  # 4

# (name, features, graph_dict, graph_mode)
CONDITIONS = [
    ("gene_only",       gene_feats,   graph_spatial,    "spatial"),
    ("image_only",      image_feats,  graph_spatial,    "spatial"),
    ("concat_fused",    concat_feats, graph_spatial,    "spatial"),
    ("gated_fused",     gated_feats,  graph_spatial,    "spatial"),
    ("gene_imagegraph", gene_feats,   graph_imagegraph, "intersect"),
]


def run_sedr_once(feats, graph_dict, n_clusters, seed):
    """Train SEDR once; return the latent embedding (clustering done outside)."""
    fix_seed(seed)
    sedr = Sedr(X=feats, graph_dict=graph_dict, mode="clustering", device=DEVICE)
    sedr.model.dec_cluster_n = n_clusters
    sedr.model.cluster_layer = torch.nn.Parameter(
        torch.Tensor(n_clusters, sedr.model.latent_dim).to(DEVICE))
    torch.nn.init.xavier_normal_(sedr.model.cluster_layer.data)
    sedr.train_without_dec(epochs=200, lr=0.01, decay=0.01)
    sedr.train_with_dec(epochs=200)
    latent_z, _, _, _ = sedr.process()
    return latent_z


resolutions = [("fine", N_FINE, labels_fine), ("coarse", N_COARSE, labels_coarse)]
rows = []

for name, feats, gdict, gmode in CONDITIONS:
    print(f"\n{'='*60}\nCondition: {name}  (features dim={feats.shape[1]}, graph={gmode})\n{'='*60}")

    # ── KMeans-on-raw-features baseline (no GNN) — for context ──
    for res_name, k, gold in resolutions:
        aris = [adjusted_rand_score(gold, KMeans(n_clusters=k, n_init=k*2,
                random_state=s).fit_predict(feats)) for s in SEEDS]
        rows.append({"condition": name, "graph_mode": gmode, "model": "KMeans_features",
                     "resolution": res_name, "cluster_method": "kmeans", "refine": "raw",
                     "input_dim": feats.shape[1],
                     "ARI_mean": round(np.mean(aris), 4), "ARI_std": round(np.std(aris), 4),
                     "n_seeds": len(SEEDS)})

    # ── SEDR per resolution ──
    for res_name, k, gold in resolutions:
        print(f"  [SEDR {res_name}, k={k}]")
        per_seed_z = []
        for s in SEEDS:
            print(f"    seed={s} ...", end=" ", flush=True)
            z = run_sedr_once(feats, gdict, k, s)
            per_seed_z.append((s, z))
            quick = adjusted_rand_score(gold, KMeans(n_clusters=k, n_init=k*2,
                    random_state=s).fit_predict(z))
            print(f"(kmeans raw ARI={quick:.4f})")

        # evaluate every (cluster_method x refine) across seeds
        for method in CLUSTER_METHODS:
            raw_aris, ref_aris = [], []
            for s, z in per_seed_z:
                pred  = cluster_latent(z, k, s, method=method)
                predr = refine_labels(pred, xy, n_neigh=N_REFINE_NEIGH, n_iter=1)
                raw_aris.append(adjusted_rand_score(gold, pred))
                ref_aris.append(adjusted_rand_score(gold, predr))
            for refine_tag, aris in [("raw", raw_aris), ("refined", ref_aris)]:
                rows.append({"condition": name, "graph_mode": gmode, "model": "SEDR",
                             "resolution": res_name, "cluster_method": method, "refine": refine_tag,
                             "input_dim": feats.shape[1],
                             "ARI_mean": round(np.mean(aris), 4), "ARI_std": round(np.std(aris), 4),
                             "n_seeds": len(SEEDS)})
            print(f"    {method:<6} fine/coarse raw={np.mean(raw_aris):.4f}±{np.std(raw_aris):.4f}  "
                  f"refined={np.mean(ref_aris):.4f}±{np.std(ref_aris):.4f}")

        # save best-seed embedding (fine, by kmeans raw ARI) for visualization
        if res_name == "fine":
            best_s, best_z, best_ari = None, None, -1
            for s, z in per_seed_z:
                a = adjusted_rand_score(gold, KMeans(n_clusters=k, n_init=k*2,
                    random_state=s).fit_predict(z))
                if a > best_ari:
                    best_ari, best_z, best_s = a, z, s
            np.save(os.path.join(RESULT_DIR, f"embeddings_{name}.npy"), best_z)

# ── Save + summarize ───────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "ari_results.csv"), index=False)

print(f"\n\n{'='*70}\nFINAL RESULTS — fine (k=20), SEDR\n{'='*70}")
fine = df[(df["resolution"] == "fine") & (df["model"] == "SEDR")]
pivot = fine.pivot_table(index=["condition", "graph_mode"],
                         columns=["cluster_method", "refine"], values="ARI_mean")
print(pivot.to_string())

print(f"\nPublished baselines (fine ARI, k=20): "
      f"Seurat 0.4612 | SEDR 0.3668 | STAGATE 0.4944 | TGR-NMF 0.5286")
print(f"\nSaved: results/ari_results.csv  (long format: condition x graph x model x "
      f"resolution x cluster_method x refine)")
