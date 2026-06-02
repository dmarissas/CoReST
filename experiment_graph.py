# Mechanism 1 experiment: image-informed spatial graph (graph-level gate).
#
# Holds the NODE FEATURES fixed at gene_only (200d) and varies only the SEDR
# message-passing graph. Isolates the graph's contribution.
#
# This version adds the two controls needed to make an HONEST causal claim:
#   1. DENSITY-MATCHED control: plain spatial graphs at smaller k (k3/k4/k5,
#      no image). 'intersect' is sparser than spatial-k6; to claim the IMAGE
#      (not just "fewer edges") helps, intersect must beat a spatial graph of
#      similar edge density.
#   2. FALSIFICATION control: same graph built from SHUFFLED image features.
#      If the gain survives shuffling it was not real image signal.
#
# Also logs graph density (avg degree) per mode and does a PAIRED per-seed
# comparison vs the spatial baseline (more rigorous than mean +/- std for a
# small ~0.02 effect).
#
# Honesty: operating point chosen by LABEL-FREE silhouette, never by argmax ARI.
#
# Output: results/graph_experiment_ari.csv (summary)
#         results/graph_experiment_perseed.csv (per-seed long format)

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from SEDR_model import Sedr
from graph_func import graph_construction_fused
from utils_func import fix_seed, cluster_latent, refine_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS   = [42, 123, 456, 789, 1234]
DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
N_FINE  = 20
N_REFINE_NEIGH = 6
CLUSTER_METHOD = "kmeans"
GENE_ONLY_BASELINE = 0.4783

# (label, builder-kwargs, shuffle_image)
GRAPH_MODES = [
    ("spatial",            dict(mode_fuse="spatial"),             False),  # baseline anchor (k6)
    # --- density-matched controls: plain spatial, fewer neighbors, NO image ---
    ("spatial_k3",         dict(mode_fuse="spatial", n=3),        False),
    ("spatial_k4",         dict(mode_fuse="spatial", n=4),        False),
    ("spatial_k5",         dict(mode_fuse="spatial", n=5),        False),
    # --- candidate image-informed graphs ---
    ("blend_a0.75",        dict(mode_fuse="blend", alpha=0.75),   False),
    ("intersect",          dict(mode_fuse="intersect"),           False),
    # --- falsification control: same graph, SHUFFLED image ---
    ("intersect_SHUFFLE",  dict(mode_fuse="intersect"),           True),
]

print("=" * 60)
print("EXPERIMENT: image-informed graph (gene features fixed)")
print("=" * 60)
print(f"Device: {DEVICE}   Seeds: {SEEDS}")

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
labels_fine = labels.loc[barcodes, "fine_annot_type"].values
xy = coords.loc[barcodes, ["x", "y"]].values

gene_feats  = np.load(os.path.join(PROC_DIR, "gene_only.npy"))    # (3798, 200) — FIXED
image_feats = np.load(os.path.join(PROC_DIR, "image_only.npy"))   # (3798, 256) — graph only
print(f"Gene features (fixed): {gene_feats.shape}   Image features: {image_feats.shape}")

_perm = np.random.RandomState(0).permutation(len(image_feats))
image_feats_shuffled = image_feats[_perm]

adata_g = sc.AnnData(X=gene_feats)
adata_g.obsm["spatial"] = xy
N = len(barcodes)


def graph_avg_degree(graph_dict):
    """Avg degree of the message-passing graph (incl. self-loops added by
    preprocess_graph). Comparable across modes for density matching."""
    nnz = graph_dict["adj_norm"].coalesce().values().shape[0]
    return nnz / N


def run_sedr_gene(feats, graph_dict, n_clusters, seed):
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


rows = []
perseed_rows = []
perseed_ref = {}   # mode -> {seed: refined_ari} for the paired test
for mode_label, kwargs, shuffle_image in GRAPH_MODES:
    print(f"\n{'='*60}\nGraph mode: {mode_label}\n{'='*60}")
    img = image_feats_shuffled if shuffle_image else image_feats
    graph_dict = graph_construction_fused(adata_g, img, **kwargs)
    avg_deg = graph_avg_degree(graph_dict)
    print(f"  avg degree (incl. self): {avg_deg:.2f}")

    raw_aris, ref_aris, sils = [], [], []
    perseed_ref[mode_label] = {}
    for s in SEEDS:
        print(f"  seed={s} ...", end=" ", flush=True)
        z = run_sedr_gene(gene_feats, graph_dict, N_FINE, s)
        pred = cluster_latent(z, N_FINE, s, method=CLUSTER_METHOD)
        pred_ref = refine_labels(pred, xy, n_neigh=N_REFINE_NEIGH, n_iter=1)
        ari_raw = adjusted_rand_score(labels_fine, pred)
        ari_ref = adjusted_rand_score(labels_fine, pred_ref)
        sil = silhouette_score(z, pred)
        raw_aris.append(ari_raw); ref_aris.append(ari_ref); sils.append(sil)
        perseed_ref[mode_label][s] = ari_ref
        perseed_rows.append({"graph_mode": mode_label, "seed": s,
                             "ARI_raw": round(ari_raw, 4), "ARI_refined": round(ari_ref, 4),
                             "silhouette": round(sil, 4)})
        print(f"raw={ari_raw:.4f}  refined={ari_ref:.4f}  sil={sil:.4f}")

    rows.append({
        "graph_mode":       mode_label,
        "image_shuffled":   shuffle_image,
        "avg_degree":       round(avg_deg, 2),
        "ARI_raw_mean":     round(np.mean(raw_aris), 4),
        "ARI_raw_std":      round(np.std(raw_aris), 4),
        "ARI_refined_mean": round(np.mean(ref_aris), 4),
        "ARI_refined_std":  round(np.std(ref_aris), 4),
        "silhouette_mean":  round(np.mean(sils), 4),
        "n_seeds":          len(SEEDS),
    })
    print(f"  -> refined {np.mean(ref_aris):.4f}±{np.std(ref_aris):.4f}  "
          f"sil {np.mean(sils):.4f}  deg {avg_deg:.2f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "graph_experiment_ari.csv"), index=False)
pd.DataFrame(perseed_rows).to_csv(
    os.path.join(RESULT_DIR, "graph_experiment_perseed.csv"), index=False)

print(f"\n\n{'='*70}\nSUMMARY (gene features fixed; baseline spatial-k6 = {GENE_ONLY_BASELINE} raw)\n{'='*70}")
print(df.to_string(index=False))

# ── Paired per-seed comparison vs spatial-k6 (same seeds) ──
print(f"\n{'='*70}\nPAIRED per-seed comparison of REFINED ARI vs spatial-k6\n{'='*70}")
base = perseed_ref["spatial"]
print(f"{'mode':<20}{'mean_diff':>10}{'#better/5':>11}{'avg_deg':>9}")
for mode_label, _, _ in GRAPH_MODES:
    if mode_label == "spatial":
        continue
    diffs = [perseed_ref[mode_label][s] - base[s] for s in SEEDS]
    n_better = sum(d > 0 for d in diffs)
    deg = df[df["graph_mode"] == mode_label]["avg_degree"].values[0]
    print(f"{mode_label:<20}{np.mean(diffs):>+10.4f}{n_better:>9}/5{deg:>9.2f}")

# ── Honest interpretation hints ──
print(f"\n{'='*70}\nHOW TO READ THIS\n{'='*70}")
print("1. DENSITY: compare 'intersect' avg_degree to the spatial_kN with the")
print("   closest degree. If intersect's refined ARI beats THAT spatial_kN, the")
print("   IMAGE edge-selection helps beyond just having fewer edges.")
print("2. FALSIFICATION: 'intersect' vs 'intersect_SHUFFLE' — the silhouette gap")
print("   shows whether real image content produces cleaner clusters.")
print("3. PAIRED: #better/5 close to 5 means the gain is consistent, not a")
print("   lucky-seed mean shift.")
print(f"\nSaved: graph_experiment_ari.csv + graph_experiment_perseed.csv")
