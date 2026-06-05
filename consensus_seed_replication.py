# ABLATION: is the consensus robust to the SEED POOL itself?
#
# All the robustness checks in run_consensus.py (LOSO, seed-subset, bootstrap)
# resample WITHIN the original 5 seeds [42,123,456,789,1234]. They show the answer
# is stable to dropping/subsetting THOSE seeds — but NOT whether a genuinely
# DIFFERENT set of seeds would also land at ~0.65. This script closes that gap:
#
#   pool A (original) : [42, 123, 456, 789, 1234]      — uses cached embeddings
#   pool B (new)      : disjoint set                   — trains SEDR fresh (GPU)
#   pool A+B          : all 10 seeds                    — does more seeds help/stabilize?
#
# For each pool it builds the PLAIN consensus (5 or 10 seeds x {KMeans,GMM,Leiden}
# of the gene-only SEDR embedding) and reports its deterministic ARI + that pool's
# best single seed. If consensus_A ~ consensus_B (small |A-B|), the headline does
# NOT depend on the lucky original draw.
#
# NOT cheap: pool B requires ~5 fresh SEDR trainings (~5-6 min on GPU). Pool A and
# A+B reuse cached embeddings. Run it yourself in the GateST terminal:
#     python consensus_seed_replication.py
# Output: results/consensus_seed_replication.csv

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from graph_func import graph_construction
from utils_func import cluster_latent, refine_labels
from run_consensus import run_sedr_once, leiden_partition, _leiden_available  # module-level, import-safe
from consensus_func import co_association_matrix, consensus_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
N_REFINE_NEIGH = 6

SEEDS_A = [42, 123, 456, 789, 1234]          # original pool (cached)
SEEDS_B = [7, 88, 314, 2024, 51966]          # NEW disjoint pool (trains fresh)


def embedding_for_seed(s, gene_feats, graph, K):
    """Cached gene-only SEDR embedding for seed s (same cache as run_consensus.py)."""
    cache = os.path.join(RESULT_DIR, f"embeddings_consensus_gene_seed{s}_fine.npy")
    if os.path.exists(cache):
        return np.load(cache), True
    print(f"    seed={s}: training SEDR ...", end=" ", flush=True)
    z = run_sedr_once(gene_feats, graph, K, s)
    np.save(cache, z)
    print(f"done {z.shape}")
    return z, False


def consensus_for_pool(pool, gene_feats, graph, gold, xy, K, methods):
    """Plain consensus + best-single-seed (GMM, refined) for a seed pool."""
    embs = [embedding_for_seed(s, gene_feats, graph, K)[0] for s in pool]
    parts, singles = [], []
    for s, z in zip(pool, embs):
        # best-single comparator: GMM refined (strongest single clusterer)
        gmm = refine_labels(cluster_latent(z, K, s, method="gmm"), xy, N_REFINE_NEIGH)
        singles.append(adjusted_rand_score(gold, gmm))
        for m in methods:
            lab = leiden_partition(z, s) if m == "leiden" else cluster_latent(z, K, s, method=m)
            if lab is not None:
                parts.append(lab)
    C = co_association_matrix(parts)
    cons = refine_labels(consensus_labels(C, K), xy, N_REFINE_NEIGH)
    return adjusted_rand_score(gold, cons), max(singles), len(parts)


def main():
    print("=" * 64)
    print("ABLATION — is the consensus robust to the SEED POOL?")
    print("=" * 64)
    print(f"Device: {DEVICE}")
    print(f"pool A (original, cached): {SEEDS_A}")
    print(f"pool B (new, trains)     : {SEEDS_B}")

    barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
    labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
    coords   = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
    gold = labels.loc[barcodes, "fine_annot_type"].values
    xy   = coords.loc[barcodes, ["x", "y"]].values
    gene_feats = np.load(os.path.join(PROC_DIR, "gene_only.npy"))
    K = len(np.unique(gold))

    adata_g = sc.AnnData(X=gene_feats)
    adata_g.obsm["spatial"] = xy
    graph = graph_construction(adata_g, n=6, mode="KNN")

    methods = ["kmeans", "gmm", "leiden"]
    if not _leiden_available():
        methods.remove("leiden")
        print("[!] leiden deps not found — using kmeans+gmm only")

    pools = [("A_original", SEEDS_A), ("B_new", SEEDS_B), ("A+B_pooled", SEEDS_A + SEEDS_B)]
    rows = []
    for tag, pool in pools:
        print(f"\n[{tag}] {len(pool)} seeds")
        cons_ari, best_single, n_parts = consensus_for_pool(
            pool, gene_feats, graph, gold, xy, K, methods)
        rows.append({"pool": tag, "n_seeds": len(pool), "n_partitions": n_parts,
                     "consensus_ARI": round(cons_ari, 4),
                     "best_single_seed": round(best_single, 4),
                     "consensus_minus_best": round(cons_ari - best_single, 4)})
        print(f"    consensus={cons_ari:.4f}  best_single={best_single:.4f}  "
              f"(+{cons_ari-best_single:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "consensus_seed_replication.csv"), index=False)

    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    print(df.to_string(index=False))

    a = df[df["pool"] == "A_original"]["consensus_ARI"].values[0]
    b = df[df["pool"] == "B_new"]["consensus_ARI"].values[0]
    print(f"\nReplication check: |consensus_A - consensus_B| = {abs(a-b):.4f}")
    print(f"  pool A consensus = {a:.4f}   pool B consensus = {b:.4f}")
    if abs(a - b) <= 0.02:
        print("  -> SMALL gap: the headline is robust to the seed pool (not a lucky draw).")
    else:
        print("  -> LARGE gap: the result depends on which seeds were drawn — report honestly.")
    print(f"\nSaved: results/consensus_seed_replication.csv")


if __name__ == "__main__":
    main()
