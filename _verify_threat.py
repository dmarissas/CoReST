import os, sys, numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
SEEDS = [42, 123, 456, 789, 1234]
RES = "fine"
N_REFINE = 6
sys.path.insert(0, CODE_DIR)

from utils_func import cluster_latent, refine_labels
from consensus_func import co_association_matrix, consensus_labels
import scanpy as sc

def leiden_partition(z, seed, resolution=1.0):
    ad = sc.AnnData(X=np.asarray(z, dtype=np.float32))
    sc.pp.neighbors(ad, use_rep="X", random_state=seed)
    try:
        sc.tl.leiden(ad, resolution=resolution, random_state=seed,
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(ad, resolution=resolution, random_state=seed)
    return ad.obs["leiden"].astype(int).to_numpy()

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
coords = pd.read_csv(os.path.join(PROC_DIR, "coords_final.csv"), index_col="barcode")
gold = labels.loc[barcodes, "fine_annot_type"].values
xy = coords.loc[barcodes, ["x", "y"]].values
K = len(np.unique(gold))
print(f"N={len(barcodes)} gold k={K}")

embeddings = [np.load(os.path.join(RESULT_DIR, f"embeddings_consensus_gene_seed{s}_{RES}.npy")) for s in SEEDS]

# Build base partitions exactly like run_consensus.py (kmeans, gmm, leiden)
base_parts, base_seeds, base_methods = [], [], []
for s, z in zip(SEEDS, embeddings):
    for m in ["kmeans", "gmm", "leiden"]:
        if m == "leiden":
            lab = leiden_partition(z, s)
        else:
            lab = cluster_latent(z, K, s, method=m)
        base_parts.append(lab); base_seeds.append(s); base_methods.append(m)
        print(f"  {m:<7} s={s} k_found={len(np.unique(lab))}")
B = len(base_parts)

# Reproduce published plain consensus
C_plain = co_association_matrix(base_parts)
cons20 = consensus_labels(C_plain, K)
cons20_ref = refine_labels(cons20, xy, N_REFINE)
print(f"\nREPRODUCE plain consensus cut@20:")
print(f"  raw ARI={adjusted_rand_score(gold, cons20):.4f} (post k={len(np.unique(cons20))})")
print(f"  refined ARI={adjusted_rand_score(gold, cons20_ref):.4f} (post k={len(np.unique(cons20_ref))})")
print(f"  [published plain refined = 0.6504]")

# Cut sweep (skeptic's central evidence) - plain leiden-inclusive consensus
print(f"\nCUT SWEEP (plain consensus, vary cut k -> post-refine k, refined ARI):")
for cutk in [16,17,18,19,20,21,22,23,24]:
    cons = consensus_labels(C_plain, cutk)
    consr = refine_labels(cons, xy, N_REFINE)
    print(f"  cut@{cutk:>2}: pre_k={len(np.unique(cons)):>2} -> post_k={len(np.unique(consr)):>2}  refined ARI={adjusted_rand_score(gold, consr):.4f}")

# Symmetric control: single-seed methods on k-grid, best-over-k
print(f"\nSYMMETRIC CONTROL: single-seed methods on k-grid {{18,19,20}}, refined ARI:")
for m in ["kmeans", "gmm"]:
    print(f"  --- {m} ---")
    bestmean = -1; bestk=None
    for kk in [18,19,20]:
        vals=[]
        for s,z in zip(SEEDS, embeddings):
            lab = cluster_latent(z, kk, s, method=m)
            vals.append(adjusted_rand_score(gold, refine_labels(lab, xy, N_REFINE)))
        vals=np.array(vals)
        print(f"    k={kk}: mean={vals.mean():.4f} +/- {vals.std():.4f} best={vals.max():.4f}")
        if vals.mean()>bestmean: bestmean=vals.mean(); bestk=kk
    print(f"    best-over-k mean = {bestmean:.4f} at k={bestk}")
# leiden best over resolution would be apples-to-oranges; skip but note
print("\nDONE")
