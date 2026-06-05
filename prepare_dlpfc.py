# Prepare a DLPFC section .h5ad for the gene-only consensus pipeline (the DLPFC
# generalization test). Reads the single self-contained AnnData (counts + spatial
# + layer labels) and writes the SAME processed files the consensus scripts read,
# so consensus_robustness.py / run_consensus.py work on DLPFC unchanged:
#   processed/gene_only.npy        (N x 200 gene PCA, SEDR preprocessing)
#   processed/barcodes_final.csv   (barcode)
#   processed/labels_final.csv     (barcode, layer_guess)   [7 cortical layers]
#   processed/coords_final.csv     (barcode, x, y)
# Spots with no layer label are dropped.
#
# Data: Maynard et al. 2021 (Nat Neurosci, DOI 10.1038/s41593-020-00787-0) DLPFC,
# distributed via spatialLIBD, repackaged as .h5ad on Figshare
# (DOI 10.6084/m9.figshare.22004273, CC BY 4.0). Verified on 151673.h5ad:
# 3639 spots, X = raw counts, label col 'sce.layer_guess' (Layer1-6 + WM, 28 NA).
#
# Run:  python prepare_dlpfc.py                 (default: data/dlpfc_151673)
#       python prepare_dlpfc.py --dataset dlpfc_151507

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from utils_func import adata_preprocess


def find_label_col(obs):
    """Locate the manual layer annotation column (handles the 'sce.' prefix and
    common alternative names)."""
    cands = [c for c in obs.columns if "layer_guess" in c.lower()]
    cands += [c for c in obs.columns
              if c.lower() in ("layer", "region", "ground_truth", "spatiallibd")]
    return cands[0] if cands else None


def main(dataset="dlpfc_151673"):
    base = os.path.join(CODE_DIR, "data", dataset)
    proc = os.path.join(base, "processed")
    os.makedirs(proc, exist_ok=True)

    h5s = sorted(glob.glob(os.path.join(base, "*.h5ad")))
    if not h5s:
        print(f"MISSING *.h5ad in {base}")
        sys.exit(1)
    print("=" * 60)
    print(f"PREPARE {dataset}  ({os.path.basename(h5s[0])})")
    print("=" * 60)

    adata = sc.read_h5ad(h5s[0])
    adata.var_names_make_unique()
    print(f"loaded: {adata.shape[0]} spots x {adata.shape[1]} genes")

    col = find_label_col(adata.obs)
    if col is None:
        print(f"NO layer-label column found in obs: {list(adata.obs.columns)}")
        sys.exit(1)
    print(f"label column: '{col}'  "
          f"({adata.obs[col].nunique(dropna=True)} classes, {int(adata.obs[col].isna().sum())} NA)")

    # drop unlabeled spots
    adata = adata[adata.obs[col].notna()].copy()
    print(f"after dropping NA labels: {adata.shape[0]} spots")

    # adata_preprocess calls X.toarray(); ensure the (raw-count) matrix is sparse
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # capture aligned barcodes / labels / coords (preprocess subsets genes only,
    # so spot order is preserved — asserted below)
    barcodes = list(adata.obs_names)
    labels = adata.obs[col].astype(str).values
    xy = np.asarray(adata.obsm["spatial"])[:, :2]

    # gene PCA 200 — identical SEDR preprocessing to HBRC (HVG seurat_v3 + PCA)
    print("HVG (seurat_v3, 2000) + PCA 200d ...")
    adata = adata_preprocess(adata, min_cells=50, min_counts=10, pca_n_comps=200)
    gene_pca = np.asarray(adata.obsm["X_pca"]).astype(np.float32)
    assert list(adata.obs_names) == barcodes, "spot order changed during preprocess"

    # write the *_final files the consensus scripts read
    np.save(os.path.join(proc, "gene_only.npy"), gene_pca)
    pd.DataFrame({"barcode": barcodes}).to_csv(
        os.path.join(proc, "barcodes_final.csv"), index=False)
    pd.DataFrame({"layer_guess": labels}, index=barcodes).rename_axis("barcode").to_csv(
        os.path.join(proc, "labels_final.csv"))
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=barcodes).rename_axis("barcode").to_csv(
        os.path.join(proc, "coords_final.csv"))

    print(f"\nwrote -> {proc}")
    print(f"  gene_only.npy      {gene_pca.shape}")
    print(f"  barcodes_final.csv ({len(barcodes)} spots)")
    print(f"  labels_final.csv   (col 'layer_guess', k={len(set(labels))})")
    print(f"  coords_final.csv   (x, y)")
    print("\nNext: run the robustness study pointing at this dataset "
          "(label column = 'layer_guess', k auto = %d)." % len(set(labels)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dlpfc_151673",
                    help="folder under data/ containing the .h5ad (default dlpfc_151673)")
    args = ap.parse_args()
    main(args.dataset)
