# Extract gene expression features using SEDR preprocessing.
#
# Input : raw Visium .h5 file
# Output: processed/gene_features_200d.csv  [barcode, gene_pca_0..199]
#         processed/coords.csv              [barcode, x, y]
#         processed/labels.csv             [barcode, annot_type, fine_annot_type]

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
H5_FILE  = "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5"
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
from utils_func import adata_preprocess

PROC_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Gene Features + Coordinates + Labels")
print("=" * 60)

# ── Load Visium ────────────────────────────────────────────────────
print("\n[1] Loading Visium data...")
# load_images=False: this pipeline only needs counts + spot coordinates.
# The tissue_hires/lowres PNG thumbnails (part of the 10x spatial tar.gz) are
# not used here; histology features come from the full .tif in image_features.py.
adata = sc.read_visium(path=BASE_DIR, count_file=H5_FILE, load_images=False)
print(f"    Spots: {adata.n_obs}  Genes: {adata.n_vars}")

# ── Save coordinates (pixel coords from tissue_positions_list.csv) ─
# Read directly from the positions file rather than adata.obsm["spatial"],
# because read_visium(load_images=False) does not populate the spatial slot.
# Columns (10x v1.1, no header): barcode, in_tissue, array_row, array_col,
#                                pxl_row_in_fullres, pxl_col_in_fullres
# scanpy's obsm["spatial"] convention is [pxl_col_in_fullres, pxl_row_in_fullres].
print("\n[2] Saving spatial coordinates...")
pos = pd.read_csv(
    os.path.join(BASE_DIR, "spatial", "tissue_positions_list.csv"),
    header=None,
    names=["barcode", "in_tissue", "array_row", "array_col",
           "pxl_row_in_fullres", "pxl_col_in_fullres"],
    index_col="barcode",
)
coords = pd.DataFrame({
    "x": pos["pxl_col_in_fullres"],
    "y": pos["pxl_row_in_fullres"],
}).loc[adata.obs_names]
coords.index.name = "barcode"
coords.to_csv(os.path.join(PROC_DIR, "coords.csv"))
print(f"    Saved coords.csv  shape={coords.shape}")

# ── Load gold standard labels ──────────────────────────────────────
print("\n[3] Loading gold standard labels...")
meta_path = os.path.join(BASE_DIR, "metadata.tsv")
meta = pd.read_csv(meta_path, sep="\t", index_col=0)
meta.index.name = "barcode"

# Align to adata barcode order
common = adata.obs_names.intersection(meta.index)
print(f"    Spots with labels: {len(common)} / {adata.n_obs}")
print(f"    annot_type distribution:")
print(meta.loc[common, "annot_type"].value_counts().to_string())

meta_aligned = meta.loc[common]
# .loc with an intersected Index drops the index name, so re-assert it —
# prepare_features.py reads this file with index_col="barcode".
meta_aligned.index.name = "barcode"
meta_aligned.to_csv(os.path.join(PROC_DIR, "labels.csv"))
print(f"    Saved labels.csv")

# ── Gene feature extraction ────────────────────────────────────────
print("\n[4] SEDR gene preprocessing (HVG + PCA 200d)...")
adata = adata_preprocess(adata, min_cells=50, min_counts=10, pca_n_comps=200)
print(f"    Genes after filter: {adata.n_vars}")
print(f"    PCA shape: {adata.obsm['X_pca'].shape}")

df_gene = pd.DataFrame(
    adata.obsm["X_pca"],
    index=adata.obs_names,
    columns=[f"gene_pca_{i}" for i in range(200)]
)
df_gene.index.name = "barcode"
df_gene.to_csv(os.path.join(PROC_DIR, "gene_features_200d.csv"))

print(f"\n[5] Summary:")
print(f"    gene_features_200d.csv : {df_gene.shape}")
print(f"    coords.csv             : {coords.shape}")
print(f"    labels.csv             : {meta_aligned.shape}")