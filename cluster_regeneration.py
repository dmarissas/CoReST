import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
RESULT_DIR = os.path.join(BASE_DIR, "results")
PROC_DIR   = os.path.join(BASE_DIR, "processed")

barcodes = pd.read_csv(os.path.join(PROC_DIR, "barcodes_final.csv"))["barcode"].tolist()
labels   = pd.read_csv(os.path.join(PROC_DIR, "labels_final.csv"), index_col="barcode")
labels_fine   = labels.loc[barcodes, "fine_annot_type"].values
labels_coarse = labels.loc[barcodes, "annot_type"].values

conditions = {
    "gene_only":    "embeddings_gene_only.npy",
    "image_only":   "embeddings_image_only.npy",
    "concat_fused": "embeddings_concat_fused.npy",
    "gated_fused":  "embeddings_gated_fused.npy",
}

for cond, emb_file in conditions.items():
    emb_path = os.path.join(RESULT_DIR, emb_file)
    if not os.path.exists(emb_path):
        print(f"Missing: {emb_file}")
        continue
    
    emb = np.load(emb_path)
    
    # Fine k=20
    km = KMeans(n_clusters=20, n_init=40, random_state=42)
    pred_k20 = km.fit_predict(emb)
    np.save(os.path.join(RESULT_DIR, f"clusters_{cond}_k20.npy"), pred_k20)
    ari = adjusted_rand_score(labels_fine, pred_k20)
    print(f"{cond} k20 ARI={ari:.4f} → saved")
    
    # Coarse k=4
    km = KMeans(n_clusters=4, n_init=8, random_state=42)
    pred_k4 = km.fit_predict(emb)
    np.save(os.path.join(RESULT_DIR, f"clusters_{cond}_k4.npy"), pred_k4)
    ari = adjusted_rand_score(labels_coarse, pred_k4)
    print(f"{cond} k4  ARI={ari:.4f} → saved")