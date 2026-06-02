#
import os
import torch
import numpy as np
import scanpy as sc

def adata_preprocess(adata_vis, min_cells=50, min_counts=10, pca_n_comps=200):
    adata_vis.layers['count'] = adata_vis.X.toarray()
    sc.pp.filter_genes(adata_vis, min_cells=min_cells)
    sc.pp.filter_genes(adata_vis, min_counts=min_counts)

#     adata_vis.obs['mean_exp'] = adata_vis.X.toarray().mean(axis=1)
#     adata_vis.var['mean_exp'] = adata_vis.X.toarray().mean(axis=0)
#
#     # Load scRNA-seq data
#     adata_ref = sc.read_h5ad('/home/xuhang/disco_500t/Projects/spTrans/data/reference_data/GSE144136_DLPFC/raw/processed_raw.h5ad')
#     adata_ref.obs['mean_exp'] = adata_ref.X.toarray().mean(axis=1)
#     adata_ref.var['mean_exp'] = adata_ref.X.toarray().mean(axis=0)
#     common_genes = np.intersect1d(adata_vis.var.index, adata_ref.var.index)
#     adata_vis = adata_vis[:, common_genes]
#     adata_ref = adata_ref[:, common_genes]
#     adata_vis.var['ref_mean_exp'] = adata_ref.var['mean_exp']
#     adata_vis.var['ratio'] = np.log10(adata_vis.var['mean_exp'] / adata_vis.var['ref_mean_exp']+1)
#     adata_vis.var['selected'] = adata_vis.var['ratio'] < 1.5
#     remain_genes = adata_vis.var[adata_vis.var['selected']==True].index.tolist()
#     adata_vis = adata_vis[:, remain_genes]
#
#
    sc.pp.normalize_total(adata_vis, target_sum=1e6)
    # sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata_vis = adata_vis[:, adata_vis.var['highly_variable'] == True]
    sc.pp.scale(adata_vis)

    from sklearn.decomposition import PCA
    adata_X = PCA(n_components=pca_n_comps, random_state=42).fit_transform(adata_vis.X)
    adata_vis.obsm['X_pca'] = adata_X
    return adata_vis


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def cluster_latent(z, n_clusters, seed, method="kmeans"):
    """Cluster a latent embedding `z` (N, d) into `n_clusters`.

    method:
      "kmeans" — KMeans with n_init = 2*n_clusters (the project default).
      "gmm"    — GaussianMixture(covariance_type="full"), the dependency-free
                 stand-in for R's mclust (full ~ mclust "VVV"), which is what
                 SEDR/STAGATE papers use as their final clusterer.
    Returns integer labels (N,).
    """
    from sklearn.cluster import KMeans
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=n_clusters * 2,
                      random_state=seed).fit_predict(z)
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=n_clusters, covariance_type="full",
                               n_init=10, random_state=seed).fit_predict(z)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}")


def refine_labels(pred, coords, n_neigh=6, n_iter=1):
    """Spatial label refinement (post-clustering smoothing).

    Reassign each spot's cluster label to the majority label among its
    `n_neigh` nearest spatial neighbours (self included). Ties keep the spot's
    current label. This is the standard SEDR/STAGATE/DLPFC post-processing trick
    and typically adds ~0.02-0.07 ARI. Uses ONLY the spatial coords, so when
    applied across conditions it treats every condition identically.

    pred   : (N,) integer cluster labels
    coords : (N, 2) spatial pixel coordinates, row-aligned to pred
    """
    from sklearn.neighbors import NearestNeighbors
    pred = np.asarray(pred).copy()
    coords = np.asarray(coords)
    # +1 because the first neighbour returned is the spot itself
    nn = NearestNeighbors(n_neighbors=n_neigh + 1).fit(coords)
    _, idx = nn.kneighbors(coords)  # (N, n_neigh+1), idx[:,0] == self
    for _ in range(n_iter):
        new = pred.copy()
        for i in range(len(pred)):
            vals, counts = np.unique(pred[idx[i]], return_counts=True)
            winners = vals[counts == counts.max()]
            # tie-break: keep current label if it is among the winners
            new[i] = pred[i] if pred[i] in winners else winners[0]
        pred = new
    return pred
