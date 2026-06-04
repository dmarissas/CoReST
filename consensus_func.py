# Partition-level CONSENSUS (evidence accumulation) for spatial-domain clustering.
#
# Pure, dependency-light functions: NO SEDR, NO torch. Everything here takes
# plain numpy arrays (label partitions / embeddings / coords) and returns numpy
# arrays, so the whole module is unit-testable standalone — run
#     python consensus_func.py
# to execute the self-test on synthetic data (no GPU, no project files needed).
#
# Why co-association and NOT embedding-averaging:
#   Each SEDR seed learns its own latent axes — rotations/permutations of the
#   embedding are arbitrary, so averaging embeddings across seeds is meaningless.
#   Whether two spots land in the SAME cluster, however, is invariant to that
#   arbitrariness. So we average over the *partitions*, not the *coordinates*:
#       C[i,j] = fraction of base partitions that put spots i and j together.
#   A deterministic cut of C at k=gold gives the final consensus partition.
#
# Pipeline:
#   base partitions  ->  co_association_matrix(C)  ->  consensus_labels (cut@k)
#   (optional twists) ->  spatial_regularize / silhouette weights
#   (deliverable)     ->  stability_map  (label-free per-spot reproducibility)

import numpy as np


# ── partition bookkeeping ───────────────────────────────────────────
def _as_label_matrix(partitions):
    """list of B label vectors (each (N,))  OR  (B, N) array  ->  (B, N) int64.

    Base partitions need NOT share the same number of clusters (KMeans/GMM at
    k, Leiden at whatever it finds) — co-association only cares about same/not-
    same, so heterogeneous cluster counts mix fine.
    """
    P = np.asarray(partitions)
    if P.ndim == 1:
        P = P[None, :]
    if P.ndim != 2:
        raise ValueError(f"partitions must be (B, N) or a list of (N,); got ndim={P.ndim}")
    return P.astype(np.int64)


# ── 1. co-association matrix ────────────────────────────────────────
def co_association_matrix(partitions, weights=None):
    """Co-association matrix C (N, N) float32.

    C[i,j] = (weighted) fraction of base partitions in which spots i and j share
    a cluster label. C[i,i] == 1. Symmetric, every entry in [0, 1].

    partitions : list of B label vectors (each (N,)), or array (B, N).
    weights    : optional (B,) non-negative per-partition weights (e.g. a
                 silhouette quality score). None -> uniform. Normalised to sum 1
                 internally; a partition with weight 0 contributes nothing.

    Memory: one N x N float32 (~58 MB at N=3798) plus a transient N x N bool per
    partition. Built in a simple loop so peak memory stays at ~2 dense matrices.
    """
    P = _as_label_matrix(partitions)
    B, N = P.shape
    if weights is None:
        w = np.full(B, 1.0 / B, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != (B,):
            raise ValueError(f"weights must be shape ({B},), got {w.shape}")
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(B, 1.0 / B)

    C = np.zeros((N, N), dtype=np.float32)
    for b in range(B):
        same = (P[b][:, None] == P[b][None, :])      # (N, N) bool
        C += w[b] * same                             # broadcast add into float32
    np.fill_diagonal(C, 1.0)
    return C


# ── 2a. spatial twist: affinity matrix ──────────────────────────────
def spatial_affinity(coords, n_neigh=6, mode="knn", sigma=None):
    """Spatial affinity S (N, N) float32 in [0, 1], row-aligned to coords.

    mode="knn"   : S[i,j] = 1 if j is among i's n_neigh nearest spatial
                   neighbours (symmetrised), self included; else 0. Cheap,
                   parameter-light, matches the SEDR spatial graph (k=6).
    mode="gauss" : S[i,j] = exp(-d_ij^2 / (2 sigma^2)); sigma defaults to the
                   median nearest-neighbour distance. Smooth but fully dense.

    Used by spatial_regularize: C' = (1-lam)C + lam (C ⊙ S) — only trust co-
    association between spots that are ALSO spatial neighbours (kills long-range
    "same cluster" votes between physically distant tissue regions).
    """
    from sklearn.neighbors import NearestNeighbors
    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    if mode == "knn":
        nn = NearestNeighbors(n_neighbors=min(n_neigh + 1, N)).fit(coords)
        _, idx = nn.kneighbors(coords)               # (N, n_neigh+1), idx[:,0]==self
        S = np.zeros((N, N), dtype=np.float32)
        rows = np.repeat(np.arange(N), idx.shape[1])
        S[rows, idx.ravel()] = 1.0
        return np.maximum(S, S.T)                     # symmetrise
    elif mode == "gauss":
        from scipy.spatial.distance import cdist
        D = cdist(coords, coords)
        if sigma is None:
            nn = NearestNeighbors(n_neighbors=2).fit(coords)
            d1, _ = nn.kneighbors(coords)
            sigma = float(np.median(d1[:, 1])) or 1.0
        return np.exp(-(D ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    raise ValueError(f"Unknown spatial affinity mode: {mode!r}")


def spatial_regularize(C, S, lam=0.5):
    """Spatial-regularised co-association: C' = (1-lam) C + lam (C ⊙ S).

    lam in [0, 1]. lam=0 -> plain C (no spatial prior). lam=1 -> co-association
    is fully gated by spatial proximity. Returns float32 (N, N).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    C = np.asarray(C, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    return ((1.0 - lam) * C + lam * (C * S)).astype(np.float32)


# ── 2b. quality twist: per-partition silhouette weights ─────────────
def silhouette_weights(embeddings, partitions, floor=0.0, sample_size=None,
                       random_state=0):
    """Per-partition quality weights = silhouette of each base partition on the
    embedding it was computed from.

    embeddings : either a single (N, d) array (reused for every partition) or a
                 list/array of B embeddings (one per partition).
    partitions : (B, N) labels (or list of B label vectors).
    floor      : minimum weight; silhouette is in [-1, 1], we clip to >= floor so
                 a partition no better than random gets ~0 vote. Default 0.0.
    sample_size: optional silhouette subsample for speed (passed to sklearn).

    Returns (B,) float64 weights (raw, un-normalised — co_association_matrix
    normalises). A partition with a single cluster gets weight = floor.
    """
    from sklearn.metrics import silhouette_score
    P = _as_label_matrix(partitions)
    B, N = P.shape
    emb_list = embeddings if isinstance(embeddings, (list, tuple)) else None
    w = np.empty(B, dtype=np.float64)
    for b in range(B):
        emb_b = np.asarray(emb_list[b]) if emb_list is not None else np.asarray(embeddings)
        labels_b = P[b]
        if len(np.unique(labels_b)) < 2:
            w[b] = floor
            continue
        s = silhouette_score(emb_b, labels_b, sample_size=sample_size,
                             random_state=random_state)
        w[b] = max(float(s), floor)
    return w


# ── 3. consensus partition ──────────────────────────────────────────
def consensus_labels(C, n_clusters, linkage="average"):
    """Final consensus partition: agglomerative clustering on distance 1 - C.

    Fully deterministic (no random seed). Average-linkage on the co-association
    distance is classic evidence accumulation (Fred & Jain 2005). Returns (N,)
    int labels in [0, n_clusters).
    """
    from sklearn.cluster import AgglomerativeClustering
    C = np.asarray(C, dtype=np.float64)
    D = 1.0 - C
    D = 0.5 * (D + D.T)                  # enforce exact symmetry
    np.fill_diagonal(D, 0.0)
    np.clip(D, 0.0, None, out=D)
    # sklearn renamed affinity -> metric in 1.2; support both.
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                        linkage=linkage)
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed",
                                        linkage=linkage)
    return model.fit_predict(D)


# ── 4. per-spot stability map (label-free deliverable) ──────────────
def _align_to_reference(part, ref, k_ref):
    """Relabel `part` so its clusters best match reference `ref` (k_ref clusters)
    by Hungarian matching on the overlap (contingency) matrix. Labels in `part`
    that have no match get pushed to fresh ids >= k_ref (so they read as
    'disagreement' rather than colliding with an aligned cluster).
    """
    from scipy.optimize import linear_sum_assignment
    part = np.asarray(part)
    ref = np.asarray(ref)
    pa = np.unique(part)
    k_p = len(pa)
    remap_idx = {lbl: i for i, lbl in enumerate(pa)}
    cont = np.zeros((k_p, k_ref), dtype=np.int64)
    for a, b in zip(part, ref):
        if 0 <= b < k_ref:
            cont[remap_idx[a], b] += 1
    row, col = linear_sum_assignment(-cont)          # maximise total overlap
    mapping = {pa[r]: int(c) for r, c in zip(row, col)}
    nxt = k_ref
    out = np.empty_like(part)
    for lbl in pa:
        if lbl not in mapping:
            mapping[lbl] = nxt
            nxt += 1
    for i, a in enumerate(part):
        out[i] = mapping[a]
    return out


def stability_map(C, consensus, partitions=None):
    """Per-spot stability scores in [0, 1] (higher = more reproducible).

    Returns a dict:
      'confidence' (N,): mean co-association of spot i with the OTHER spots in
                         its consensus cluster — how tightly the base partitions
                         agree that i belongs with its assigned group. Pure
                         function of C + consensus, no labels needed.
      'entropy'    (N,): (only if `partitions` given) normalised Shannon entropy
                         of spot i's label across base partitions, after each
                         partition is Hungarian-aligned to the consensus. 0 = the
                         spot got the same aligned label every time.
      'stability'  (N,): (only if `partitions` given) 1 - entropy, so it reads
                         the same direction as 'confidence' (higher = stabler).

    Both signals are LABEL-FREE (never touch gold), so the map is an honest
    confidence deliverable you can show without circular use of the answer key.
    """
    C = np.asarray(C, dtype=np.float64)
    consensus = np.asarray(consensus)
    N = len(consensus)
    confidence = np.zeros(N, dtype=np.float64)
    for c in np.unique(consensus):
        members = np.where(consensus == c)[0]
        if len(members) <= 1:
            confidence[members] = 1.0                # singleton: trivially "agrees"
            continue
        sub = C[np.ix_(members, members)].copy()
        np.fill_diagonal(sub, 0.0)                   # exclude self
        confidence[members] = sub.sum(axis=1) / (len(members) - 1)

    out = {"confidence": confidence}

    if partitions is not None:
        P = _as_label_matrix(partitions)
        B = P.shape[0]
        k_ref = int(consensus.max()) + 1
        aligned = np.stack([_align_to_reference(P[b], consensus, k_ref)
                            for b in range(B)])       # (B, N)
        ent = np.zeros(N, dtype=np.float64)
        log_norm = np.log(B) if B > 1 else 1.0        # max entropy with B votes
        for i in range(N):
            _, counts = np.unique(aligned[:, i], return_counts=True)
            p = counts / counts.sum()
            ent[i] = -(p * np.log(p)).sum() / log_norm
        out["entropy"] = ent
        out["stability"] = 1.0 - ent
    return out


# ── self-test (no GPU, no project files) ────────────────────────────
def _selftest():
    """Synthetic sanity check: 3 well-separated blobs, many noisy base
    partitions. Consensus should recover near-perfect ARI and stability should
    be high in cluster cores. Run:  python consensus_func.py
    """
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.RandomState(0)
    N, K = 300, 3
    X, ytrue = make_blobs(n_samples=N, centers=K, cluster_std=1.0,
                          random_state=0)
    coords = X[:, :2]

    # B noisy base partitions: KMeans with different seeds + a little label noise
    B = 12
    parts, embs = [], []
    per_seed_ari = []
    for b in range(B):
        lab = KMeans(n_clusters=K, n_init=10, random_state=b).fit_predict(X)
        flip = rng.rand(N) < 0.08                    # 8% random label corruption
        lab = lab.copy()
        lab[flip] = rng.randint(0, K, flip.sum())
        parts.append(lab)
        embs.append(X)
        per_seed_ari.append(adjusted_rand_score(ytrue, lab))

    C = co_association_matrix(parts)
    assert C.shape == (N, N)
    assert np.allclose(np.diag(C), 1.0), "diagonal must be 1"
    assert C.min() >= 0 and C.max() <= 1.0 + 1e-6, "entries must be in [0,1]"
    assert np.allclose(C, C.T), "C must be symmetric"

    cons = consensus_labels(C, K)
    ari_cons = adjusted_rand_score(ytrue, cons)

    # weighted (silhouette) variant
    w = silhouette_weights(embs, parts)
    Cw = co_association_matrix(parts, weights=w)
    ari_consw = adjusted_rand_score(ytrue, consensus_labels(Cw, K))

    # spatial twist shapes
    S = spatial_affinity(coords, n_neigh=6, mode="knn")
    assert S.shape == (N, N) and S.max() <= 1.0 and S.min() >= 0.0
    Cs = spatial_regularize(C, S, lam=0.5)
    ari_conss = adjusted_rand_score(ytrue, consensus_labels(Cs, K))

    # stability map
    sm = stability_map(C, cons, partitions=parts)
    for key in ("confidence", "entropy", "stability"):
        v = sm[key]
        assert v.shape == (N,), f"{key} wrong shape"
        assert v.min() >= -1e-9 and v.max() <= 1.0 + 1e-9, f"{key} out of [0,1]"

    best_seed = max(per_seed_ari)
    mean_seed = float(np.mean(per_seed_ari))
    std_seed = float(np.std(per_seed_ari))
    print("=" * 56)
    print("consensus_func.py self-test (synthetic 3-blob)")
    print("=" * 56)
    print(f"base partitions      : B={B}, k={K}, N={N}")
    print(f"per-seed ARI         : mean={mean_seed:.4f} ± {std_seed:.4f}  "
          f"(best={best_seed:.4f})")
    print(f"consensus ARI        : {ari_cons:.4f}")
    print(f"consensus+weight ARI : {ari_consw:.4f}")
    print(f"consensus+spatial ARI: {ari_conss:.4f}")
    print(f"mean confidence      : {sm['confidence'].mean():.4f}")
    print(f"mean stability       : {sm['stability'].mean():.4f}")
    ok = ari_cons >= best_seed - 1e-9
    print(f"\nconsensus >= best base seed ? {'YES' if ok else 'no'}  "
          f"({ari_cons:.4f} vs {best_seed:.4f})")
    print("all shape/range assertions passed.")
    return ok


if __name__ == "__main__":
    _selftest()
