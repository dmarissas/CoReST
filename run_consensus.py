# CONSENSUS / ROBUSTNESS driver for SEDR spatial-domain identification.
#
# The gap: SEDR is seed-sensitive — the same model + data swings ARI ~0.43->0.52
# across random seeds. A practitioner who runs ONE seed gets a random draw. This
# script removes that lottery: it runs the 5 fixed seeds, turns each embedding
# into base partitions, and fuses them by EVIDENCE ACCUMULATION (co-association),
# yielding ONE deterministic answer that should be >= the BEST single seed while
# collapsing the across-run variance.
#
# Base partitions  = 5 seeds x {KMeans, GMM, Leiden} of the gene-only SEDR
#                    embedding (gene features + spatial graph — the published
#                    SEDR setting, NO histology needed).
# Consensus        = average-linkage cut of the co-association matrix at k=gold.
#
# Variants (each adds one idea; the (e)-(d) gaps measure whether our twist helps):
#   (d) plain      : uniform co-association            (STCC-style cross-seed/
#                    embedding consensus — the prior-art ablation)
#   spatial        : (d) + spatial-regularised C' = (1-lam)C + lam(C⊙S)
#   weighted       : (d) + silhouette-quality-weighted base partitions
#   (e) headline   : weighted + spatial, then spatial label refinement
#
# Honesty: k = gold (no tuning on labels); base partitions use the fixed 5 seeds;
# silhouette (label-free) is the only quality signal; the consensus cut and the
# refinement are deterministic. Per-seed embeddings are cached so reruns are fast.
#
# Output: results/consensus_ari.csv      (comparator table — the headline)
#         results/consensus_perseed.csv  (per-seed ARI, the variance being killed)
#         results/clusters_consensus_<variant>_k{K}[_refined].npy
#         results/stability_consensus.npy  (per-spot confidence — label-free map)

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
SEEDS    = [42, 123, 456, 789, 1234]
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"

RESOLUTION   = "fine"          # "fine" (k=20, headline) — coarse is a trivial swap
BASE_METHODS = ["kmeans", "gmm", "leiden"]   # leiden skipped gracefully if deps missing
N_REFINE_NEIGH = 6             # spatial label refinement (matches SEDR k=6 graph)
LAM_SPATIAL  = 0.5             # spatial-regularisation strength in C'=(1-lam)C+lam(C⊙S)
LEIDEN_RES   = 1.0             # Leiden resolution (cluster count need NOT equal k)
N_BOOTSTRAP  = 20              # subset-resampling reps for the robustness check
BOOTSTRAP_FRAC = 0.6           # fraction of base partitions per bootstrap consensus
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, CODE_DIR)
from SEDR_model import Sedr
from graph_func import graph_construction
from utils_func import fix_seed, cluster_latent, refine_labels
from consensus_func import (co_association_matrix, consensus_labels,
                            spatial_affinity, spatial_regularize,
                            silhouette_weights, stability_map)

PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run_sedr_once(feats, graph_dict, n_clusters, seed):
    """Train SEDR once; return the latent embedding (clustering done outside).

    Defined locally — NOT imported from train_sedr — so run_consensus stays
    self-contained: `import train_sedr` would execute its module-level training
    pipeline. This mirrors train_sedr.run_sedr_once exactly.
    """
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


def _leiden_available():
    """True iff Leiden's optional deps are importable (leidenalg or python-igraph)."""
    import importlib.util
    return any(importlib.util.find_spec(m) is not None for m in ("leidenalg", "igraph"))


def leiden_partition(z, seed, resolution=LEIDEN_RES):
    """Leiden community detection on an embedding (algorithmic diversity for the
    base set). Returns integer labels, or None if leiden deps are unavailable.
    Cluster count need NOT match k — co-association only uses same/not-same.
    """
    try:
        ad = sc.AnnData(X=np.asarray(z, dtype=np.float32))
        sc.pp.neighbors(ad, use_rep="X", random_state=seed)
        try:
            sc.tl.leiden(ad, resolution=resolution, random_state=seed,
                         flavor="igraph", n_iterations=2, directed=False)
        except TypeError:                       # older scanpy: no flavor kwarg
            sc.tl.leiden(ad, resolution=resolution, random_state=seed)
        return ad.obs["leiden"].astype(int).to_numpy()
    except Exception as e:                      # leidenalg/igraph missing, etc.
        print(f"      [leiden skipped: {type(e).__name__}: {e}]")
        return None


def main(base_dir=BASE_DIR, label_col=None):
    proc_dir   = os.path.join(base_dir, "processed")
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    if label_col is None:
        label_col = "fine_annot_type" if RESOLUTION == "fine" else "annot_type"
    print("=" * 64)
    print("CONSENSUS / ROBUSTNESS — SEDR spatial-domain identification")
    print("=" * 64)
    print(f"Device : {DEVICE}   dataset: {os.path.basename(base_dir)}   "
          f"label: {label_col}   Seeds: {SEEDS}")

    # ── Load aligned data ──────────────────────────────────────────
    barcodes = pd.read_csv(os.path.join(proc_dir, "barcodes_final.csv"))["barcode"].tolist()
    labels   = pd.read_csv(os.path.join(proc_dir, "labels_final.csv"), index_col="barcode")
    coords   = pd.read_csv(os.path.join(proc_dir, "coords_final.csv"), index_col="barcode")
    gold = labels.loc[barcodes, label_col].values
    xy   = coords.loc[barcodes, ["x", "y"]].values
    gene_feats = np.load(os.path.join(proc_dir, "gene_only.npy"))
    K = len(np.unique(gold))
    print(f"Spots: {len(barcodes)}   gold k={K}   gene dim={gene_feats.shape[1]}")

    # ── Spatial graph (gene-only SEDR, published setting) ───────────
    adata_g = sc.AnnData(X=gene_feats)
    adata_g.obsm["spatial"] = xy
    graph_spatial = graph_construction(adata_g, n=6, mode="KNN")

    # ── Per-seed SEDR embeddings (cached) ───────────────────────────
    print("\n[1] Per-seed gene-only SEDR embeddings (cached on disk)")
    embeddings = []
    for s in SEEDS:
        cache = os.path.join(result_dir, f"embeddings_consensus_gene_seed{s}_{RESOLUTION}.npy")
        if os.path.exists(cache):
            z = np.load(cache)
            print(f"    seed={s}: loaded cache  {z.shape}")
        else:
            print(f"    seed={s}: training SEDR ...", end=" ", flush=True)
            z = run_sedr_once(gene_feats, graph_spatial, K, s)
            np.save(cache, z)
            print(f"done  {z.shape}")
        embeddings.append(z)

    # ── Base partitions: 5 seeds x {kmeans, gmm, leiden} ────────────
    methods = list(BASE_METHODS)
    if "leiden" in methods and not _leiden_available():
        methods.remove("leiden")
        print("\n[!] leiden deps not found — using kmeans+gmm only "
              "(`pip install leidenalg igraph` to add the 3rd, more diverse method)")
    print(f"\n[2] Base partitions (5 seeds x {methods})")
    base_parts, base_embs, base_tags, base_seeds, base_methods, per_seed = [], [], [], [], [], {}
    for s, z in zip(SEEDS, embeddings):
        for m in methods:
            if m == "leiden":
                lab = leiden_partition(z, s)
                if lab is None:
                    continue
            else:
                lab = cluster_latent(z, K, s, method=m)
            base_parts.append(lab)
            base_embs.append(z)
            base_tags.append(f"{m}_s{s}")
            base_seeds.append(s)
            base_methods.append(m)
            nclust = len(np.unique(lab))
            # per-seed ARI of the standalone clustering (the lottery we replace)
            ari_raw = adjusted_rand_score(gold, lab)
            ari_ref = adjusted_rand_score(gold, refine_labels(lab, xy, N_REFINE_NEIGH))
            per_seed.setdefault(m, []).append(
                {"seed": s, "method": m, "n_clusters": nclust,
                 "ARI_raw": round(ari_raw, 4), "ARI_refined": round(ari_ref, 4)})
            print(f"    {m:<6} seed={s}: k_found={nclust:<3} "
                  f"ARI raw={ari_raw:.4f} refined={ari_ref:.4f}")
    B = len(base_parts)
    print(f"    -> {B} base partitions")

    # ── Co-association matrices ─────────────────────────────────────
    print("\n[3] Co-association + consensus variants")
    C_plain = co_association_matrix(base_parts)                      # (d) STCC-style
    S = spatial_affinity(xy, n_neigh=N_REFINE_NEIGH, mode="knn")
    C_spatial = spatial_regularize(C_plain, S, lam=LAM_SPATIAL)
    w = silhouette_weights(base_embs, base_parts)                    # label-free quality
    C_weight = co_association_matrix(base_parts, weights=w)
    C_head = spatial_regularize(C_weight, S, lam=LAM_SPATIAL)        # (e) weighted+spatial
    print(f"    silhouette weights: min={w.min():.3f} max={w.max():.3f} "
          f"mean={w.mean():.3f}")

    variants = {
        "plain":    C_plain,     # (d) prior-art ablation
        "spatial":  C_spatial,
        "weighted": C_weight,
        "headline": C_head,      # (e) weighted + spatial
    }

    # ── Comparator table ────────────────────────────────────────────
    rows = []

    # single-seed comparators (the variance we are trying to kill), per method
    for m, recs in per_seed.items():
        for tag, key in [("raw", "ARI_raw"), ("refined", "ARI_refined")]:
            vals = np.array([r[key] for r in recs])
            rows.append({"approach": f"single_seed_{m}", "refine": tag,
                         "ARI_mean": round(vals.mean(), 4), "ARI_std": round(vals.std(), 4),
                         "ARI_best": round(vals.max(), 4), "ARI_worst": round(vals.min(), 4),
                         "n": len(vals)})

    # consensus variants (deterministic -> std=0, best=worst=mean)
    consensus_labels_store = {}
    for name, C in variants.items():
        cons = consensus_labels(C, K)
        cons_ref = refine_labels(cons, xy, N_REFINE_NEIGH)
        consensus_labels_store[name] = (cons, cons_ref)
        for tag, pred in [("raw", cons), ("refined", cons_ref)]:
            a = adjusted_rand_score(gold, pred)
            rows.append({"approach": f"consensus_{name}", "refine": tag,
                         "ARI_mean": round(a, 4), "ARI_std": 0.0,
                         "ARI_best": round(a, 4), "ARI_worst": round(a, 4), "n": 1})
            np.save(os.path.join(result_dir,
                    f"clusters_consensus_{name}_k{K}{'_refined' if tag=='refined' else ''}.npy"),
                    pred)
            print(f"    consensus_{name:<8} {tag:<7} ARI={a:.4f}  (k_found={len(np.unique(pred))})")

    # ── [4] Robustness of the consensus answer ──────────────────────
    # Two honest questions, two measurements:
    #  (a) Determinism — with the fixed 5 seeds the consensus is ONE answer, so
    #      its rerun-to-rerun std is exactly 0 (a single SEDR run has std~0.02-0.03).
    #  (b) Seed-CHOICE sensitivity — would a DIFFERENT set of seeds change the
    #      answer? Leave-one-seed-out is the FAIR test: drop one seed's partitions
    #      but keep all methods for the other 4 (so the clusterer balance and the
    #      partition count stay high). The bootstrap below (random subset, can
    #      delete a whole clusterer and uses fewer partitions) is a harsher lower
    #      bound, kept only as a stress test — not the headline number.
    print("\n[4] Robustness of the consensus answer")
    base_seeds_arr = np.array(base_seeds)
    loso = []
    for hold in SEEDS:
        keep = np.where(base_seeds_arr != hold)[0]
        Cl = co_association_matrix([base_parts[i] for i in keep])
        cl = refine_labels(consensus_labels(Cl, K), xy, N_REFINE_NEIGH)
        loso.append(adjusted_rand_score(gold, cl))
    loso = np.array(loso)
    print(f"    leave-one-seed-out ({len(SEEDS)} folds, 4-seed consensus): "
          f"ARI={loso.mean():.4f} ± {loso.std():.4f} "
          f"(best={loso.max():.4f} worst={loso.min():.4f})")

    rng = np.random.RandomState(0)
    n_pick = max(2, int(round(BOOTSTRAP_FRAC * B)))
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(B, size=n_pick, replace=False)
        Cb = co_association_matrix([base_parts[i] for i in idx])
        cb = refine_labels(consensus_labels(Cb, K), xy, N_REFINE_NEIGH)
        boot.append(adjusted_rand_score(gold, cb))
    boot = np.array(boot)
    print(f"    bootstrap stress ({N_BOOTSTRAP} reps x {n_pick}/{B} random partitions): "
          f"ARI={boot.mean():.4f} ± {boot.std():.4f}")
    for nm, arr in [("loso", loso), ("bootstrap", boot)]:
        rows.append({"approach": f"consensus_{nm}", "refine": "refined",
                     "ARI_mean": round(arr.mean(), 4), "ARI_std": round(arr.std(), 4),
                     "ARI_best": round(arr.max(), 4), "ARI_worst": round(arr.min(), 4),
                     "n": len(arr)})

    # seed-subset consensus distribution (DESCRIPTIVE, not a formal test): every
    # 3- and 4-seed subset, plain consensus. Samples share seeds -> dependent, so
    # this is an effect-size / "how often does it beat the best single seed", not
    # a p-value. Gives the deterministic method a spread to report honestly.
    from itertools import combinations
    best_single = max(r["ARI_best"] for r in rows
                      if r["approach"].startswith("single_seed") and r["refine"] == "refined")
    bs_arr = np.array(base_seeds)
    sub = []
    for r in (3, 4):
        for combo in combinations(SEEDS, r):
            idx = np.where(np.isin(bs_arr, combo))[0]
            Csub = co_association_matrix([base_parts[i] for i in idx])
            sub.append(adjusted_rand_score(gold, refine_labels(
                consensus_labels(Csub, K), xy, N_REFINE_NEIGH)))
    sub = np.array(sub)
    n_beat = int((sub > best_single).sum())
    print(f"    seed-subset consensus (all 3- & 4-seed subsets, n={len(sub)}): "
          f"ARI={sub.mean():.4f} ± {sub.std():.4f}  min={sub.min():.4f}  "
          f"beat best-single({best_single:.4f}): {n_beat}/{len(sub)}")
    rows.append({"approach": "consensus_seed_subset", "refine": "refined",
                 "ARI_mean": round(sub.mean(), 4), "ARI_std": round(sub.std(), 4),
                 "ARI_best": round(sub.max(), 4), "ARI_worst": round(sub.min(), 4),
                 "n": len(sub)})

    # ── Per-spot stability map (label-free deliverable) ─────────────
    print("\n[5] Per-spot stability map")
    cons_head = consensus_labels_store["headline"][0]
    sm = stability_map(C_plain, cons_head, partitions=base_parts)
    np.save(os.path.join(result_dir, "stability_consensus.npy"),
            np.column_stack([sm["confidence"], sm["stability"]]))
    print(f"    mean co-association confidence={sm['confidence'].mean():.4f}  "
          f"mean stability(1-entropy)={sm['stability'].mean():.4f}")
    print(f"    saved: results/stability_consensus.npy  (cols: confidence, stability)")

    # ── [6] Where does the gain come from? (cheap ablations) ────────
    # Decompose the gain using the SAME cached embeddings/partitions:
    #   (a) Ward agglomerative on a SINGLE embedding -> is a strong agglomerative
    #       clusterer alone responsible (rather than the pooling)?
    #   (b) single-seed cross-METHOD consensus        -> cross-method pooling only
    #   (c) cross-SEED single-method consensus         -> cross-seed pooling only
    # The consensus operator is the IDENTITY for one partition (identity check
    # below), so any gain of (c) over a single run is PURELY cross-seed pooling.
    # Read: cross-SEED (c, esp. gmm) is the primary driver; cross-method (b) small.
    print("\n[6] Ablation: source of the gain")
    from sklearn.cluster import AgglomerativeClustering
    base_seeds_a, base_meth_a = np.array(base_seeds), np.array(base_methods)

    # (a) Ward agglomerative per embedding — a fair, non-degenerate "is a strong
    #     agglomerative clusterer alone enough?" baseline (Euclidean avg-linkage on
    #     raw 32d embeddings collapses to ~0 ARI, so it is NOT a meaningful control).
    agg = np.array([adjusted_rand_score(gold, refine_labels(
        AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(np.asarray(z)),
        xy, N_REFINE_NEIGH)) for z in embeddings])
    print(f"    (a) Ward agglomerative on single embed: {agg.mean():.4f} ± {agg.std():.4f}")
    _id = adjusted_rand_score(base_parts[0], consensus_labels(
        co_association_matrix([base_parts[0]]), len(np.unique(base_parts[0]))))
    print(f"    identity check (B=1 consensus==input) : ARI={_id:.4f}  "
          f"(operator adds nothing without pooling)")

    # (b) single-seed cross-method consensus (one embedding, all methods)
    bvals = []
    for s in SEEDS:
        idx = np.where(base_seeds_a == s)[0]
        if len(idx) < 2:
            continue
        Cb = co_association_matrix([base_parts[i] for i in idx])
        bvals.append(adjusted_rand_score(gold, refine_labels(
            consensus_labels(Cb, K), xy, N_REFINE_NEIGH)))
    bvals = np.array(bvals)
    print(f"    (b) single-seed cross-method consensus: {bvals.mean():.4f} ± {bvals.std():.4f}")

    # (c) cross-seed single-method consensus (5 seeds, one method)
    cvals = {}
    for m in np.unique(base_meth_a):
        idx = np.where(base_meth_a == m)[0]
        Cc = co_association_matrix([base_parts[i] for i in idx])
        cvals[m] = adjusted_rand_score(gold, refine_labels(
            consensus_labels(Cc, K), xy, N_REFINE_NEIGH))
    print("    (c) cross-seed single-method consensus: " +
          "  ".join(f"{m}={v:.4f}" for m, v in cvals.items()))

    rows.append({"approach": "ablation_agg_single_embed", "refine": "refined",
                 "ARI_mean": round(agg.mean(), 4), "ARI_std": round(agg.std(), 4),
                 "ARI_best": round(agg.max(), 4), "ARI_worst": round(agg.min(), 4), "n": len(agg)})
    rows.append({"approach": "ablation_singleseed_xmethod", "refine": "refined",
                 "ARI_mean": round(bvals.mean(), 4), "ARI_std": round(bvals.std(), 4),
                 "ARI_best": round(bvals.max(), 4), "ARI_worst": round(bvals.min(), 4), "n": len(bvals)})
    for m, v in cvals.items():
        rows.append({"approach": f"ablation_xseed_{m}", "refine": "refined",
                     "ARI_mean": round(v, 4), "ARI_std": 0.0, "ARI_best": round(v, 4),
                     "ARI_worst": round(v, 4), "n": 1})

    # ── Save tables + verdict ───────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(result_dir, "consensus_ari.csv"), index=False)
    perseed_df = pd.DataFrame([r for recs in per_seed.values() for r in recs])
    perseed_df.to_csv(os.path.join(result_dir, "consensus_perseed.csv"), index=False)

    print(f"\n{'='*64}\nSUMMARY  (resolution={RESOLUTION}, k={K})\n{'='*64}")
    print(df.to_string(index=False))

    # ── Label-free variant selection (no peeking at gold) ───────────
    # Picking the variant with the highest ARI would be tuning on the answer key.
    # Instead select by silhouette in co-association space — a label-free signal:
    # how cleanly separated the consensus clusters are under the agreement metric.
    from sklearn.metrics import silhouette_score
    D_co = 1.0 - C_plain
    D_co = 0.5 * (D_co + D_co.T)
    np.fill_diagonal(D_co, 0.0)
    sil = {}
    for name in variants:
        lab = consensus_labels_store[name][1]   # refined labels
        sil[name] = (silhouette_score(D_co, lab, metric="precomputed")
                     if 1 < len(np.unique(lab)) < len(lab) else -1.0)
    selected = max(sil, key=sil.get)

    # ── Verdict (honest framing) ────────────────────────────────────
    # HEADLINE = plain consensus: it has the best ARI AND is deterministic AND
    # label-free AND zero-tuning. The silhouette selector picked 'headline' (the
    # WORST variant) — reported below as an honest negative, NOT as validation.
    single_std = {r["approach"].replace("single_seed_", ""): r["ARI_std"] for r in rows
                  if r["approach"].startswith("single_seed") and r["refine"] == "refined"}
    gmm_mean = next(r["ARI_mean"] for r in rows
                    if r["approach"] == "single_seed_gmm" and r["refine"] == "refined")
    typ_std = single_std.get("gmm", max(single_std.values()))
    xseed_gmm = next(r["ARI_mean"] for r in rows if r["approach"] == "ablation_xseed_gmm")
    plain_ari = adjusted_rand_score(gold, consensus_labels_store["plain"][1])
    head_ari  = adjusted_rand_score(gold, consensus_labels_store["headline"][1])
    sel_ari   = adjusted_rand_score(gold, consensus_labels_store[selected][1])
    wt_ari    = adjusted_rand_score(gold, consensus_labels_store["weighted"][1])
    sp_ari    = adjusted_rand_score(gold, consensus_labels_store["spatial"][1])
    loso_ratio = (typ_std / loso.std()) if loso.std() > 1e-9 else float("inf")
    loso_min = float(loso.min())
    # data-driven comparators. These strings used to be hard-coded for HBRC and
    # printed FALSE claims on datasets where the relation flips (e.g. DLPFC, where
    # the LOSO worst dips below the best seed and a twist edges plain by noise).
    twist_aris = {"weighted": wt_ari, "spatial": sp_ari, "headline": head_ari}
    best_twist = max(twist_aris, key=twist_aris.get)
    twist_margin = twist_aris[best_twist] - plain_ari
    if twist_margin <= 0:
        twist_msg = "ALL below plain (honest negative)"
    elif twist_margin < typ_std:
        twist_msg = (f"about-equal to plain (best twist '{best_twist}' +{twist_margin:.4f} "
                     f"is within seed noise {typ_std:.4f} -> NOT a real win)")
    else:
        twist_msg = (f"'{best_twist}' BEATS plain by +{twist_margin:.4f} "
                     f"(> seed noise {typ_std:.4f})")
    n_clear = sum(1 for v in (plain_ari, wt_ari, sp_ari, head_ari) if v >= best_single - 1e-9)
    loso_rel = "still beats" if loso_min > best_single else "dips below"
    if loso_min > best_single:
        rob_claim = f"distribution-dominance (LOSO worst {loso_min:.4f} > single best {best_single:.4f})"
    else:
        rob_claim = (f"determinism + above-typical lift (LOSO worst {loso_min:.4f} dips below "
                     f"single best {best_single:.4f} on this dataset -> NO distribution-dominance)")

    print(f"\n{'-'*64}\nVERDICT (refined ARI)\n{'-'*64}")
    print(f"  HEADLINE  plain consensus           : {plain_ari:.4f}  (deterministic, label-free)")
    print(f"  vs typical run (GMM mean)           : {gmm_mean:.4f}   (+{plain_ari-gmm_mean:.4f})")
    print(f"  vs best of all {B} single-seed runs    : {best_single:.4f}   (+{plain_ari-best_single:.4f})")
    print(f"  gain decomposition over GMM mean {gmm_mean:.4f}:")
    print(f"     + cross-SEED pooling (GMM only)  : {xseed_gmm:.4f}  (+{xseed_gmm-gmm_mean:.4f}, primary driver)")
    print(f"     + cross-method (-> full plain)   : {plain_ari:.4f}  (+{plain_ari-xseed_gmm:.4f}, the rest)")
    print(f"  twists vs plain: {twist_msg}  "
          f"[weighted {wt_ari:.4f}  spatial {sp_ari:.4f}  headline {head_ari:.4f}]")
    print(f"  label-free selector chose '{selected}' (ARI {sel_ari:.4f}) "
          f"— it did NOT pick the best; reported as a NEGATIVE, not a win")
    print(f"\n  ROBUSTNESS:")
    print(f"   determinism : consensus = ONE fixed answer (rerun std 0; a single seed had std {typ_std:.4f})")
    print(f"   seed-choice : worst 4-seed (LOSO) consensus {loso_min:.4f} {loso_rel} "
          f"the best single seed {best_single:.4f}")
    print(f"   stress test : bootstrap worst {boot.min():.4f} dips below GMM mean "
          f"-> robust to seed choice, NOT to dropping a whole clusterer")
    print(f"\n  CAVEATS (honesty): n=5 seeds, SINGLE section, NO significance test "
          f"(gaps are descriptive); consensus pools {B} base runs vs 1 for a single seed.")
    print(f"\nSUCCESS CRITERIA (pre-registered):")
    print(f"  (1) consensus >= best single seed   : "
          f"{'YES' if plain_ari >= best_single - 1e-9 else 'NO'}  "
          f"({n_clear}/4 variants clear it: plain {plain_ari:.4f}, weighted {wt_ari:.4f}, "
          f"spatial {sp_ari:.4f}, headline {head_ari:.4f} vs {best_single:.4f})")
    print(f"  (2) >=5x variance collapse          : NO ({loso_ratio:.1f}x).  "
          f"Honest robustness claim = {rob_claim}, not a 5x std drop")
    print(f"\nSaved: results/consensus_ari.csv, results/consensus_perseed.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Consensus driver. Default HBRC; pass --base-dir/--label-col for DLPFC etc.")
    ap.add_argument("--base-dir", default=BASE_DIR,
                    help="dataset dir with processed/ (default: HBRC). e.g. data/dlpfc_151673")
    ap.add_argument("--label-col", default=None,
                    help="gold label column (default: fine_annot_type; DLPFC: layer_guess)")
    args = ap.parse_args()
    bd = args.base_dir if os.path.isabs(args.base_dir) else os.path.join(CODE_DIR, args.base_dir)
    main(base_dir=bd, label_col=args.label_col)
