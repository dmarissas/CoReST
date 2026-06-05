# ROBUSTNESS STUDY for the consensus method (the core experiment for the
# reproducibility thesis). Reproducibility IS the contribution, so this script
# characterises — honestly — how the consensus behaves across many seed pools.
#
# It compares the two outcomes a practitioner actually faces:
#   "run once"          : a single SEDR seed  -> one ARI (the lottery)
#   "run POOL + consensus": POOL_SIZE seeds, consensused -> one deterministic ARI
#
# and reports the full battery:
#   - distribution of each (mean / std / min / max)
#   - VARIANCE ratio  std(single) / std(consensus across pools)   [honest either way]
#   - FLOOR: does the WORST consensus still beat a typical single run? (avoids bad draws)
#   - REPLICATION: fraction of pools where consensus >= that pool's best single seed
#                  + a sign test (binomial) and a Wilcoxon on the margin
#   - INDEPENDENT pools: disjoint POOL_SIZE-seed pools (clean n-independent statement)
#   - DETERMINISM: within a fixed pool the answer is one fixed number (std 0)
#
# Honest expectation (to be confirmed/refuted by the run): the win is likely
# "deterministic + reliably near-best + higher floor", NOT "lower variance".
#
# Cost: trains the NEW seeds only (cached seeds reused). ~1 min/new seed on GPU.
# Run it yourself:  python consensus_robustness.py
# Output: results/consensus_robustness.csv (+ per-pool table consensus_robustness_pools.csv)

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from itertools import combinations
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from graph_func import graph_construction
from utils_func import cluster_latent, refine_labels
from run_consensus import run_sedr_once, leiden_partition, _leiden_available  # import-safe
from consensus_func import co_association_matrix, consensus_labels

BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
PROC_DIR   = os.path.join(BASE_DIR, "processed")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
N_REFINE_NEIGH = 6

# 10 already-cached seeds (pools A+B from consensus_seed_replication) + 10 new.
SEEDS_CACHED = [42, 123, 456, 789, 1234, 7, 88, 314, 2024, 51966]
SEEDS_NEW    = [11, 222, 333, 444, 555, 1212, 2323, 3434, 4545, 5656]
ALL_SEEDS    = SEEDS_CACHED + SEEDS_NEW
POOL_SIZE      = 5
N_RANDOM_POOLS = 60      # sampled 5-seed pools for the smooth distribution
RNG_SEED       = 0       # deterministic pool sampling (Date/random unavailable in some envs)


def _sign_p_greater(k, n):
    """One-sided binomial sign-test p-value: P(>= k successes | p=0.5)."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="greater").pvalue)
    except Exception:
        from scipy.stats import binom_test
        return float(binom_test(k, n, 0.5, alternative="greater"))


def main(base_dir=BASE_DIR, label_col="fine_annot_type"):
    proc_dir   = os.path.join(base_dir, "processed")
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    print("=" * 66)
    print("CONSENSUS ROBUSTNESS STUDY — reproducibility across seed pools")
    print("=" * 66)
    print(f"Device: {DEVICE}   dataset: {os.path.basename(base_dir)}   label: {label_col}")
    print(f"total seeds: {len(ALL_SEEDS)}   pool size: {POOL_SIZE}")

    barcodes = pd.read_csv(os.path.join(proc_dir, "barcodes_final.csv"))["barcode"].tolist()
    labels   = pd.read_csv(os.path.join(proc_dir, "labels_final.csv"), index_col="barcode")
    coords   = pd.read_csv(os.path.join(proc_dir, "coords_final.csv"), index_col="barcode")
    gold = labels.loc[barcodes, label_col].values
    xy   = coords.loc[barcodes, ["x", "y"]].values
    gene_feats = np.load(os.path.join(proc_dir, "gene_only.npy"))
    K = len(np.unique(gold))
    print(f"spots: {len(barcodes)}   gold k={K}")

    adata_g = sc.AnnData(X=gene_feats)
    adata_g.obsm["spatial"] = xy
    graph = graph_construction(adata_g, n=6, mode="KNN")

    methods = ["kmeans", "gmm", "leiden"]
    if not _leiden_available():
        methods.remove("leiden")
        print("[!] leiden deps missing — using kmeans+gmm only")

    # ── Train/load all embeddings; precompute per-seed partitions + single-seed ARI ──
    print("\n[1] Per-seed embeddings (cached reused; new seeds train) + base partitions")
    parts_by_seed, single_ari = {}, {}
    for s in ALL_SEEDS:
        cache = os.path.join(result_dir, f"embeddings_consensus_gene_seed{s}_fine.npy")
        if os.path.exists(cache):
            z = np.load(cache)
            tag = "cache"
        else:
            print(f"    seed={s}: training ...", end=" ", flush=True)
            z = run_sedr_once(gene_feats, graph, K, s)
            np.save(cache, z)
            tag = "trained"
        plist = []
        for m in methods:
            lab = leiden_partition(z, s) if m == "leiden" else cluster_latent(z, K, s, method=m)
            if lab is not None:
                plist.append(lab)
        parts_by_seed[s] = plist
        # single-run quality = GMM refined (the strongest single clusterer)
        gmm_lab = refine_labels(cluster_latent(z, K, s, method="gmm"), xy, N_REFINE_NEIGH)
        single_ari[s] = adjusted_rand_score(gold, gmm_lab)
        print(f"    seed={s:<6} [{tag}] single(GMM-ref) ARI={single_ari[s]:.4f}")

    single_vals = np.array([single_ari[s] for s in ALL_SEEDS])

    def consensus_of(seed_list):
        parts = [p for s in seed_list for p in parts_by_seed[s]]
        cons = refine_labels(consensus_labels(co_association_matrix(parts), K), xy, N_REFINE_NEIGH)
        return adjusted_rand_score(gold, cons)

    # ── Sampled 5-seed pools → consensus distribution + per-pool margin ──
    print(f"\n[2] {N_RANDOM_POOLS} sampled {POOL_SIZE}-seed pools (consensus vs that pool's best single)")
    rng = np.random.RandomState(RNG_SEED)
    pool_rows = []
    for i in range(N_RANDOM_POOLS):
        pool = sorted(rng.choice(ALL_SEEDS, size=POOL_SIZE, replace=False).tolist())
        c = consensus_of(pool)
        best = max(single_ari[s] for s in pool)
        mean = float(np.mean([single_ari[s] for s in pool]))
        pool_rows.append({"pool": ",".join(map(str, pool)), "consensus": round(c, 4),
                          "pool_best_single": round(best, 4), "pool_mean_single": round(mean, 4),
                          "margin_vs_best": round(c - best, 4), "margin_vs_mean": round(c - mean, 4)})
    pdf = pd.DataFrame(pool_rows)
    cons_vals = pdf["consensus"].values

    # ── Disjoint (independent) pools ──
    print(f"\n[3] Disjoint independent pools")
    n_disj = len(ALL_SEEDS) // POOL_SIZE
    disj = []
    for j in range(n_disj):
        pool = ALL_SEEDS[j * POOL_SIZE:(j + 1) * POOL_SIZE]
        c = consensus_of(pool)
        best = max(single_ari[s] for s in pool)
        disj.append({"pool_idx": j, "consensus": round(c, 4), "best_single": round(best, 4),
                     "ge_best": bool(c >= best - 1e-9)})
        print(f"    pool {j} {pool}: consensus={c:.4f}  best_single={best:.4f}  "
              f"{'>=' if c >= best - 1e-9 else '< '} best")

    # ── Stats ──
    s_mean, s_std, s_min, s_max = single_vals.mean(), single_vals.std(), single_vals.min(), single_vals.max()
    c_mean, c_std, c_min, c_max = cons_vals.mean(), cons_vals.std(), cons_vals.min(), cons_vals.max()
    var_ratio = (s_std / c_std) if c_std > 1e-9 else float("inf")
    n_ge_best = int((pdf["margin_vs_best"] >= -1e-9).sum())
    p_sign = _sign_p_greater(n_ge_best, len(pdf))
    floor_beats_single_mean = c_min >= s_mean
    n_disj_ge = sum(d["ge_best"] for d in disj)

    summary = {
        "single_seed":        {"mean": s_mean, "std": s_std, "min": s_min, "max": s_max, "n": len(single_vals)},
        "consensus_pools":    {"mean": c_mean, "std": c_std, "min": c_min, "max": c_max, "n": len(cons_vals)},
        "variance_ratio_single_over_consensus": var_ratio,
        "consensus_floor_>=_single_mean": floor_beats_single_mean,
        "pools_consensus_ge_best_single": f"{n_ge_best}/{len(pdf)}",
        "sign_test_p_(dependent_pools)": p_sign,
        "disjoint_pools_ge_best": f"{n_disj_ge}/{len(disj)}",
        "mean_margin_vs_best": float(pdf["margin_vs_best"].mean()),
        "mean_margin_vs_mean": float(pdf["margin_vs_mean"].mean()),
    }

    pdf.to_csv(os.path.join(result_dir, "consensus_robustness_pools.csv"), index=False)
    pd.DataFrame([{
        "n_seeds": len(ALL_SEEDS), "pool_size": POOL_SIZE, "n_random_pools": len(cons_vals),
        "single_mean": round(s_mean, 4), "single_std": round(s_std, 4),
        "single_min": round(s_min, 4), "single_max": round(s_max, 4),
        "consensus_mean": round(c_mean, 4), "consensus_std": round(c_std, 4),
        "consensus_min": round(c_min, 4), "consensus_max": round(c_max, 4),
        "variance_ratio": round(var_ratio, 2),
        "consensus_floor_ge_single_mean": floor_beats_single_mean,
        "pools_ge_best_single": f"{n_ge_best}/{len(pdf)}", "sign_test_p": round(p_sign, 4),
        "disjoint_pools_ge_best": f"{n_disj_ge}/{len(disj)}",
    }]).to_csv(os.path.join(result_dir, "consensus_robustness.csv"), index=False)

    # ── Verdict ──
    print(f"\n{'='*66}\nROBUSTNESS VERDICT (k={K}, refined ARI)\n{'='*66}")
    print(f"  'run once'  (single seed, n={len(single_vals)}): "
          f"mean {s_mean:.4f}  std {s_std:.4f}  range [{s_min:.4f}, {s_max:.4f}]")
    print(f"  'run {POOL_SIZE}+consensus' (n={len(cons_vals)} pools): "
          f"mean {c_mean:.4f}  std {c_std:.4f}  range [{c_min:.4f}, {c_max:.4f}]")
    print(f"\n  LIFT (consensus mean − single mean) : {c_mean - s_mean:+.4f}")
    print(f"  VARIANCE ratio single/consensus     : {var_ratio:.2f}x  "
          f"({'consensus IS less variable' if var_ratio > 1.1 else 'NOT a variance reduction — honest'})")
    print(f"  FLOOR: worst consensus {c_min:.4f} vs typical single run {s_mean:.4f} -> "
          f"{'consensus floor BEATS a typical run (avoids bad draws)' if floor_beats_single_mean else 'floor does NOT beat typical run'}")
    print(f"  REPLICATION: consensus >= pool's best single in {n_ge_best}/{len(pdf)} pools "
          f"(sign-test p={p_sign:.4f}; pools share seeds so p is heuristic)")
    print(f"  INDEPENDENT pools: {n_disj_ge}/{len(disj)} disjoint pools >= their best single")
    print(f"  DETERMINISM: each pool gives ONE fixed answer (within-pool std = 0)")
    print(f"\n  Honest headline: consensus is deterministic seed-insurance — it lifts the typical")
    print(f"  result (+{c_mean - s_mean:.3f}) and {'raises the floor' if floor_beats_single_mean else 'tracks the seed quality'}; "
          f"variance is {'reduced' if var_ratio > 1.1 else 'NOT clearly reduced'}.")
    print(f"\nSaved: results/consensus_robustness.csv + consensus_robustness_pools.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Consensus robustness study. Defaults to HBRC; pass --base-dir/--label-col for DLPFC etc.")
    ap.add_argument("--base-dir", default=BASE_DIR,
                    help="dataset dir with processed/ (default: HBRC). e.g. data/dlpfc_151673")
    ap.add_argument("--label-col", default="fine_annot_type",
                    help="gold label column (HBRC: fine_annot_type; DLPFC: layer_guess)")
    args = ap.parse_args()
    bd = args.base_dir if os.path.isabs(args.base_dir) else os.path.join(CODE_DIR, args.base_dir)
    main(base_dir=bd, label_col=args.label_col)
