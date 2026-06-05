# Part 2 ROBUSTNESS figure (from consensus_robustness.py results).
#
# Works on ANY dataset run through consensus_robustness.py: HBRC by default, or
# pass --base-dir/--label-col (e.g. DLPFC). All titles are DATA-DRIVEN — no
# hard-coded narrative — so they stay honest per dataset (e.g. "tighter" vs
# "WIDER", "ties" vs "BEATS" the best-of-pool).
#
# Panel A: "run once" (single-seed ARIs) vs "run 5 + consensus" (pools) — lift + spread.
# Panel B: the margins — consensus vs a TYPICAL run, and vs the BEST-of-pool.
# Reads results/consensus_robustness_pools.csv + recomputes single-seed ARIs from
# the cached embeddings (no SEDR training). Style matches the other viz scripts.

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from utils_func import cluster_latent, refine_labels

N_REFINE_NEIGH = 6
DEFAULT_BASE = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
# Must match consensus_robustness.py (the 20 seeds it used).
SEEDS_ALL = [42, 123, 456, 789, 1234, 7, 88, 314, 2024, 51966,
             11, 222, 333, 444, 555, 1212, 2323, 3434, 4545, 5656]


def main(base_dir=DEFAULT_BASE, label_col="fine_annot_type"):
    proc = os.path.join(base_dir, "processed")
    res  = os.path.join(base_dir, "results")
    figd = os.path.join(base_dir, "figures")
    os.makedirs(figd, exist_ok=True)
    name = os.path.basename(base_dir)
    print("=" * 60)
    print(f"robustness figure — {name}")
    print("=" * 60)

    ppath = os.path.join(res, "consensus_robustness_pools.csv")
    if not os.path.exists(ppath):
        print(f"MISSING {ppath} — run consensus_robustness.py for this dataset first.")
        sys.exit(1)
    pools = pd.read_csv(ppath)
    cons   = pools["consensus"].values
    m_mean = pools["margin_vs_mean"].values
    m_best = pools["margin_vs_best"].values

    barcodes = pd.read_csv(os.path.join(proc, "barcodes_final.csv"))["barcode"].tolist()
    gold = pd.read_csv(os.path.join(proc, "labels_final.csv"), index_col="barcode").loc[barcodes, label_col].values
    xy   = pd.read_csv(os.path.join(proc, "coords_final.csv"), index_col="barcode").loc[barcodes, ["x", "y"]].values
    K = len(np.unique(gold))

    print("recomputing single-seed ARIs (GMM, refined) from cache ...")
    singles = []
    for s in SEEDS_ALL:
        z = np.load(os.path.join(res, f"embeddings_consensus_gene_seed{s}_fine.npy"))
        singles.append(adjusted_rand_score(
            gold, refine_labels(cluster_latent(z, K, s, method="gmm"), xy, N_REFINE_NEIGH)))
    singles = np.array(singles)

    s_mean, s_std = singles.mean(), singles.std()
    c_mean, c_std = cons.mean(), cons.std()
    var_ratio = s_std / c_std if c_std > 1e-9 else float("inf")
    N = len(cons)
    n_vs_mean = int((m_mean > 0).sum())
    n_vs_best = int((m_best >= -1e-9).sum())
    tighter = var_ratio > 1.05
    beats_best = n_vs_best > 0.6 * N
    print(f"  single {s_mean:.4f}±{s_std:.4f}   consensus {c_mean:.4f}±{c_std:.4f}   "
          f"var {var_ratio:.2f}x   vs_mean>0 {n_vs_mean}/{N}   vs_best>=0 {n_vs_best}/{N}")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))
    rng = np.random.RandomState(0)

    # Panel A — the two clouds
    for xc, vals, color, lab in [
            (0, singles, "#95a5a6", f"run once — single seed (n={len(singles)}):  {s_mean:.3f} ± {s_std:.3f}"),
            (1, cons, "#27ae60", f"run 5 + consensus (n={N} pools):  {c_mean:.3f} ± {c_std:.3f}")]:
        jit = rng.uniform(-0.10, 0.10, size=len(vals))
        axA.scatter(np.full(len(vals), xc) + jit, vals, s=45, color=color, alpha=0.7,
                    edgecolor="white", linewidth=0.4, zorder=3, label=lab)
        axA.errorbar(xc, vals.mean(), yerr=vals.std(), fmt="_", color="black", markersize=46,
                     elinewidth=2, capsize=10, capthick=2, zorder=4)
    axA.annotate("", xy=(1, c_mean), xytext=(0, s_mean),
                 arrowprops=dict(arrowstyle="->", color="#1e8449", lw=2))
    axA.text(0.5, (s_mean + c_mean) / 2 - 0.02, f"lift +{c_mean - s_mean:.3f}",
             ha="center", color="#1e8449", fontsize=11, fontweight="bold")
    axA.set_xlim(-0.5, 1.5); axA.set_xticks([0, 1])
    axA.set_xticklabels(["Run once\n(single seed)", "Run 5 + consensus"])
    axA.set_ylim(min(singles.min(), cons.min()) - 0.03, max(singles.max(), cons.max()) + 0.03)
    axA.set_ylabel(f"ARI (k={K}, refined)")
    axA.set_title(f"Run once vs run-5-and-consensus\n"
                  f"consensus is higher (+{c_mean - s_mean:.3f}), "
                  f"{'tighter' if tighter else 'WIDER'} ({var_ratio:.1f}×), and deterministic")
    axA.grid(axis="y", alpha=0.3); axA.legend(fontsize=8, loc="lower right")

    # Panel B — the margins (vs typical run, vs best-of-pool)
    lo = min(m_best.min(), m_mean.min()) - 0.005
    hi = max(m_best.max(), m_mean.max()) + 0.005
    bins = np.linspace(lo, hi, 22)
    axB.hist(m_mean, bins=bins, color="#27ae60", alpha=0.65,
             label=f"vs a TYPICAL run  ({n_vs_mean}/{N} above 0)")
    axB.hist(m_best, bins=bins, color="#e67e22", alpha=0.65,
             label=f"vs the BEST-of-pool  ({n_vs_best}/{N} ≥ 0)")
    axB.axvline(0, color="black", ls="--", lw=1.5)
    axB.axvline(m_mean.mean(), color="#1e8449", lw=2, label=f"mean vs typical = +{m_mean.mean():.3f}")
    axB.axvline(m_best.mean(), color="#b9770e", lw=2, label=f"mean vs best = {m_best.mean():+.3f}")
    axB.set_xlabel("consensus ARI − single-seed ARI")
    axB.set_ylabel(f"number of pools (of {N})")
    axB.set_title(f"Consensus beats a TYPICAL run; {'BEATS' if beats_best else 'ties'} the best-of-pool\n"
                  f"(vs typical {n_vs_mean}/{N} above 0; vs best {n_vs_best}/{N} ≥ 0)")
    axB.legend(fontsize=8, loc="upper left"); axB.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Consensus robustness — {len(SEEDS_ALL)} seeds, {N} sampled 5-seed pools "
                 f"({name}, k={K})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(figd, "consensus_robustness.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Robustness figure. Default HBRC; pass --base-dir/--label-col for DLPFC etc.")
    ap.add_argument("--base-dir", default=DEFAULT_BASE)
    ap.add_argument("--label-col", default="fine_annot_type")
    a = ap.parse_args()
    bd = a.base_dir if os.path.isabs(a.base_dir) else os.path.join(CODE_DIR, a.base_dir)
    main(bd, a.label_col)
