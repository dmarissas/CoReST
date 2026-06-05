# Figures for Part 2 (Consensus / Robustness). Works on ANY dataset run through
# run_consensus.py: HBRC by default, or pass --base-dir/--label-col (e.g. DLPFC).
# All titles are DATA-DRIVEN (no hard-coded narrative), so they stay honest per
# dataset (e.g. "beats" vs "ties" the best single seed; the actual lift numbers).
#
# Reads ONLY saved run_consensus.py outputs (no retraining, no GPU):
#   results/consensus_ari.csv, consensus_perseed.csv,
#   results/clusters_consensus_plain_k{K}_refined.npy, results/stability_consensus.npy
# Figures -> data/<dataset>/figures/:
#   consensus_vs_lottery.png  — single-seed scatter vs the ONE deterministic consensus
#   consensus_gain.png        — variants bar + gain decomposition
#   consensus_spatial.png     — gold | consensus prediction | per-spot confidence

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(CODE_DIR, "data", "block_a_section1_v110")


def main(base_dir=DEFAULT_BASE, label_col="fine_annot_type"):
    proc = os.path.join(base_dir, "processed")
    res  = os.path.join(base_dir, "results")
    figd = os.path.join(base_dir, "figures")
    os.makedirs(figd, exist_ok=True)
    name = os.path.basename(base_dir)
    print("=" * 60)
    print(f"Part 2 figures — {name}")
    print("=" * 60)

    ari      = pd.read_csv(os.path.join(res, "consensus_ari.csv"))
    perseed  = pd.read_csv(os.path.join(res, "consensus_perseed.csv"))
    barcodes = pd.read_csv(os.path.join(proc, "barcodes_final.csv"))["barcode"].tolist()
    coords   = pd.read_csv(os.path.join(proc, "coords_final.csv"), index_col="barcode")
    labels   = pd.read_csv(os.path.join(proc, "labels_final.csv"), index_col="barcode")
    xy   = coords.loc[barcodes, ["x", "y"]].values
    gold = labels.loc[barcodes, label_col].astype(str).values
    K = len(np.unique(gold))

    def val(approach, col="ARI_mean", refine="refined"):
        r = ari[(ari["approach"] == approach) & (ari["refine"] == refine)]
        return float(r.iloc[0][col]) if len(r) else None

    best_single = max(v for v in [val("single_seed_kmeans", "ARI_best"),
                                  val("single_seed_gmm", "ARI_best"),
                                  val("single_seed_leiden", "ARI_best")] if v is not None)
    plain    = val("consensus_plain")
    gmm_mean = val("single_seed_gmm")
    # Honest, data-driven reference clusterer = the BEST single clusterer by mean
    # refined ARI (GMM on HBRC, KMeans on DLPFC). A hard-coded GMM baseline
    # understates the typical run and misattributes the gain on datasets where
    # GMM is not the best clusterer.
    _cmeans = {m: val(f"single_seed_{m}") for m in ("kmeans", "gmm", "leiden")}
    _cmeans = {m: v for m, v in _cmeans.items() if v is not None}
    best_clust = max(_cmeans, key=_cmeans.get) if _cmeans else "gmm"
    best_clust_mean = _cmeans.get(best_clust, gmm_mean)

    # ── Figure 1: seed lottery vs the deterministic consensus ───────
    print("\n[1] consensus_vs_lottery.png")
    methods  = ["kmeans", "gmm", "leiden"]
    colors_m = {"kmeans": "#95a5a6", "gmm": "#2980b9", "leiden": "#8e44ad"}
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(methods):
        vals = perseed[perseed["method"] == m]["ARI_refined"].values
        if len(vals) == 0:
            continue
        jit = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=80, color=colors_m[m],
                   alpha=0.8, edgecolor="white", zorder=3, label=f"{m} — 5 seeds")
        ax.hlines(vals.mean(), i - 0.18, i + 0.18, color=colors_m[m], lw=2.5, zorder=4)
    lo_w, lo_b = val("consensus_loso", "ARI_worst"), val("consensus_loso", "ARI_best")
    if lo_w is not None and lo_b is not None:
        ax.axhspan(lo_w, lo_b, color="#27ae60", alpha=0.10, zorder=0,
                   label=f"LOSO 4-seed consensus range [{lo_w:.3f}, {lo_b:.3f}]")
    ax.axhline(plain, color="#27ae60", lw=2.6, zorder=2,
               label=f"plain consensus = {plain:.4f}  (deterministic)")
    ax.axhline(best_single, color="#e74c3c", ls="--", lw=1.8, zorder=2,
               label=f"best single seed = {best_single:.4f}")
    ax.axhline(best_clust_mean, color="#7f8c8d", ls=":", lw=1.6, zorder=1,
               label=f"best single-clusterer mean ({best_clust.upper()}) = {best_clust_mean:.4f}")
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylabel(f"ARI (k={K}, refined)")
    rel = "beats" if plain > best_single + 1e-6 else ("ties" if abs(plain - best_single) <= 1e-6 else "below")
    ax.set_title(f"Removing the seed lottery — {name} (k={K})\n"
                 f"consensus = ONE deterministic answer ({plain:.3f}); {rel} best single seed "
                 f"({best_single:.3f}); best single clusterer ({best_clust.upper()}) {best_clust_mean:.3f}",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figd, "consensus_vs_lottery.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: consensus_vs_lottery.png")

    # ── Figure 2: variants bar + gain decomposition ────────────────
    print("\n[2] consensus_gain.png")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))
    labelsA = ["KMeans\n(seed mean)", "GMM\n(seed mean)", "Leiden\n(seed mean)",
               "Consensus\nplain", "Consensus\nweighted", "Consensus\nspatial", "Consensus\nheadline"]
    meansA = [val("single_seed_kmeans"), val("single_seed_gmm"), val("single_seed_leiden"),
              val("consensus_plain"), val("consensus_weighted"), val("consensus_spatial"),
              val("consensus_headline")]
    stdsA = [val("single_seed_kmeans", "ARI_std"), val("single_seed_gmm", "ARI_std"),
             val("single_seed_leiden", "ARI_std"), 0, 0, 0, 0]
    meansA = [(m if m is not None else 0) for m in meansA]
    stdsA  = [(s if s is not None else 0) for s in stdsA]
    colsA = ["#95a5a6", "#2980b9", "#8e44ad", "#27ae60", "#16a085", "#16a085", "#16a085"]
    xA = np.arange(len(labelsA))
    barsA = axL.bar(xA, meansA, yerr=stdsA, capsize=4, color=colsA, alpha=0.9, error_kw={"linewidth": 1.3})
    axL.axhline(best_single, color="#e74c3c", ls="--", lw=1.6, label=f"best single seed = {best_single:.4f}")
    for b, m, sd in zip(barsA, meansA, stdsA):
        axL.text(b.get_x() + b.get_width() / 2, b.get_height() + (sd or 0) + 0.008, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8)
    axL.set_xticks(xA); axL.set_xticklabels(labelsA, fontsize=8)
    axL.set_ylabel(f"ARI (k={K}, refined)"); axL.set_ylim(0, max(meansA) + 0.12)
    twist_vals = {"weighted": val("consensus_weighted"), "spatial": val("consensus_spatial"),
                  "headline": val("consensus_headline")}
    noise = val("single_seed_gmm", "ARI_std") or 0.02   # seed-noise yardstick
    best_twist = max((t for t in twist_vals.values() if t is not None), default=plain)
    tmargin = best_twist - plain
    if tmargin <= 1e-6:
        twist_txt = "twists ≤ plain (don't help)"
    elif tmargin < noise:                                # edges plain only by noise
        twist_txt = "twists ≈ plain (within seed noise)"
    else:
        twist_txt = "a twist > plain"
    cons_rel = ">" if plain > best_single + 1e-6 else ("≈" if abs(plain - best_single) <= 1e-6 else "<")
    axL.set_title(f"Consensus vs single-seed clusterers — {name}\n"
                  f"consensus {cons_rel} best single seed; {twist_txt}")
    axL.legend(fontsize=8, loc="lower left"); axL.grid(axis="y", alpha=0.3)

    bc = best_clust.upper()
    labelsB = ["Ward agg.\n(1 embed)", f"{bc} mean\n(best clusterer)", "+ cross-method\n(1 seed, 3 styles)",
               f"+ cross-seed\n(5 seeds, {bc})", "Full plain\nconsensus"]
    ward = val("ablation_agg_single_embed"); xmethod = val("ablation_singleseed_xmethod")
    xseed_best = val(f"ablation_xseed_{best_clust}")
    meansB = [(m if m is not None else 0) for m in [ward, best_clust_mean, xmethod, xseed_best, plain]]
    colsB = ["#bdc3c7", "#2980b9", "#f39c12", "#e67e22", "#27ae60"]
    xB = np.arange(len(labelsB))
    barsB = axR.bar(xB, meansB, color=colsB, alpha=0.9)
    axR.axhline(best_clust_mean, color="#2980b9", ls=":", lw=1.1, alpha=0.6, zorder=0)  # best-clusterer line
    for b, m in zip(barsB, meansB):
        axR.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=8)
    # gains over the best single clusterer; annotate whichever pooling axis is the
    # real driver (data-driven — cross-seed on both datasets so far, not assumed)
    cs = (xseed_best - best_clust_mean) if xseed_best is not None else 0.0
    cm = (xmethod - best_clust_mean) if xmethod is not None else 0.0
    if cs >= cm and xseed_best is not None:
        axR.annotate(f"{cs:+.3f}\ncross-seed", xy=(3, xseed_best),
                     xytext=(3, xseed_best + 0.045), ha="center", fontsize=9,
                     color="#d35400", fontweight="bold")
    elif xmethod is not None:
        axR.annotate(f"{cm:+.3f}\ncross-method", xy=(2, xmethod),
                     xytext=(2, xmethod + 0.045), ha="center", fontsize=9,
                     color="#d35400", fontweight="bold")
    axR.set_xticks(xB); axR.set_xticklabels(labelsB, fontsize=8)
    axR.set_ylabel(f"ARI (k={K}, refined)"); axR.set_ylim(0, max(meansB) + 0.12)
    axR.set_title(f"Where the lift comes from (vs the best single clusterer, {bc})\n"
                  f"cross-seed {cs:+.3f} vs cross-method {cm:+.3f}")
    axR.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figd, "consensus_gain.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: consensus_gain.png")

    # ── Figure 3: gold | consensus prediction | per-spot confidence ─
    print("\n[3] consensus_spatial.png")
    uniq = sorted(np.unique(gold)); g2i = {l: i for i, l in enumerate(uniq)}
    gi = np.array([g2i[l] for l in gold])
    cmap = plt.cm.get_cmap("tab20" if K > 10 else "tab10", K)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=gi, cmap=cmap, s=6, vmin=0, vmax=K - 1, alpha=0.9)
    axes[0].set_title(f"Gold standard (k={K})", fontsize=12)

    cpath = os.path.join(res, f"clusters_consensus_plain_k{K}_refined.npy")
    if os.path.exists(cpath):
        pred = np.load(cpath)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=pred, cmap=plt.cm.get_cmap("tab20" if K > 10 else "tab10", K),
                        s=6, alpha=0.9)
        axes[1].set_title(f"Plain consensus prediction\n(refined, ARI = {plain:.4f})", fontsize=12)
    else:
        axes[1].text(0.5, 0.5, f"No data\n(run run_consensus.py for {name})", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("Plain consensus prediction", fontsize=12)

    spath = os.path.join(res, "stability_consensus.npy")
    if os.path.exists(spath):
        conf = np.load(spath)[:, 0]
        scat = axes[2].scatter(xy[:, 0], xy[:, 1], c=conf, cmap="magma", s=6, alpha=0.95, vmin=0, vmax=1)
        fig.colorbar(scat, ax=axes[2], fraction=0.046, pad=0.04, label="co-association confidence")
        axes[2].set_title(f"Per-spot consensus confidence\n(label-free; mean = {conf.mean():.3f})", fontsize=12)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Per-spot confidence", fontsize=12)

    for a in axes:
        a.set_aspect("equal"); a.axis("off")
    plt.suptitle(f"Consensus ({name}): gold vs prediction vs label-free per-spot confidence", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(figd, "consensus_spatial.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("    Saved: consensus_spatial.png")
    print(f"\nAll Part-2 figures saved to: {figd}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Part-2 consensus figures. Default HBRC; pass --base-dir/--label-col for DLPFC.")
    ap.add_argument("--base-dir", default=DEFAULT_BASE)
    ap.add_argument("--label-col", default="fine_annot_type")
    a = ap.parse_args()
    bd = a.base_dir if os.path.isabs(a.base_dir) else os.path.join(CODE_DIR, a.base_dir)
    main(bd, a.label_col)
