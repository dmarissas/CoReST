# GateST: Gated Multimodal Fusion for Spatial Transcriptomics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Spatial tissue-domain identification on breast-cancer Visium data — two honest studies: (1) gated multimodal fusion of UNI histology + gene expression, and (2) a consensus/robustness method that removes SEDR's random-seed lottery.

---

## Two studies in this repo

Two connected investigations on the same SEDR pipeline and dataset:

- **Part 1 — Gated multimodal fusion** *(histology + genes)* — an **honest negative**: *where* you fuse matters (feature fusion hurts, graph fusion ties), but UNI histology adds **no robust gain** over genes on this section. → [Overview](#overview), [Results](#results).
- **Part 2 — Consensus / robustness** *(current direction)* — a deterministic cross-seed + cross-method **consensus** of SEDR embeddings that **tames the random-seed lottery**. Across 20 seeds it **lifts a typical single run by +0.037** (0.549 → 0.586) with **~1.4× lower variance**, deterministically and **label-free** (≈ the best-of-pool result, which you otherwise couldn't pick without the answer key). Honest scope: it does **not** reliably beat the *best* single seed (≈ coin flip) and the absolute ARI is pool-dependent — the value is **reproducible, above-typical, label-free** performance, not a guaranteed win. → [Part 2 — Consensus / Robustness](#part-2--consensus--robustness).

---

## Overview

Spatial transcriptomics methods typically rely on gene expression alone for tissue domain identification. GateST proposes a **gated multimodal fusion strategy** that adaptively combines:

- **Gene expression features** — 200d PCA from 2000 highly variable genes (standard SEDR preprocessing)
- **Histology image features** — multi-scale UNI embeddings (3×1024d → 256d PCA) from H&E-stained tissue patches

The fused representation feeds into **SEDR** (Spatial Embedding by Deep learning with Regularization), a VAE + GNN model that exploits spatial neighborhood structure for unsupervised domain clustering. GateST compares gating at two levels: a per-spot **feature-space gate**, and a per-edge **graph-level gate** that lets histology shape SEDR's spatial graph instead of its node features. The finding: *where* you fuse matters (feature fusion hurts, graph fusion doesn't), but the histology modality adds no robust accuracy gain over gene expression on this benchmark.

### Key finding

**Where you fuse the modalities matters — but the histology modality adds no robust gain here.**

1. **In the feature space, histology hurts.** Naive concatenation of gene + image features falls below gene-only, and a learned per-spot *feature* gate, while better than concat, still does not reach gene-only. The weak, noisy image dimensions dilute SEDR's dominant reconstruction signal. This is robust across both clusterers (KMeans and GMM).

2. **In the graph, histology is harmless — it ties gene-only.** Moving the gate to the **graph edges** (keep a spatial connection only when two spots are *also* morphologically similar — an **image-gated spatial graph**) keeps node features pure-gene, so nothing is diluted. This *recovers* gene-only performance, and the two are statistically tied: the image-gated graph **wins under KMeans** (refined ARI 0.5459 vs 0.5237) but is **marginally behind under the stronger GMM clusterer** (0.5711 vs 0.5746), a difference well within seed noise.

**Honest bottom line:** on this single section, UNI histology provides **no robust improvement** over gene expression for tissue-domain identification, regardless of fusion strategy or clusterer. The contribution is methodological — *graph-level* fusion preserves the strong gene signal (does no harm) whereas *feature-level* fusion degrades it — plus an honest negative result on image utility.

> The image-gated graph's KMeans-only edge is verified to be image content (not graph sparsity): a **density-matched** spatial graph scores only 0.5178, and a **shuffled-image** control collapses cluster quality (silhouette 0.288 → 0.237). But that edge does not survive switching to the stronger GMM clusterer, hence the "tie" conclusion.
>
> Results are mean ± std over 5 SEDR seeds. Spatial label refinement (KNN majority vote) and both clusterers are applied identically to every condition, so all comparisons are fair.

---

## Results

Performance on HBRC Block A Section 1 (Xu et al. gold standard, 20 fine-grained domains):

Fine-grained ARI (k=20), SEDR latent + spatial refinement, mean over 5 seeds. **Both clusterers are shown because the gene-vs-graph ranking flips between them.**

| Condition (features → graph) | KMeans (refined) | GMM (refined) |
|--------|--------|--------|
| Image only → spatial | 0.2745 | 0.3019 |
| Concat fusion → spatial | 0.4687 | 0.5224 |
| Feature gate → spatial | 0.5107 | 0.5363 |
| Gene only → spatial | 0.5237 | **0.5746** |
| **Gene → image-gated graph** | **0.5459** | 0.5711 |

Published single-run baselines (clusterer / refinement unknown): SEDR 0.3668 · Seurat 0.4612 · STAGATE 0.4944 · TGR-NMF 0.5286.

**Reading the table honestly:**
- **Feature fusion (concat, feature-gate) stays below gene-only under *both* clusterers** — fusing in the feature space hurts. This is robust.
- **The image-gated graph ties gene-only** — it wins under KMeans (+0.022) but is marginally behind under the stronger GMM clusterer (−0.004, within seed noise ≈ 0.02). So histology gives **no robust ARI gain** here.
- Our refined+GMM numbers (≈0.57) are not directly comparable to the published single-run baselines (different pipelines); shown for rough context only.

Controls (gene features fixed; only the graph changes) — these confirm the *KMeans-level* effect is genuine image content, even though it does not survive GMM:

| Control | fine ARI (KMeans, refined) | reading |
|---|---|---|
| Image-gated graph (intersect) | 0.5459 | the effect |
| Density-matched spatial graph (k=3, equal sparsity) | 0.5178 | rules out "just fewer edges" |
| Shuffled-image graph | silhouette 0.237 (vs 0.288) | rules out artifact; image content matters |

### Is the bottleneck the features or the modality?

To check whether better image *features* could change the picture, we measured image-only ARI across representations built from the raw multi-scale UNI embeddings (different PCA dims; each scale alone). Image-only fine ARI (GMM, refined):

| Image representation | GMM (refined) |
|---|---|
| all scales, PCA 256 (baseline) | 0.3143 |
| all scales, PCA 512 | 0.3340 |
| scale 1 (1×, cellular) | 0.2949 |
| scale 2 (2×) | 0.3061 |
| **scale 3 (3×, tissue-context)** | **0.3363** |
| *(gene-only ceiling)* | *0.5746* |

Every representation caps at **~0.30–0.34**, a ~0.24 gap below gene-only that no extraction choice closes. **The bottleneck is the modality, not the pipeline:** UNI histology lacks the fine-domain signal that gene expression carries on this section. (Two minor positives: the tissue-context scale (3×) carries more domain signal than the cellular scale (1×), as expected, and keeping more PCA dimensions helps marginally.)

### Visualizations

**ARI comparison across feature conditions (KMeans baseline vs SEDR):**

![Bar Plot](figures/results_barplot.png)

**Gold standard tissue domains vs GateST image-gated-graph prediction:**

![Comparison Map](figures/comparison_map.png)

**t-SNE of SEDR embeddings colored by gold standard (20 fine-grained domains):**

![t-SNE](figures/tsne_embeddings.png)

**Predicted spatial clusters across all conditions (k=20):**

![Spatial Clusters k20](figures/spatial_clusters_k20.png)

---

## Pipeline

```
Step 1: Gene feature extraction
  └── Visium .h5 → 2000 HVGs → PCA 200d

Step 2: Image feature extraction
  └── H&E TIFF → UNI (multi-scale 3×1024d) → PCA 256d

Step 3: Feature variants
  ├── gene_only      (200d) — baseline
  ├── image_only     (256d) — ablation
  ├── concat_fused   (456d) — naive z-score concatenation
  └── gated_fused    (128d) — learned feature-space gate

Step 4: SEDR training + evaluation
  ├── spatial k=6 KNN graph → VAE + GNN → {KMeans, GMM} → refine → ARI
  └── gene → IMAGE-GATED graph (intersect of spatial-KNN & image-KNN)  ← graph-level fusion
      (controls: density-matched spatial-k, shuffled-image — see experiment_graph.py)

Step 5: Visualization
  └── Bar plots, spatial cluster maps, comparison map, t-SNE
```

---

## Installation

```bash
git clone https://github.com/dmarissas/GateST.git
cd GateST
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- torch-geometric
- scanpy
- timm
- torchstain
- scikit-learn
- huggingface_hub

---

## Data

This project uses the **10x Visium Human Breast Cancer Block A Section 1** dataset.

**Download from 10x Genomics** ([dataset page](https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0)):
- **Filtered feature barcode matrix (.h5)** — `V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5`
- **High-resolution tissue image (.tif)** — `V1_Breast_Cancer_Block_A_Section_1_image.tif`
- **Spatial imaging data (.tar.gz)** — this archive contains the `spatial/` folder. Extract it and you will find `tissue_positions_list.csv` and `scalefactors_json.json` (among other files) inside.

**Gold standard annotations:**
- `metadata.tsv` from the SEDR analyses repo — [JinmiaoChenLab/SEDR_analyses → data/BRCA1/metadata.tsv](https://github.com/JinmiaoChenLab/SEDR_analyses/blob/master/data/BRCA1/metadata.tsv) — providing the 20 fine-grained spatial domain labels for Block A Section 1 (originally from [Xu et al. 2022](https://doi.org/10.1038/s41592-022-01494-7)).

Place data in the following structure:

```
GateST/
└── data/
    └── block_a_section1_v110/
        ├── V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5
        ├── V1_Breast_Cancer_Block_A_Section_1_image.tif
        ├── metadata.tsv
        └── spatial/
            ├── tissue_positions_list.csv
            └── scalefactors_json.json
```

**UNI model access:**
UNI requires HuggingFace access approval from MahmoodLab.
Request access at: https://huggingface.co/MahmoodLab/uni

---

## Usage

```bash
# Step 1: Extract gene features + coordinates + labels (~2 min)
python gene_features.py

# Step 2: Extract UNI image features (~15 min, GPU recommended)
python image_features.py

# Step 3: Compute all fusion variants + train gate network (~5 min)
python prepare_features.py

# Step 4: Train SEDR on all conditions (~10-12 min: 5 conditions x 5 seeds x 2
# resolutions, evaluated with KMeans+GMM, raw+refined)
python train_sedr.py

# Step 4b: Regenerate per-spot cluster assignments from the saved embeddings (~1 min)
# REQUIRED before Step 5 — train_sedr.py saves only embeddings + ARI scores,
# while visualize.py reads the clusters_<condition>_{k20,k4}.npy files produced here.
python cluster_regeneration.py

# Step 5: Generate all figures (~5 min)
python visualize.py
```

> **Part 2 (Consensus / Robustness)** has its own entry point, independent of Steps 4b–5:
> ```bash
> python run_consensus.py     # ~6 min first run; ~1 min on reruns (embeddings cached)
> ```
> See the [Consensus / Robustness](#part-2--consensus--robustness) section below.

---

## Experiments & Ablations

Beyond the 5-step pipeline, the repo includes the experiment/ablation scripts that produced the results and controls in this README. Run them in the `GateST` environment after Steps 1–3 have populated `processed/`.

### Main comparison — `train_sedr.py`
```bash
python train_sedr.py        # ~10-12 min
```
Trains SEDR for **5 conditions** — `gene_only`, `image_only`, `concat_fused`, `gated_fused` (spatial graph), and `gene_imagegraph` (gene features on the **image-gated** "intersect" graph) — and evaluates each with **{KMeans, GMM} × {raw, refined}** over 5 seeds. Builds both the spatial (k=6 KNN) and image-gated graphs internally.
**Output:** `results/ari_results.csv` (long format: `condition, graph_mode, model, resolution, cluster_method, refine, ARI_mean, ARI_std, n_seeds`). **Read:** compare each condition's `cluster_method` × `refine` cells (this is the source of the Results table — feature fusion < gene-only; image-gated graph ties gene-only).

### Re-cluster saved embeddings — `cluster_regeneration.py`
```bash
python cluster_regeneration.py
```
Re-clusters the saved `embeddings_<cond>.npy` with KMeans **and** GMM, raw **and** refined (spatial KNN majority vote). **Required before `visualize.py`** (it writes the `clusters_<cond>_{k20,k4}[_refined].npy` the figures read). **Output:** `results/cluster_regeneration_ari.csv` (per-condition raw vs refined, KMeans vs GMM).

### Image-gated-graph study + controls — `experiment_graph.py`
```bash
python experiment_graph.py  # ~6 min
```
Holds node features = gene_only and varies only the SEDR **graph**. Modes: `spatial` (baseline anchor — must reproduce gene-only), `reweight_b*`, `blend_a*`, `union`, `intersect`, plus the **controls**: density-matched `spatial_k3/k4/k5` (no image) and `intersect_SHUFFLE` (scrambled image), with a paired per-seed comparison.
**Output:** `results/graph_experiment_ari.csv` + `graph_experiment_perseed.csv`.
**How to read the verdict:** compare `intersect` to the **same-density** `spatial_kN` (≈ deg 4.4 → `spatial_k3`) — if `intersect` beats it, the *image edge-selection* helps (not just sparsity); the `intersect_SHUFFLE` silhouette collapse (0.288→0.237) confirms it's image *content*. Note this effect appears under KMeans but does not survive GMM (see Results).

### Image-representation diagnostic — `image_feature_study.py`
```bash
python image_features.py        # re-run once: also saves image_features_raw3072.npy
python image_feature_study.py   # ~6 min
```
Tests whether *better image features* would change the conclusion: builds image-only representations from the raw UNI embeddings (PCA dims {128, 256, 512} and each scale {1×, 2×, 3×} alone) and reports image-only fine ARI (KMeans + GMM, refined).
**Output:** `results/image_feature_study.csv`. **Read:** every representation caps at ~0.30–0.34 vs gene-only's ~0.57 — the evidence that the limit is the **modality, not the features** (tissue-context 3× > cellular 1× is a minor positive).

### Figures — `visualize.py`
```bash
python visualize.py
```
**Output (`figures/`):** `results_barplot.png` (bar chart, **KMeans vs GMM** so the clusterer-dependent ranking is visible), `comparison_map.png` (gold vs image-gated-graph), `spatial_clusters_{k20,k4}.png`, `gold_standard_map.png`, `tsne_embeddings.png`.

### Consensus / robustness — `run_consensus.py` (Part 2)
```bash
python run_consensus.py     # ~6 min first run; ~1 min on reruns (embeddings cached)
```
Builds 15 base partitions (5 seeds × {KMeans, GMM, Leiden} of the gene-only SEDR embedding), the co-association matrix, and the 4 consensus variants; then runs the robustness checks (leave-one-seed-out, bootstrap, all 3/4-seed subsets), the per-spot stability map, and the **source-of-gain ablations** (clusterer-alone / cross-method-only / cross-seed-only + an identity check). Picks the headline variant by a label-free silhouette criterion. Caches per-seed embeddings (`embeddings_consensus_gene_seed*_fine.npy`) so reruns skip SEDR training. Needs `leidenalg`+`igraph` (falls back to KMeans+GMM if missing); the consensus math also has a standalone self-test: `python consensus_func.py`.
**Output:** `results/consensus_ari.csv` (long format: single-seed / consensus variants / LOSO / bootstrap / seed-subset / ablations), `consensus_perseed.csv`, `stability_consensus.npy`. **Read:** the printed **VERDICT** block — consensus lifts the typical single run and is deterministic (seed-insurance ≈ best-of-pool, label-free); cross-seed pooling is the driver; the twists are honest negatives. (It does *not* reliably beat the *best* seed and ARI is pool-dependent — see `consensus_robustness.py` / Part 2.)

### Seed-pool replication — `consensus_seed_replication.py` (Part 2)
```bash
python consensus_seed_replication.py    # trains 5 NEW seeds (~5-6 min GPU); pools A / A+B reuse cache
```
The key robustness ablation: rebuilds the consensus on a **disjoint** seed pool B `{7,88,314,2024,51966}` and on the 10-seed A+B, and compares to the original pool A. This is what revealed **0.6504 is a lucky pool** (pool B → 0.6202; 10-seed → 0.6132; ~0.61 typical) and reframed the result as *seed-insurance*. **Output:** `results/consensus_seed_replication.csv`. **Read:** the `|consensus_A − consensus_B|` replication check and each pool's `consensus_minus_best`.

### Robustness study — `consensus_robustness.py` (Part 2, the definitive test)
```bash
python consensus_robustness.py    # trains 10 NEW seeds (~10-15 min GPU; 10 cached reused)
```
Scales to **20 seeds**, samples **60 random 5-seed pools** + disjoint pools, and compares the *distribution* a practitioner faces — "run once" (single seed) vs "run 5 + consensus" — reporting lift, variance ratio, floor, the **replication rate** (fraction of pools where consensus ≥ that pool's best single) + a sign test, and determinism. **Output:** `results/consensus_robustness.csv` (+ `consensus_robustness_pools.csv`). **Read:** the VERDICT — consensus lifts the typical run +0.037 with ~1.4× lower variance (deterministic), but ≈ matches the best-of-pool rather than beating it (~48% of pools).

### Part-2 + ablation figures — `visualize_consensus.py`, `visualize_ablations.py`, `visualize_robustness.py`
```bash
python visualize_ablations.py    # graph_experiment.png, image_feature_study.png   (Part 1 ablations)
python visualize_consensus.py    # consensus_vs_lottery.png, consensus_gain.png, consensus_spatial.png   (Part 2)
python visualize_robustness.py   # consensus_robustness.png   (Part 2 — the 20-seed reproducibility figure)
```
All read **saved CSVs / `.npy`** (no retraining; `visualize_robustness.py` recomputes the 20 single-seed ARIs from cached embeddings, ~1–2 min). `visualize_ablations.py` turns the graph-experiment and image-study tables into figures; `visualize_consensus.py` plots the seed-lottery scatter, the gain decomposition, and the per-spot stability map; `visualize_robustness.py` plots the "run once vs run-5-and-consensus" clouds + the margin histograms (the reproducibility centerpiece). **Output:** the six PNGs above in `figures/`.

---

## Project Structure

```
GateST/
├── gene_features.py        # Step 1: Gene PCA extraction + labels + coords
├── image_features.py       # Step 2: Multi-scale UNI image embedding
├── prepare_features.py     # Step 3: Gated fusion + all feature variants
├── train_sedr.py           # Step 4: SEDR training + ARI eval (incl. image-gated graph)
├── cluster_regeneration.py # Step 4b: Regenerate cluster files from embeddings (required for Step 5)
├── visualize.py            # Step 5: Figure generation
├── experiment_graph.py     # Image-gated-graph experiment + controls (density-matched, shuffle)
├── image_feature_study.py  # Diagnostic: image-only ARI by representation (per-scale, PCA dims)
├── consensus_func.py       # Part 2: pure co-association / consensus / stability functions (+ self-test)
├── run_consensus.py        # Part 2: consensus driver (base partitions → co-association → variants → ablations)
├── consensus_seed_replication.py # Part 2: seed-pool ablation (disjoint pool B + A∪B) — showed 0.6504 was a lucky pool
├── consensus_robustness.py # Part 2: 20-seed robustness study (many pools → lift / variance / replication + sign test)
├── visualize_consensus.py  # Part 2 figures (seed-lottery scatter, gain decomposition, stability map)
├── visualize_ablations.py  # Part 1 ablation figures (graph experiment, image-representation study)
├── visualize_robustness.py # Part 2: 20-seed reproducibility figure (run-once vs consensus clouds + margins)
├── SEDR_model.py           # SEDR training loop (VAE + GNN wrapper)
├── graph_func.py           # Spatial + image-gated graph construction
├── utils_func.py           # Preprocessing, clustering (KMeans/GMM), spatial refinement
├── requirements.txt
└── README.md
```

---

## Two gating mechanisms

**(A) Feature-space gate** (`prepare_features.py`) — a per-spot scalar gate that mixes the two modalities *before* SEDR:

```python
g = sigmoid(W_gate * concat([gene_features, image_features]))   # per-spot gate in (0,1)
h_gene, h_image = ReLU(W_gene * gene), ReLU(W_image * image)
h_fused = g * h_gene + (1 - g) * h_image
loss    = recon_gene + 0.5 * recon_image - 0.3 * entropy(g)
```

This recovers most of the concatenation loss but does not beat gene-only — the gate collapses to a near-constant ~0.52, and the image still enters SEDR's reconstruction target where it dilutes the gene signal.

**(B) Graph-level gate** (`graph_func.py`, `graph_construction_fused`) — **the better-behaved fusion.** Instead of mixing *features*, the gate acts on the *edges* of SEDR's spatial graph. A spatial connection is kept only when the two spots are also morphologically similar in UNI-embedding space:

```python
# per-edge gate: keep edge (u,v) iff spots are spatial neighbours AND look alike
A_fused[u, v] = A_spatial[u, v]  AND  A_image_knn[u, v]      # "intersect" mode
# node features fed to SEDR stay PURE GENE — never diluted by image noise
```

Because histology only reshapes *which spots smooth toward each other* (not what they contain), it adds morphological information without harming the strong gene signal — so it *ties* gene-only rather than degrading it like feature fusion does (see Results; the small KMeans-level edge does not survive GMM).

---

## Why feature fusion hurts but graph fusion doesn't

Feature-space fusion drags performance below gene-only; graph-level fusion instead *preserves* it (a statistical tie). Fine-grained refined ARI:

```
                                   KMeans   GMM
Image only   → spatial graph:      0.2745   0.3019   ← weak on its own
Concat        → spatial graph:     0.4687   0.5224   ← gluing features HURTS
Feature gate  → spatial graph:     0.5107   0.5363   ← still < gene-only
Gene only     → spatial graph:     0.5237   0.5746   ← strong baseline
Gene          → image-gated graph: 0.5459   0.5711   ← ties gene-only (wins KMeans, loses GMM)
```

The reason: SEDR's reconstruction loss (`rec_w=10`) dominates training. Putting image *features* into the input forces SEDR to reconstruct 256 noisy image dimensions, diluting genes (concat/feature-gate fall below gene-only). Putting image into the *graph* leaves the reconstruction target pure-gene and only changes message passing — so it does no harm and matches gene-only. But the extra morphological signal is not strong enough to *beat* gene expression once a capable clusterer (GMM) is used: the histology modality carries little domain-discriminative information beyond genes on this section.

---

## Part 2 — Consensus / Robustness

> **Different question.** Part 1 asked *"can histology help?"* (no). Part 2 asks: *can we remove SEDR's random-seed lottery and get one reproducible answer that is reliably as good as the best single run?* (yes — and note "as good as," not "better than"; see [Seed-pool replication](#seed-pool-replication) for why.)

### The gap

SEDR is **seed-sensitive** — the same model on the same data swings fine ARI by ~0.06 across random seeds (gene-only, GMM, refined):

| Seed | 42 | 123 | 456 | 789 | 1234 |
|---|---|---|---|---|---|
| ARI | 0.6052 | 0.5684 | 0.5819 | 0.5429 | 0.5743 |

A practitioner who runs **once** gets a random draw. We remove that lottery with **partition-level consensus** (evidence-accumulation clustering, Fred & Jain 2005 — *applied* here, not reinvented).

### Method

Base partitions = **5 seeds × {KMeans, GMM, Leiden}** of the gene-only SEDR embedding (32d; gene features + spatial graph; **no histology**) → 15 partitions → a **co-association matrix** `C` (how often each spot-pair is grouped together) → deterministic **average-linkage** cut at k=gold. Averaging *agreements* is valid where averaging *embeddings* is not (each seed's latent axes are arbitrary; co-grouping is frame-invariant). Files: `consensus_func.py` (pure functions + self-test), `run_consensus.py` (driver).

### Results (fine k=20, refined ARI)

| Approach | ARI | vs best single seed (0.6052) |
|---|---|---|
| Typical single run (GMM mean) | 0.5745 ± 0.0202 | — |
| Best of all 15 single-seed runs | 0.6052 | — |
| **Plain consensus** | **0.6504** | **+0.045** (deterministic) |
| consensus + quality weight | 0.6405 | twist — *below* plain |
| consensus + spatial reg. | 0.6383 | twist — *below* plain |
| consensus + both (pre-reg. headline) | 0.6355 | twist — *below* plain |

The consensus lands **at or above the luckiest of 15** single-seed runs and is **deterministic** (same answer every run). ⚠️ **But 0.6504 is pool-dependent — do not read it as "the number."** On an independent seed pool the consensus is 0.6202, and the honest point estimate is the 10-seed value **0.6132**; the margin over the best single seed is within seed noise. See [Seed-pool replication](#seed-pool-replication) below — the honest claim is *variance reduction*, not a meaningful ARI gain.

### Where the gain comes from (ablations)

| Source | ARI | adds |
|---|---|---|
| Ward agglomerative on one embedding (clusterer alone) | 0.5518 ± 0.0349 | nothing special (≈ a single run) |
| Typical GMM run | 0.5745 | — |
| Cross-method only (1 seed, 3 styles) | 0.5843 ± 0.0209 | +0.010 |
| **Cross-seed only (5 seeds, GMM)** | **0.6294** | **+0.055 ← primary driver** |
| Full plain consensus | 0.6504 | +0.021 more |

An **identity check** confirms the consensus operator returns a single partition unchanged (ARI 1.0) — so any gain is attributable purely to *pooling*, not to the clusterer.

### Robustness — does the answer depend on *which* seeds?

- **Determinism:** consensus is one fixed answer (rerun std = 0; a single seed has std ~0.02).
- **Leave-one-seed-out:** the worst 4-seed consensus, **0.6190, still beats the best single seed (0.6052)**.
- **Seed-subsets:** across all 3- and 4-seed combinations, **13/15 beat the best single seed** (mean 0.6212 ± 0.0221).
- **Stress test (bootstrap, can drop a whole clusterer):** worst case 0.5562 dips below the GMM mean — robust to *seed choice*, **not** to shrinking the ensemble. Reported honestly.

### Seed-pool replication

The strongest robustness test: retrain SEDR on a **disjoint** set of 5 seeds and rebuild the consensus (`consensus_seed_replication.py`). *Does a different seed pool also land at ~0.65?*

| Pool | seeds | consensus ARI | best single seed |
|---|---|---|---|
| A (original) | 42,123,456,789,1234 | 0.6504 | 0.6052 |
| B (independent) | 7,88,314,2024,51966 | 0.6202 | 0.5852 |
| A+B (10 seeds) | all ten | 0.6132 | 0.6052 |

What this revealed (verified by a diagnostic workflow):
- **0.6504 was a lucky draw.** Enumerating *all 252* five-seed subsets of the ten seeds gives mean **0.6059 ± 0.024** (range 0.556–0.653); pool A's 0.6504 is the **98th percentile** (5th of 252). The honest point estimate is the 10-seed **0.6132**.
- **At 2 pools it looked like consensus ≥ best single — but that did NOT survive scaling.** Pools A and B both happened to land ≥ their best single (+0.045, +0.035), but the 20-seed study below shows that is a **coin flip** (~48% of pools). The honest claim is *consensus ≈ best-of-pool*, not *> best seed*.
- **A+B below both pools = regression to the mean, not a bug.** The two pools *agree* strongly (ARI(consensus_A, consensus_B) = 0.84); the pooled co-association matrix is the exact average of the two. Pool A being a lucky outlier, the 10-seed estimate regresses toward the true ~0.61.
- **More seeds ≠ higher ARI.** Adding seeds gives a more honest, lower-variance estimate and guards against a bad seed; it does **not** raise the number (and lowers it when the starting pool was lucky). Never pick the seed pool that maximizes the reported number.

### Robustness across 20 seeds (`consensus_robustness.py` — the definitive test)

We then scaled to **20 seeds** and sampled **60 random 5-seed pools** (plus disjoint pools), comparing the two outcomes a practitioner faces:

| Outcome | mean ARI | std | range |
|---|---|---|---|
| **"run once"** (single seed, n=20) | 0.5486 | 0.0327 | 0.481–0.605 |
| **"run 5 + consensus"** (n=60 pools) | 0.5858 | 0.0234 | 0.505–0.641 |

- **Lift over a typical run: +0.037**, and **across-pool variance is reduced ~1.4×** (0.0327 → 0.0234) — both real.
- **Determinism:** each pool returns one fixed answer (within-pool std 0).
- **But consensus does NOT reliably beat the pool's best single seed** — only **29/60 pools (48%, sign-test p = 0.65)**. The consensus mean (0.586) ≈ the *expected best-of-5* (~0.587), so it **matches the best-of-pool, not exceeds it**. (The 2-pool "≥ best single" was small-sample luck.)
- **The floor isn't a hard guarantee:** a weak pool can still land ~0.50 (below a typical single run).

**Honest claim:** the consensus is deterministic **seed-insurance** — it delivers ≈ the **best-of-pool** result **deterministically and label-free** (you cannot pick the best single seed without the gold labels), **lifting a typical run by +0.037 with ~1.4× lower across-pool variance**. It does **not** beat the luckiest seed (≈ coin flip) and is not a hard floor. The value is *label-free, reproducible, above-typical performance* — not a guaranteed win. (`results/consensus_robustness.csv`)

### Honest negatives

- **The absolute 0.6504 does not replicate** — it's a 98th-percentile lucky seed pool; an independent pool gives 0.6202 and the 10-seed estimate is 0.6132. The honest claim is *seed-insurance / variance reduction*, not "consensus beats the best seed."
- **Consensus does not reliably beat the *best* single seed** — at 20 seeds it's only 29/60 pools (~48%, a coin flip; p = 0.65). It *matches* best-of-pool (label-free), it does not exceed it.
- The **spatial** and **quality-weight twists all score *below* plain consensus** — the pre-registered "twist beats plain" novelty **failed**.
- The **label-free variant selector picked the *worst*** of the four variants — reported as a negative, not as validation.
- The pre-registered **"≥5× variance collapse" criterion was missed** (only ~1.2×).

### Honest framing & scope

This is an **application / robustness study**, not a new clustering algorithm. The contributions are (1) a deterministic consensus that delivers ≈ the **best-of-pool** result **label-free** — lifting a typical single run by **+0.037** with **~1.4× lower across-pool variance** (seed-insurance), though it does **not** beat the *best* single seed (≈ coin flip at 20 seeds), (2) a label-free **per-spot stability map**, and (3) the attribution ablations. It was stress-tested by **two adversarial workflows + a 20-seed robustness study** that successively corrected the claim from "0.6504 beats the best seed" → "≈ best-of-pool, label-free, +0.037 over a typical run."

**Caveats:** n = 5 seeds, **single tissue section**, **no formal significance test** (gaps are descriptive), and consensus uses 15 base runs vs 1 for a single seed. The highest-value next step is **generalization to ≥3 DLPFC sections** (gene-only, no histology).

### Run it

```bash
python run_consensus.py     # ~6 min first run; ~1 min on reruns (embeddings cached)
```
Requires `leidenalg` + `igraph` for the Leiden base method (`pip install leidenalg igraph`; falls back to KMeans+GMM if absent). **Output:** `results/consensus_ari.csv`, `consensus_perseed.csv`, `stability_consensus.npy`. The consensus math has a standalone, GPU-free self-test: `python consensus_func.py`.

---

## Reproducibility

- Gate network seed: `torch.manual_seed(42)` in `prepare_features.py`
- SEDR evaluation: 5 seeds [42, 123, 456, 789, 1234], mean ± std reported
- All other steps are deterministic

---

## Limitations & honest scope

- **Single tissue section.** All results are on HBRC Block A Section 1. The findings (feature fusion hurts, graph fusion ties, image adds no robust gain) are demonstrated here only and may not transfer to other tissues where morphology is more domain-discriminative.
- **The headline is a negative/methods result, not a state-of-the-art claim.** GateST does *not* beat gene-only by adding histology; its contribution is the controlled finding that *where* you fuse matters, plus evidence that the UNI image modality lacks fine-domain signal beyond genes on this section.
- **Clusterer dependence.** The image-gated graph's small edge appears under KMeans but not under GMM; we report both and conclude a tie. Single-clusterer reporting would have been misleading.
- **Refinement comparability.** Our refined numbers are not directly comparable to published single-run baselines whose post-processing is unknown.
- **One foundation model.** Only UNI was tested. A different pathology encoder (CONCH, Virchow, GigaPath) might carry more signal, but the ~0.24 ARI gap to gene-only makes a reversal unlikely.
- **Part 2 has its own scope.** The [consensus result](#part-2--consensus--robustness) is also single-section (N=1), uses 5 seeds, and has no formal significance test; it is an *application* of evidence-accumulation clustering, not a new algorithm. (The bullets above are about Part 1, the fusion study.)

---

## Citation

If you use this code, please cite:

```bibtex
@misc{GateST2025,
  author = {Delfina Amarissa Sumanang},
  title  = {GateST: Gated Multimodal Fusion for Spatial Transcriptomics Domain Identification},
  year   = {2026},
  url    = {https://github.com/dmarissas/GateST}
}
```

---

## Acknowledgements

- [SEDR](https://github.com/JinmiaoChenLab/SEDR) — base spatial GNN model (Li et al.)
- [UNI](https://huggingface.co/MahmoodLab/uni) — pathology foundation model (Chen et al., 2024)
- [Xu et al. 2022](https://doi.org/10.1038/s41592-022-01494-7) — HBRC gold standard annotations
- [TGR-NMF](https://academic.oup.com/bib/article/26/1/bbae707/7945615) — published baseline (Li et al., 2024)
- [Brussee et al. 2024](https://arxiv.org/abs/2406.12808) — GNN in histopathology review
