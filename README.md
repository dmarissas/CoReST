# GateST: Gated Multimodal Fusion for Spatial Transcriptomics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Gated fusion of UNI histology image features and gene expression for spatial tissue domain identification in breast cancer Visium data.

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
