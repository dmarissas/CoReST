# GateST: Gated Multimodal Fusion for Spatial Transcriptomics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Gated fusion of UNI histology image features and gene expression for spatial tissue domain identification in breast cancer Visium data.

---

## Overview

Spatial transcriptomics methods typically rely on gene expression alone for tissue domain identification. GateST proposes a **gated multimodal fusion strategy** that adaptively combines:

- **Gene expression features** — 200d PCA from 2000 highly variable genes (standard SEDR preprocessing)
- **Histology image features** — multi-scale UNI embeddings (3×1024d → 256d PCA) from H&E-stained tissue patches

A lightweight gate network learns a per-spot weighting between the two modalities, then feeds the fused representation into **SEDR** (Spatial Embedding by Deep learning with Regularization), a VAE + GNN model that exploits spatial neighborhood structure for unsupervised domain clustering.

### Key finding

Naive concatenation of gene and image features **hurts** performance (ARI 0.4445 ± 0.0092) compared to gene-only (ARI 0.4783 ± 0.0302). The proposed gated fusion **substantially recovers this loss** (ARI 0.4705 ± 0.0305), clearly outperforming naive concatenation — demonstrating that adaptive per-spot weighting is a better fusion strategy than equal-weight concatenation.

However, on this benchmark the UNI histology features do **not** improve over gene expression alone: gene-only remains the strongest single condition (0.4783). In other words, gating is the better *way to combine* modalities, but the image modality adds limited domain-discriminative signal beyond gene expression for this breast-cancer section. This is consistent with the learned gate being near-constant (mean ≈ 0.52, std ≈ 0.02), i.e. only weakly per-spot adaptive.

> Results are the mean ± std over 5 SEDR seeds with a fixed gate-network seed (42). All four conditions feed SEDR at their native feature width (gene 200d, image 256d, concat 456d, gated 128d).

---

## Results

Performance on HBRC Block A Section 1 (Xu et al. gold standard, 20 fine-grained domains):

| Method | ARI (fine, k=20) |
|--------|-----------------|
| GateST — Image only (SEDR) | 0.2653 ± 0.0027 |
| SEDR (published) | 0.3668 |
| GateST — Concat fusion (SEDR) | 0.4445 ± 0.0092 |
| Seurat | 0.4612 |
| **GateST — Gated fusion (SEDR)** | **0.4705 ± 0.0305** |
| GateST — Gene only (SEDR) | 0.4783 ± 0.0302 |
| STAGATE | 0.4944 |
| TGR-NMF | 0.5286 |

> Results reported as mean ± std over 5 SEDR random seeds with fixed gate network seed=42.
> Published baselines (Seurat, STAGATE, TGR-NMF, SEDR) are single-run values with unknown seeds.
>
> **Takeaway:** gated fusion (0.4705) > concat fusion (0.4445) — adaptive gating beats naive
> concatenation — but gene-only (0.4783) edges out gated fusion, so the histology modality does
> not add domain-discriminative value beyond gene expression on this section.

### Visualizations

**ARI comparison across feature conditions (KMeans baseline vs SEDR):**

![Bar Plot](figures/results_barplot.png)

**Gold standard tissue domains vs GateST gated fusion prediction:**

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

Step 3: Feature fusion
  ├── gene_only      (200d) — baseline
  ├── image_only     (256d) — ablation
  ├── concat_fused   (456d) — naive z-score concatenation
  └── gated_fused    (456d) — learned adaptive gate (novel)

Step 4: SEDR training + evaluation
  └── Spatial k=6 KNN graph → VAE + GNN → KMeans → ARI vs gold standard

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

# Step 4: Train SEDR on all conditions (~90 min, 5 seeds each)
python train_sedr.py

# Step 4b: Regenerate per-spot cluster assignments from the saved embeddings (~1 min)
# REQUIRED before Step 5 — train_sedr.py saves only embeddings + ARI scores,
# while visualize.py reads the clusters_<condition>_{k20,k4}.npy files produced here.
python cluster_regeneration.py

# Step 5: Generate all figures (~5 min)
python visualize.py
```

---

## Project Structure

```
GateST/
├── gene_features.py        # Step 1: Gene PCA extraction + labels + coords
├── image_features.py       # Step 2: Multi-scale UNI image embedding
├── prepare_features.py     # Step 3: Gated fusion + all feature variants
├── train_sedr.py           # Step 4: SEDR training + ARI evaluation
├── cluster_regeneration.py # Step 4b: Regenerate cluster files from embeddings (required for Step 5)
├── visualize.py            # Step 5: Figure generation
├── SEDR_model.py           # SEDR model architecture (VAE + GNN)
├── graph_func.py           # Spatial graph construction
├── utils_func.py           # Preprocessing utilities
├── requirements.txt
└── README.md
```

---

## Gated Fusion

The core contribution is a lightweight gate network trained in `prepare_features.py`:

```python
# For each spot, compute a scalar gate g ∈ (0,1)
g = sigmoid(W_gate * concat([gene_features, image_features]))

# Project each modality to shared space
h_gene  = ReLU(W_gene  * gene_features)
h_image = ReLU(W_image * image_features)

# Adaptive weighted combination
h_fused = g * h_gene + (1 - g) * h_image
```

The gate is trained by minimizing reconstruction loss for **both** modalities simultaneously, with entropy regularization to prevent gate collapse:

```
loss = recon_gene + 0.5 * recon_image - 0.3 * entropy(g)
```

This encourages the model to genuinely leverage both modalities rather than ignoring one.

---

## Why Naive Concatenation Hurts

A key finding of this work is that simply concatenating gene and image features reduces performance below gene-only:

```
Gene only SEDR:    ARI 0.4783  ← strongest single condition
Gated SEDR:        ARI 0.4705  ← recovers most of the concat loss
Concat SEDR:       ARI 0.4445  ← worse than gene only
Image only SEDR:   ARI 0.2653  ← weak on its own
```

Equal-weight concatenation dilutes the stronger gene-expression signal with the weaker image features. The gate network mitigates this by learning to down-weight the image modality, recovering most of the lost performance and clearly beating concatenation. It does not, however, exceed gene-only here — because the gate converges to a near-constant ~0.52 mix and the image features carry little domain-discriminative signal on this section, so any image contribution is on balance a slight dilution rather than a gain.

---

## Reproducibility

- Gate network seed: `torch.manual_seed(42)` in `prepare_features.py`
- SEDR evaluation: 5 seeds [42, 123, 456, 789, 1234], mean ± std reported
- All other steps are deterministic

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
