# Align gene + image features by barcode, then produce 4 fusion variants:
#
#   1. gene_only    — 200d gene PCA (baseline, matches published SEDR)
#   2. image_only   — 256d UNI PCA
#   3. concat       — 456d naive concatenation (z-score each, then concat)
#   4. gated        — 456d adaptive gated fusion (learned per-spot weighting)
#
# The gated fusion is the novel contribution:
# Instead of equal weighting, a sigmoid gate is computed per spot from
# both modalities, allowing the model to weight gene vs image features
# based on local tissue context. Motivated by Azher et al. (2024) and
# the adaptive fusion discussion in Brussee et al. (2024).
#
# Output: processed/
#   barcodes_final.csv
#   labels_final.csv
#   coords_final.csv
#   gene_only.npy       (3798, 200)
#   image_only.npy      (3798, 256)
#   concat_fused.npy    (3798, 456)
#   gated_fused.npy     (3798, 456)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
# ──────────────────────────────────────────────────────────────────

PROC_DIR = os.path.join(BASE_DIR, "processed")


print("=" * 60)
print("STEP 3: Feature Alignment + Fusion Variants")
print("=" * 60)

# ── Load features ──────────────────────────────────────────────────
print("\n[1] Loading features...")
df_gene  = pd.read_csv(os.path.join(PROC_DIR, "gene_features_200d.csv"), index_col="barcode")
df_image = pd.read_csv(os.path.join(PROC_DIR, "image_features_256d.csv"), index_col="barcode")
df_labels = pd.read_csv(os.path.join(PROC_DIR, "labels.csv"), index_col="barcode")
df_coords = pd.read_csv(os.path.join(PROC_DIR, "coords.csv"), index_col="barcode")

print(f"    Gene  : {df_gene.shape}")
print(f"    Image : {df_image.shape}")
print(f"    Labels: {df_labels.shape}")
print(f"    Coords: {df_coords.shape}")

# ── Align barcodes ────────────────────────────────────────────────
print("\n[2] Aligning barcodes...")
common = (df_gene.index
          .intersection(df_image.index)
          .intersection(df_labels.index)
          .intersection(df_coords.index))
print(f"    Common barcodes: {len(common)}")

df_gene   = df_gene.loc[common]
df_image  = df_image.loc[common]
df_labels = df_labels.loc[common]
df_coords = df_coords.loc[common]

# Save aligned reference files
pd.DataFrame({"barcode": common.tolist()}).to_csv(
    os.path.join(PROC_DIR, "barcodes_final.csv"), index=False)
df_labels.to_csv(os.path.join(PROC_DIR, "labels_final.csv"))
df_coords.to_csv(os.path.join(PROC_DIR, "coords_final.csv"))

print(f"    annot_type distribution:")
print(df_labels["annot_type"].value_counts().to_string())

# ── Z-score normalize each modality independently ─────────────────
print("\n[3] Z-score normalizing...")
gene_raw  = df_gene.values.astype(np.float32)
image_raw = df_image.values.astype(np.float32)

gene_scaled  = StandardScaler().fit_transform(gene_raw).astype(np.float32)
image_scaled = StandardScaler().fit_transform(image_raw).astype(np.float32)

print(f"    Gene  — mean={gene_scaled.mean():.4f}  std={gene_scaled.std():.4f}")
print(f"    Image — mean={image_scaled.mean():.4f}  std={image_scaled.std():.4f}")

# ── Variant 1: gene_only ──────────────────────────────────────────
gene_only = gene_scaled
print(f"\n[4] gene_only    : {gene_only.shape}")

# ── Variant 2: image_only ─────────────────────────────────────────
image_only = image_scaled
print(f"[5] image_only   : {image_only.shape}")

# ── Variant 3: concat (naive) ─────────────────────────────────────
concat_fused = np.concatenate([gene_scaled, image_scaled], axis=1)
print(f"[6] concat_fused : {concat_fused.shape}")

# ── Variant 4: gated fusion (novel contribution) ──────────────────
# Each spot gets a scalar gate g ∈ (0,1) computed from both modalities.
# fused = g * gene + (1-g) * image_padded
#
# The gate is computed as:
#   g = sigmoid(W_g * concat([gene, image]) + b_g)
#
# We learn this with a small autoencoder-style objective:
# minimize reconstruction of gene (dominant modality) from fused features.
# This encourages the gate to upweight gene when gene signal is strong
# and upweight image when gene signal is ambiguous.

print(f"\n[7] Computing gated fusion...")

GENE_DIM  = gene_scaled.shape[1]   # 200
IMAGE_DIM = image_scaled.shape[1]  # 256
HIDDEN    = 128
EPOCHS    = 300
LR        = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

# ── Fix random seed for reproducibility ───────────────────────────
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
# ──────────────────────────────────────────────────────────────────

gene_t  = torch.tensor(gene_scaled,  dtype=torch.float32).to(device)
image_t = torch.tensor(image_scaled, dtype=torch.float32).to(device)

class GatedFusion(nn.Module):
    def __init__(self, gene_dim, image_dim, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(gene_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.gene_proj  = nn.Linear(gene_dim,  hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        # Reconstruct BOTH modalities from fused
        self.recon_gene  = nn.Linear(hidden_dim, gene_dim)
        self.recon_image = nn.Linear(hidden_dim, image_dim)

    def forward(self, gene, image):
        g       = self.gate_net(torch.cat([gene, image], dim=1))
        h_gene  = torch.relu(self.gene_proj(gene))
        h_image = torch.relu(self.image_proj(image))
        h_fused = g * h_gene + (1 - g) * h_image
        return h_fused, self.recon_gene(h_fused), self.recon_image(h_fused), g

model     = GatedFusion(GENE_DIM, IMAGE_DIM, HIDDEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"    Training gate network ({EPOCHS} epochs)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    _, gene_recon, image_recon, g = model(gene_t, image_t)

    recon_gene_loss  = criterion(gene_recon,  gene_t)
    recon_image_loss = criterion(image_recon, image_t)

    # Entropy regularization — penalize gate collapse toward 0 or 1
    entropy_reg = -(
        g * torch.log(g + 1e-8) +
        (1 - g) * torch.log(1 - g + 1e-8)
    ).mean()

    # Combined loss: reconstruct both + maximize gate entropy
    loss = recon_gene_loss + 0.5 * recon_image_loss - 0.3 * entropy_reg
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"    Epoch {epoch:4d}  "
              f"gene={recon_gene_loss.item():.4f}  "
              f"image={recon_image_loss.item():.4f}  "
              f"gate_mean={g.mean().item():.3f}")

# Extract fused embeddings
model.eval()
with torch.no_grad():
    h_fused, _, _, gates = model(gene_t, image_t)
    h_fused = h_fused.cpu().numpy()
    gates   = gates.cpu().numpy().squeeze()

print(f"\n    Gate statistics:")
print(f"    mean={gates.mean():.3f}  std={gates.std():.3f}  "
      f"min={gates.min():.3f}  max={gates.max():.3f}")
print(f"    (gate>0.5 = gene-dominant spots: {(gates>0.5).sum()}/"
      f"{len(gates)})")

# Pad to 456d to match concat_fused dims for SEDR
# SEDR input_dim must be consistent — we pad with zeros to 456d
# so all conditions use the same SEDR architecture
pad_size   = concat_fused.shape[1] - h_fused.shape[1]  # 456 - 128 = 328
gated_fused = np.concatenate([
    h_fused,
    np.zeros((len(h_fused), pad_size), dtype=np.float32)
], axis=1)
print(f"\n    gated_fused (padded to 456d): {gated_fused.shape}")

# ── Save all variants ──────────────────────────────────────────────
print("\n[8] Saving feature arrays...")
np.save(os.path.join(PROC_DIR, "gene_only.npy"),     gene_only.astype(np.float32))
np.save(os.path.join(PROC_DIR, "image_only.npy"),    image_only.astype(np.float32))
np.save(os.path.join(PROC_DIR, "concat_fused.npy"),  concat_fused.astype(np.float32))
np.save(os.path.join(PROC_DIR, "gated_fused.npy"),   gated_fused.astype(np.float32))
np.save(os.path.join(PROC_DIR, "gates.npy"),         gates.astype(np.float32))

print(f"    gene_only.npy     : {gene_only.shape}")
print(f"    image_only.npy    : {image_only.shape}")
print(f"    concat_fused.npy  : {concat_fused.shape}")
print(f"    gated_fused.npy   : {gated_fused.shape}")
print(f"    gates.npy         : {gates.shape}  (per-spot gate values)")