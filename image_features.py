# Extract multi-scale UNI embeddings per Visium spot.
# Uses Macenko stain normalization + 3-scale patch extraction.
#
# Input : TIFF image + spatial/tissue_positions_list.csv
# Output: processed/image_features_256d.csv  [barcode, uni_pca_0..255]
#
# NOTE: This script is unchanged from your working version.
# Only run again if you want to change SCALES or PCA_DIMS.
# Runtime: ~15 minutes on GPU.

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchstain
from sklearn.decomposition import PCA
import joblib

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

Image.MAX_IMAGE_PIXELS = None

# ── CONFIG ─────────────────────────────────────────────────────────
CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(CODE_DIR, "data", "block_a_section1_v110")
TIFF_FILE  = "V1_Breast_Cancer_Block_A_Section_1_image.tif"
SCALES     = [1, 2, 3]
BATCH_SIZE = 16
PCA_DIMS   = 256
# ──────────────────────────────────────────────────────────────────

PROC_DIR    = os.path.join(BASE_DIR, "processed")
SPATIAL_CSV = os.path.join(BASE_DIR, "spatial", "tissue_positions_list.csv")
SCALE_JSON  = os.path.join(BASE_DIR, "spatial", "scalefactors_json.json")
os.makedirs(PROC_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2: UNI Image Feature Extraction")
print("=" * 60)

# ── Load UNI ───────────────────────────────────────────────────────
print("\n[1] Loading UNI model...")
login(add_to_git_credential=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

uni_model = timm.create_model(
    "hf-hub:MahmoodLab/uni",
    pretrained=True,
    init_values=1e-5,
    dynamic_img_size=True,
)
uni_model.eval().to(device)
uni_transform = create_transform(
    **resolve_data_config(uni_model.pretrained_cfg, model=uni_model)
)
FEAT_DIM   = 1024
TOTAL_DIMS = len(SCALES) * FEAT_DIM
print(f"    {len(SCALES)} scales x {FEAT_DIM}d = {TOTAL_DIMS}d total")

# ── Stain normalizer ───────────────────────────────────────────────
print("\n[2] Fitting Macenko stain normalizer...")
tiff_path = os.path.join(BASE_DIR, TIFF_FILE)
img_full  = np.array(Image.open(tiff_path).convert("RGB"))
h, w      = img_full.shape[:2]
ref_patch = img_full[h//2-256:h//2+256, w//2-256:w//2+256]
T_plain   = transforms.ToTensor()
normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
normalizer.fit(T_plain(Image.fromarray(ref_patch)) * 255)
print("    Fitted on central 512x512 patch")

def normalize_patch(patch_pil):
    try:
        t   = T_plain(patch_pil) * 255
        out = normalizer.normalize(t, stains=False)
        img = out[0] if isinstance(out, tuple) else out
        arr = img.cpu().numpy().clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr), True
    except Exception:
        return patch_pil, False

# ── Coordinates ────────────────────────────────────────────────────
print("\n[3] Loading spot coordinates...")
pos = pd.read_csv(SPATIAL_CSV, header=None,
                  names=["barcode","in_tissue","row","col","y","x"])
pos = pos[pos["in_tissue"] == 1].copy()
with open(SCALE_JSON) as f:
    spot_diam = json.load(f)["spot_diameter_fullres"]
crop_sizes = [max(64, int(round(spot_diam * s))) for s in SCALES]
print(f"    Spots: {len(pos)}  Crop sizes: {crop_sizes}px")

# ── Patch extraction + UNI embedding ──────────────────────────────
def crop_patch(img, cx, cy, size):
    h = size // 2
    H, W = img.shape[:2]
    x0, x1 = int(cx)-h, int(cx)+h
    y0, y1 = int(cy)-h, int(cy)+h
    pl = max(0,-x0); pr = max(0,x1-W)
    pt = max(0,-y0); pb = max(0,y1-H)
    patch = img[max(0,y0):min(H,y1), max(0,x0):min(W,x1)]
    if pl or pr or pt or pb:
        patch = np.pad(patch, ((pt,pb),(pl,pr),(0,0)), mode="reflect")
    return patch

def embed_batch(patches, model, device, transform):
    tensors = torch.stack([transform(p) for p in patches]).to(device)
    with torch.inference_mode():
        feats = model(tensors)
    return feats.cpu().numpy()

print("\n[4] Extracting embeddings...")
embeddings = np.zeros((len(pos), TOTAL_DIMS), dtype=np.float32)

for si, crop_px in enumerate(crop_sizes):
    sc = si * FEAT_DIM
    ec = sc + FEAT_DIM
    batch_p, batch_i = [], []
    for i, (cx, cy) in enumerate(tqdm(
        zip(pos["x"], pos["y"]), total=len(pos),
        desc=f"  Scale {SCALES[si]}x ({crop_px}px)"
    )):
        raw = crop_patch(img_full, cx, cy, crop_px)
        pil = Image.fromarray(raw.astype(np.uint8))
        normed, _ = normalize_patch(pil)
        batch_p.append(normed)
        batch_i.append(i)
        if len(batch_p) == BATCH_SIZE:
            embeddings[batch_i, sc:ec] = embed_batch(batch_p, uni_model, device, uni_transform)
            batch_p, batch_i = [], []
    if batch_p:
        embeddings[batch_i, sc:ec] = embed_batch(batch_p, uni_model, device, uni_transform)

# Save the RAW multi-scale UNI embeddings (pre-normalization, pre-PCA) so the
# image representation can be studied offline (per-scale, different PCA dims)
# without re-running UNI extraction. Rows are in `pos` order = the same order as
# image_features_256d.csv's barcode index. Columns: scale1[0:1024], scale2[1024:2048],
# scale3[2048:3072].
np.save(os.path.join(PROC_DIR, "image_features_raw3072.npy"), embeddings.astype(np.float32))
print(f"\n[4b] Saved raw multi-scale embeddings: image_features_raw3072.npy {embeddings.shape}")

# ── L2 normalize → PCA → L2 normalize ─────────────────────────────
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-8)
embeddings = embeddings / norms

print(f"\n[5] PCA: {TOTAL_DIMS}d → {PCA_DIMS}d")
pca    = PCA(n_components=PCA_DIMS, random_state=42)
emb_pca = pca.fit_transform(embeddings).astype(np.float32)
print(f"    Variance explained: {pca.explained_variance_ratio_.sum():.3f}")

norms   = np.linalg.norm(emb_pca, axis=1, keepdims=True).clip(1e-8)
emb_pca = emb_pca / norms

# ── Save ───────────────────────────────────────────────────────────
df_image = pd.DataFrame(
    emb_pca,
    index=pos["barcode"].values,
    columns=[f"uni_pca_{i}" for i in range(PCA_DIMS)]
)
df_image.index.name = "barcode"
df_image.to_csv(os.path.join(PROC_DIR, "image_features_256d.csv"))
joblib.dump(pca, os.path.join(PROC_DIR, "pca_uni.pkl"))

print(f"\n[6] Saved:")
print(f"    image_features_256d.csv : {df_image.shape}")