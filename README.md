# dinov3_exploration

A hands-on lab for learning how **DINOv3** (`ViT-S/16`, `hidden_size=384`) works from the inside out — set up from scratch with `uv` + Hugging Face `transformers`, running on a local RTX 3050 (4 GB) GPU.

> Learning companion: `JOURNEY.md` tracks the full iteration log (mental models, roadblocks solved, and next zoom options).

## What DINOv3 is (80/20 mental model)

A self-supervised visual foundation model that cuts an image into `16×16` tiles and compresses every tile into a `384`-number fingerprint vector. For a `224×224` input it outputs `last_hidden_state` of shape `[1, 201, 384]`:

| Slice | Indices | Shape | Meaning |
|---|---|---|---|
| CLS token | `[0]` | `[1, 384]` | Fingerprint of the whole image (retrieval, classification) |
| Register tokens | `[1:5]` | `[1, 4, 384]` | Scratchpad slots that absorb global artifacts (discarded) |
| Patch tokens | `[5:]` | `[1, 196, 384]` | One vector per tile, reshapeable to `[14, 14, 384]` (dense tasks, segmentation) |

## Setup

```bash
uv venv --python 3.12
uv pip install torch transformers pillow matplotlib
```

> Note: the official `facebook/dinov3-vits16-pretrain-lvd1689m` repo is manually gated on Hugging Face. This lab uses the identical ungated mirror `computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m` — no token or approval wait needed.

## Scripts

### 1. `run_random.py` — First Forward Pass
Feeds a random tensor `(1, 3, 224, 224)` through the model and prints the output shape `(1, 201, 384)`, confirming the `1 + 4 + 196 = 201` token anatomy.

```bash
.venv/bin/python run_random.py
```

### 2. `compare_real_vs_noise.py` — Cosine Similarity Heatmaps
Runs DINOv3 on the COCO two-cats photo and on random noise, computes cosine similarity of every patch tile against anchor tile `(7, 7)`, and saves a 3-panel comparison to `similarity_comparison.png`.

```bash
.venv/bin/python compare_real_vs_noise.py
```

### 3. `pca_rainbow.py` — 3-Component PCA Rainbow Map
Projects all 196 patch embeddings (`[196, 384]`) onto their top 3 principal components (`[196, 3]`), normalizes each component to `[0, 1]`, and plots `PC1`, `PC2`, `PC3`, and the combined `[14, 14, 3]` RGB rainbow image in `pca_rainbow.png`.

```bash
.venv/bin/python pca_rainbow.py
```

### 4. `cross_image_correspondence.py` — Whole-Image Retrieval & Cross-Image Part Tracking
Demonstrates the two core DINOv3 output plugs across 3 different photos (Indoor Cats, Snowy Cat, Grizzly Bear):
- **Plug A (`cls_token` `[3, 384]`):** Computes the `3×3` whole-image retrieval similarity matrix.
- **Plug B (`patch_tokens` `[14, 14, 384]`):** Takes anchor tile `(2, 4)` (the indoor cat's left ear) and searches all 196 tiles of the snowy cat photo, automatically matching tile `(3, 4)` on the snowy cat's left ear (`cross_image_correspondence.png`).

```bash
.venv/bin/python cross_image_correspondence.py
```

## Status

- [x] Iteration 1: Setup from scratch + first run on random input
- [x] Iteration 2: `AutoImageProcessor` + patch cosine similarity (real image vs. noise)
- [x] Iteration 3: 3-channel PCA rainbow map (visualizing all 196 tiles at once)
- [x] Iteration 4: Computer Vision tasks unlocked by DINOv3 (`cls_token` retrieval + `patch_tokens` cross-image part tracking)
