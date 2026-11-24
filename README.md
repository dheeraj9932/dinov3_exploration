# dinov3_exploration

A hands-on lab for learning how **DINOv3** (`ViT-S/16`, `hidden_size=384`) works from the inside out — set up from scratch with `uv` + Hugging Face `transformers`, running on a local RTX 3050 (4 GB) GPU.

> Learning companion: `JOURNEY.md` tracks the full iteration log (mental models, exercises, and the next queued topic: the 3-channel PCA rainbow map).

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

### `run_random.py` — first forward pass
Feeds a random tensor `(1, 3, 224, 224)` through the model and prints the output shape `(1, 201, 384)`, confirming the `1 + 4 + 196 = 201` token anatomy.

```bash
.venv/bin/python run_random.py
```

### `compare_real_vs_noise.py` — cosine similarity heatmaps
Runs DINOv3 on the COCO two-cats photo and on random noise, computes cosine similarity of every patch tile against anchor tile `(7, 7)`, and saves a 3-panel comparison to `similarity_comparison.png`.

```bash
.venv/bin/python compare_real_vs_noise.py
```

Findings visible in `similarity_comparison.png`:
- On the real photo, the anchor tile (on the pink couch between the cats) lights up the whole couch while carving out both cats' silhouettes — zero-shot object grouping.
- On random noise, similarity forms a radial bullseye around the anchor — the signature of DINOv3's 2D RoPE position encoding when visual content carries no semantic signal.

## Status / next up

- [x] Iteration 1: Setup from scratch + first run on random input
- [x] Iteration 2: `AutoImageProcessor` + patch cosine similarity (real image vs. noise)
- [ ] Iteration 3: 3-channel PCA rainbow map (visualizing all 196 tiles at once)
