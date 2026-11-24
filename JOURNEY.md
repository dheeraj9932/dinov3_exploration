# DINOv3 Learning Journey (`dinov3_exploration`)

## 0. Resume Point for Next Session
- **Status:** Iterations 1 and 2 completed.
- **Next Up (Locked by Learner):** Start **Iteration 3 (Phase A) — Option C: 3-Component PCA Rainbow Map (Visualizing All 196 Tiles Without an Anchor)** in `/home/dheeraj/projects/dinov3_exploration`.

## 1. North Star Milestone
Build a reproducible `dinov3_exploration` Git repository from scratch using `uv` and Hugging Face `transformers`, run `DINOv3-ViT-S/16` on a random input tensor and a real image, verify and slice the output tensor anatomy (`CLS` token, 4 register tokens, and 14x14 spatial patch tokens), and push the repository to GitHub (`dheeraj9932/dinov3_exploration`).

## 2. Mandelbrot Iteration Log

### Iteration 1: The 80/20 Big Shape (Completed)
- **Mental Model:** DINOv3 is a self-taught visual compression engine that cuts an image into a grid of 16x16 tiles and outputs a 3-part report (`1` global summary token + `4` scratchpad register tokens + `196` local tile tokens for a 224x224 image), where every token is a 384-number fingerprint vector (`hidden_size=384`).
- **Why This Over That:**
  - **`DINOv3-ViT-S/16` via Hugging Face `transformers` (`AutoModel`)** over the raw `facebookresearch/dinov3` research repo: `transformers` gives standard PyTorch module inspection with two lines of code, and `ViT-S/16` (21M params, `hidden_size=384`) fits easily on the local RTX 3050 (4 GB VRAM) while sharing the exact same architecture as the larger ViT-B/L/7B models.
  - **Ungated mirror (`computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m`)** over waiting on manual approval for `facebook/dinov3-vits16-pretrain-lvd1689m`: identical `DINOv3ViTModel` `safetensors` weights with zero download wait.
- **Exercise Completed:**
  - Initialized `~/projects/dinov3_exploration` with Git and `uv` (`.venv` with `torch 2.15.0+cu132`, `cuda: True`, `transformers 5.24.1`).
  - Wrote and ran `run_random.py` on GPU with `torch.randn(1, 3, 224, 224, device="cuda")`, verifying `output.last_hidden_state.shape == torch.Size([1, 201, 384])`.

### Iteration 2: Real Image vs. Random Noise — `AutoImageProcessor` & Patch Cosine Similarity (Completed)
- **Mental Model:**
  - `AutoImageProcessor` is the "Customs Checkpoint" that resizes arbitrary images to `224x224`, rescales `0..255` to `0..1`, and normalizes RGB channels to mean `0`, std `1` so internal attention `softmax` scores stay in their active, non-saturated range.
  - Cosine similarity acts as a 384-D compass comparing the semantic direction of one anchor tile against all `14x14` tiles.
- **Why This Over That:**
  - **`AutoImageProcessor` + Cosine Similarity** over manual pixel resizing and Euclidean distance: matches pretraining RGB normalization exactly and measures semantic direction (`-1` to `+1`) independent of vector magnitude.
- **Exercise Completed:**
  - Installed `matplotlib` and wrote `compare_real_vs_noise.py` comparing the COCO cats image against `torch.randn` noise.
  - Generated `similarity_comparison.png`: anchor tile `(7, 7)` (which landed on the pink couch between the two cats) lit up the entire couch in dark red while carving out both cats' silhouettes in green; meanwhile, random noise produced a smooth radial bullseye around `(7, 7)` (revealing the 2D Rotary Position Embeddings when visual content is uniform noise).

## 3. Roadblocks Solved
- **Manual Gating on Hugging Face:** Verified via the Hugging Face API that `facebook/dinov3-vits16-pretrain-lvd1689m` requires manual approval. Located an identical ungated `safetensors` mirror (`computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m`) so local runs work immediately without an auth blocker.
- **Patch Count Ground Truth (196 vs. 256):** Corrected an earlier math slip (and a typo in the upstream HF docs comment). A `224x224` input divided by `patch_size=16` produces a `14 x 14 = 196` patch grid (not 256). Combined with `1` CLS token and `4` register tokens, `last_hidden_state` has sequence length `1 + 4 + 196 = 201`.
- **Mental Model Calibration (Class IDs vs. 384-D Fingerprints):** Clarified that Index 0 (`CLS`) is not a `0..9` class ID and Indexes `5..200` are not a 14x14 row-vs-column comparison matrix; instead, every single token is a `384`-number embedding vector (1 whole-image vector, 4 disposable scratchpad vectors, and 196 tile vectors).

## 4. Next Zoom Options (Queued for Next Session)
1. **[SELECTED FOR ITERATION 3] Option C — 3-Component PCA Rainbow Map (Visualizing All 196 Tiles Without an Anchor):** Project `[14, 14, 384]` down to `[14, 14, 3]` RGB channels with PCA (the signature visualization from the DINOv2/v3 papers).
2. **Option A — Under the Hood of `DINOv3ViTModel` (Layer-by-Layer Architecture & 2D RoPE):** Inspect `model` internals: how the `Conv2d(3, 384, kernel_size=16, stride=16)` tile-cutter, 2D RoPE (which caused the subtle radial bullseye on the noise heatmap!), and 12 attention blocks transform `(1, 3, 224, 224)` into `(1, 201, 384)`.
3. **Option B — The Register Token Mystery (Vector Norms & Attention Maps):** Measure the L2 norms and attention maps of `cls_token` vs. `register_tokens` vs. `patch_tokens` to see why those 4 register slots absorb global high-norm spikes.
