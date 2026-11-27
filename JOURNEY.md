# DINOv3 Learning Journey (`dinov3_exploration`)

## 0. Resume Point for Next Session
- **Status:** Iterations 1, 2, 3, and 4 completed. Iteration 5 (North Star v2 Roadmap) pitched and locked.
- **Next Up (Locked by Learner):** Start **Iteration 5 (Phase B) — Build `task1_classification.py`**:
  1. Register 3 classes with 1 example image each (`Indoor Cats`, `Snowy Cat`, `Bear`).
  2. Compute and store their `cls_token` prototypes (`[3, 384]`).
  3. Feed a 4th novel image, compute cosine similarity against the 3 prototypes, and print the predicted class.

## 1. North Star Milestone
- **Milestone v1 (Achieved):** Build `dinov3_exploration` from scratch with `uv` + HF `transformers`, run DINOv3-ViT-S/16 on random and real inputs, slice/verify the `1 + 4 + 196 = 201` token anatomy, and visualize patch cosine similarity & PCA rainbow maps.
- **Milestone v2 (Active):** Build a **DINOv3 Multi-Task CV Lab** in `dinov3_exploration` that takes any input image (frozen `ViT-S/16`, `hidden_size=384`) and runs 5 core CV tasks (Few-Shot Classification, Object Detection, Semantic Segmentation, Instance Segmentation, Monocular Depth Estimation), and wire them into a simple Streamlit UI (`app.py`) at the end. Push to GitHub (`dheeraj9932/dinov3_exploration`).

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

### Iteration 3: 3-Component PCA Rainbow Map — Visualizing All 196 Tiles Without an Anchor (Completed)
- **Mental Model:**
  - PCA rotates a camera in 384-D space to find the 3 orthogonal axes (`PC1`, `PC2`, `PC3`) of widest spread across all 196 tiles.
  - Each component is a continuous `0..1` coordinate axis (not a binary "1 object per component" switch). Mixing `(PC1, PC2, PC3)` as `(Red, Green, Blue)` paints 5+ distinct objects/parts simultaneously without picking an anchor tile.
- **Why This Over That:**
  - **3-Component PCA via `torch.pca_lowrank`** over single-tile cosine similarity or UMAP/t-SNE: deterministic, linear, built directly into PyTorch, and summarizes the top 3 semantic contrasts of the entire scene in one RGB image.
- **Exercise Completed:**
  - Wrote `pca_rainbow.py` and generated the 5-panel `pca_rainbow.png` (`Input`, `PC1`, `PC2`, `PC3`, `PCA Rainbow`).
  - Verified on `pca_rainbow.png`: `PC1` split both cats (`0.0`) from the pink couch (`1.0`) with remotes in the middle (`0.5`); `PC2` split the top couch backrest from the bottom couch seat; `PC3` split the left cat, right cat, and TV remotes — combining into 5+ distinct colors in the final RGB rainbow map.

### Iteration 4: The Map of Computer Vision Tasks with DINOv3 — Whole-Image Retrieval & Cross-Image Part Tracking (Completed)
- **Mental Model:**
  - Frozen DINOv3 offers two plugs: **Plug A (`cls_token`, `[1, 384]`)** for whole-image tasks (search/retrieval, k-NN classification, clustering, linear-probe classification) and **Plug B (`patch_tokens`, `[14, 14, 384]`)** for spatial/dense tasks (cross-image part correspondence, anomaly/defect localization, foreground segmentation, depth estimation, object detection).
- **Why This Over That:**
  - **Grouping CV tasks by (`cls_token` vs. `patch_tokens`) × (Zero-Training vs. Plug-On Head)** over training separate task networks from scratch: keeps the 21M-parameter backbone 100% frozen and solves retrieval and part correspondence with pure matrix multiplication.
- **Exercise Completed:**
  - Wrote and ran `cross_image_correspondence.py` on a batch of 3 images (Indoor Cats, Snowy Pallas's Cat, Grizzly Bear).
  - Verified Task 1 (`cls_token` retrieval): Indoor Cats matched Snowy Cat (`0.296`) while rejecting Bear (`-0.054`).
  - Verified Task 2 (`patch_tokens` cross-image correspondence in `cross_image_correspondence.png`): anchor tile `(2, 4)` on the indoor cat's left ear/head automatically located tile `(3, 4)` (`sim=0.599`) right on the snowy cat's left ear/head across completely different backgrounds and poses.

## 3. Roadblocks & Concepts Solved (Discussion Recap)
- **Manual Gating on Hugging Face:** Verified via the Hugging Face API that `facebook/dinov3-vits16-pretrain-lvd1689m` requires manual approval. Located an identical ungated `safetensors` mirror (`computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m`) so local runs work immediately without an auth blocker.
- **Patch Count Ground Truth (196 vs. 256):** Corrected an earlier math slip (and a typo in the upstream HF docs comment). A `224x224` input divided by `patch_size=16` produces a `14 x 14 = 196` patch grid (not 256). Combined with `1` CLS token and `4` register tokens, `last_hidden_state` has sequence length `1 + 4 + 196 = 201`.
- **Mental Model Calibration (Class IDs vs. 384-D Fingerprints):** Clarified that Index 0 (`CLS`) is not a `0..9` class ID and Indexes `5..200` are not a 14x14 row-vs-column comparison matrix; instead, every single token is a `384`-number embedding vector (1 whole-image vector, 4 disposable scratchpad vectors, and 196 tile vectors).
- **Mental Model Calibration (3 PCA Axes vs. Only 3 Objects):** Clarified that 3 PCA components are 3 continuous orthogonal axes `(R, G, B)` rather than 3 isolated object slots, allowing a single `[14, 14, 3]` RGB plot to separate 5+ distinct object parts by color mixing.
- **`401 RepositoryNotFoundError` from Typo in Model ID (`dinov3-vit16` vs `dinov3-vits16`):** Diagnosed a missing `s` in `MODEL_ID` inside `cross_image_correspondence.py`. Hugging Face Hub returns `401` on nonexistent repository names when unauthenticated (to avoid leaking private repo existence). Fixed to `dinov3-vits16`.
- **DINOv3 is a Recipe, Not One Architecture (ViT vs. ConvNeXt):** Meta trained a 7B ViT teacher, then distilled it into ViT students (all-to-all attention, $O(N^2)$ cost) and ConvNeXt students (4-stage CNN pyramid `56x56 -> 28x28 -> 14x14 -> 7x7`, $O(N)$ linear cost for edge/high-res deployment).
- **ConvNeXt Output Shape (No Register Tokens):** `DINOv3ConvNextModel` outputs `pooler_output` (`[1, 768]`, global average of final 49 tiles) and `last_hidden_state` (`[1, 50, 768]` = 1 averaged vector + 49 spatial tiles), plus a 4-stage multi-scale pyramid when `output_hidden_states=True`.
- **Arbitrary Resolution & Aspect Ratio (No 224x224 Squash Needed):** Verified `DINOv3ViTModel` on a `16:9` tensor `[1, 3, 368, 640]` (`23*16 x 40*16`), outputting `[1, 925, 384]` (1 CLS + 4 Registers + 920 patches). Any height/width multiple of 16 works; pass `size={"height": H, "width": W}` to `AutoImageProcessor` to preserve small object details.
- **Why Patch-to-Patch Beats CLS-to-Patch for Detection:** `cls_token` and `patch_tokens` live in different internal coordinate dialects, so detection/segmentation prototypes should be built from `patch_tokens` inside bounding boxes (patch-to-patch matching).
- **Heatmap to Bounding Box (Semantic vs. Instance Segmentation):** Thresholding a patch heatmap + connected components works for single objects/semantic masks, but touching objects fuse into one blob (requiring a trained DETR box head + NMS, or SAM point prompts for instance masks).
- **DINOv3 (Semantic Brain) vs. SAM (Precision Scissors):** DINOv3 is self-supervised on 1.7B images to learn *what* objects/parts mean in 384-D coarse grids; SAM is supervised on 1.1B masks to trace pixel-sharp borders around a click but is class-blind. They pair perfectly: DINOv3 finds the object point -> SAM cuts the sharp instance mask.
- **SOTA Landscape:** DINOv3 is SOTA for frozen general-purpose visual backbones, dense 3D/part features, and few-shot dense tasks. Text-prompt tasks need SigLIP 2 / VLMs; razor-sharp cutouts need SAM 2/3; 120-FPS fixed-class boxes need RT-DETRv2 / DEIMv2 / YOLO.

## 4. Roadmap: Iterations 5–10 (North Star v2)
1. **[NEXT UP] Iteration 5 — Few-Shot Classification (`task1_classification.py`):** Register classes with example images, store `cls_token` prototypes (`[3, 384]`), classify novel images via max cosine similarity.
2. **Iteration 6 — Object Detection (`task2_detection.py`):** Register object with example image + bbox, collect `patch_tokens` inside bbox as prototypes, generate novel-image patch heatmap, threshold + connected components to draw bounding boxes.
3. **Iteration 7 — Semantic Segmentation (`task3_semantic_seg.py`):** Convert patch similarity heatmap into an upscaled full-resolution binary object mask.
4. **Iteration 8 — Instance Segmentation (`task4_instance_seg.py`):** DINOv3 heatmap peaks become automatic point prompts for SAM (`facebook/sam-vit-base` or `sam2.1-hiera-tiny`, both ungated on HF) to trace sharp individual object masks.
5. **Iteration 9 — Monocular Depth Estimation (`task5_depth.py`):** DINOv3 ViT-S/16 has no depth head. Choose between a frozen-depth pipeline model (`depth-anything/Depth-Anything-V2-Small-hf`, ungated) for zero-shot depth or training a tiny depth head on frozen DINOv3 features (and see why specialized tasks need readout heads).
6. **Iteration 10 — Streamlit Multi-Task UI (`app.py`):** Image uploader + task selector (classification / detection / semantic / instance / depth), cached DINOv3 model loading, and example registration widgets.
