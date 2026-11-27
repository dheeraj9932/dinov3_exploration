import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import matplotlib.pyplot as plt
from transformers.image_utils import load_image

MODEL_ID = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to("cuda")

urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # 0: Two indoor cats
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",  # 1: Snowy cat
    "http://images.cocodataset.org/val2017/000000000285.jpg",  # 2: Bear
]

names = ["0: Indoor Cats", "1: Snowy Cat", "2: Bear"]
images = [load_image(u) for u in urls]

inputs = processor(images=images, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model(**inputs).last_hidden_state

# TASK 1
cls_tokens = F.normalize(out[:,0,:], dim=-1)
cls_sim_matrix = (cls_tokens @ cls_tokens.T).cpu().numpy()

print("=== task 1: whole image cls cosine similarity ===")
for i, name in enumerate(names):
    print(f"{name:15s} -> {cls_sim_matrix[i].round(3)}")

    
# =====================================================================
# TASK 2 (Plug B — patch_tokens): Cross-Image Part Tracking
# =====================================================================
grid_indoor = out[0, 5:, :].unflatten(0, (14, 14))  # (14, 14, 384)
grid_snowy = out[1, 5:, :].unflatten(0, (14, 14))   # (14, 14, 384)

# Anchor tile (row=2, col=4) = Left cat's ear/head in Image 0
ar, ac = 2, 4
anchor_vec = grid_indoor[ar, ac]  # (384,)

# Compare against all 14x14 tiles of Image 1 (Snowy Cat)
cross_sim = F.cosine_similarity(grid_snowy, anchor_vec, dim=-1).cpu()  # (14, 14)
best_idx = int(cross_sim.argmax())
br, bc = divmod(best_idx, 14)
print(f"\n=== Task 2: Cross-Image Part Match ===")
print(f"Anchor tile ({ar},{ac}) in Indoor Cats matched tile ({br},{bc}) in Snowy Cat (sim={cross_sim[br, bc]:.3f})")

# Plot 3 panels: Source Anchor -> Cross-Image Heatmap -> Target Best Match
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
axes[0].imshow(images[0].resize((224, 224)))
axes[0].scatter([ac * 16 + 8], [ar * 16 + 8], c="red", s=120, marker="x", linewidths=3)
axes[0].set_title(f"Photo 0: Anchor Ear ({ar},{ac})")

axes[1].imshow(cross_sim, cmap="turbo")
axes[1].set_title("Cross-Photo Cosine Sim (14x14)")

axes[2].imshow(images[1].resize((224, 224)))
axes[2].scatter([bc * 16 + 8], [br * 16 + 8], c="lime", s=120, marker="x", linewidths=3)
axes[2].set_title(f"Photo 1: Auto-Matched ({br},{bc})")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("cross_image_correspondence.png", dpi=150)
print("Saved cross_image_correspondence.png")