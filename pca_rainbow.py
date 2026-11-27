import torch
from transformers import AutoModel, AutoImageProcessor
import matplotlib.pyplot as plt
from transformers.image_utils import load_image

MODEL_ID = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"

model = AutoModel.from_pretrained(MODEL_ID).to("cuda")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
inputs = processor(images=image, return_tensors="pt").to("cuda")

with torch.inference_mode():
    out = model(**inputs).last_hidden_state

# 1. Slice the 196 spatial patch tokens -> (196, 384)
patches = out[0, 5:, :]

# 2. Center the 196 vectors and project onto top 3 PCA axes -> (196, 3)
patches_centered = patches - patches.mean(dim=0, keepdim=True)
U, S, V = torch.pca_lowrank(patches_centered, q=3)
pca_features = patches_centered @ V[:, :3]

# 3. Scale each component (PC1, PC2, PC3) to [0.0, 1.0] for RGB display
pca_min = pca_features.min(dim=0, keepdim=True).values
pca_max = pca_features.max(dim=0, keepdim=True).values
pca_norm = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

# 4. Reshape (196, 3) -> (14, 14, 3)
pca_grid = pca_norm.unflatten(0, (14, 14)).cpu().numpy()


# 5. Plot Input, PC1, PC2, PC3, and Combined RGB
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes[0].imshow(image.resize((224, 224)))
axes[0].set_title("Input (224x224)")
axes[1].imshow(pca_grid[:, :, 0], cmap="viridis")
axes[1].set_title("PC1 (Axis 1 -> R)")
axes[2].imshow(pca_grid[:, :, 1], cmap="viridis")
axes[2].set_title("PC2 (Axis 2 -> G)")
axes[3].imshow(pca_grid[:, :, 2], cmap="viridis")
axes[3].set_title("PC3 (Axis 3 -> B)")
axes[4].imshow(pca_grid)
axes[4].set_title("PCA Rainbow (R,G,B)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("pca_rainbow.png", dpi=150)
print("Saved pca_rainbow.png")
