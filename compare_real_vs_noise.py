import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

MODEL_ID = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to("cuda")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
real_inputs = processor(images=image, return_tensors="pt").to("cuda")

noise_pixels = torch.randn(1, 3, 224, 224, device="cuda")

with torch.inference_mode():
    real_out = model(**real_inputs).last_hidden_state
    noise_out = model(pixel_values=noise_pixels).last_hidden_state


real_grid = real_out[0, 5:, :].unflatten(0, (14, 14))    # (14, 14, 384)
noise_grid = noise_out[0, 5:, :].unflatten(0, (14, 14))  # (14, 14, 384)

real_sim = F.cosine_similarity(real_grid, real_grid[7, 7], dim=-1).cpu()    # (14, 14)
noise_sim = F.cosine_similarity(noise_grid, noise_grid[7, 7], dim=-1).cpu() # (14, 14)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image.resize((224, 224)))
axes[0].scatter([7 * 16 + 8], [7 * 16 + 8], c="red", s=80, marker="x")
axes[0].set_title("Input (red X = tile 7,7)")
axes[1].imshow(real_sim, cmap="turbo", vmin=0, vmax=1)
axes[1].set_title("Real Image Cosine Sim")
axes[2].imshow(noise_sim, cmap="turbo", vmin=0, vmax=1)
axes[2].set_title("Random Noise Cosine Sim")
plt.tight_layout()
plt.savefig("similarity_comparison.png", dpi=150)
print("Saved similarity_comparison.png")
