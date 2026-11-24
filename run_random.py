import torch
from transformers import AutoModel

MODEL_ID = "computer-vision-ai-lab/dinov3-vits16-pretrain-lvd1689m"
model = AutoModel.from_pretrained(MODEL_ID).to("cuda")

x = torch.randn(1,3,224,224, device = "cuda")

with torch.inference_mode():
    output=model(pixel_values=x)

output_last_slice = output.last_hidden_state

"""print("output shape is: ", output.shape)"""
print("output last slice shape is: ",output_last_slice.shape)

cls_tokens = output_last_slice[:,0,:]
register_tokens = output_last_slice[:,1:5,:]
patch_tokens = output_last_slice[:,5:,:]
patch_grid = patch_tokens.unflatten(1, (14, 14))

print("cls tokens shape is: ",cls_tokens.shape)
print("register tokens shape is: ",register_tokens.shape)
print("patch tokens shape is: ",patch_tokens.shape)
print("patch grid shape is: ",patch_grid.shape)
