# -----------------------------------------------------
# Code adapted from https://github.com/jacobgil/vit-explain
# Original file: vit_rollout.py
# -----------------------------------------------------

import torchvision
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch



def rollout(attentions, discard_ratio, head_fusion,device):
    result = torch.eye(attentions[0].size(-1)).to(device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result) 
    mask = result[0, 0 , 1 :]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).cpu().detach().numpy()
    mask = mask / np.max(mask)
    return mask    



def plot_heatmaps(pil_im,processor,vision_model,max_layer=-1,min_max_mean="min",patch_dim=32,alpha=0.6,device=None,plot_bar=True):
    inputs = processor(images=pil_im,text="<image>", return_tensors="pt").to(device)
    
    with torch.no_grad():
        im = inputs["pixel_values"]
        outputs = vision_model(im, output_attentions=True)
    attention_maps = outputs.attentions
    mask = rollout(attention_maps[:max_layer],0,min_max_mean,device)
    final_attention = mask.reshape(im.shape[-2]//patch_dim, im.shape[-1]//patch_dim)
    final_attention_resized = cv2.resize(final_attention, (im.shape[-2], im.shape[-1]))
    
    plt.imshow(pil_im)
    hm = plt.imshow(final_attention_resized, cmap='viridis', alpha=alpha)  
    if plot_bar:
        cbar = plt.colorbar(hm)
        cbar.set_label('Intensity')
    plt.axis('off')
    plt.show()
