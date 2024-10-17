import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from skimage.metrics import structural_similarity as ssim
import lpips
import numpy as np

lpips_metric = lpips.LPIPS(net='alex').to(device)

def compute_perceptual_similarity(ref_image_path,image_path):
    scores = {}
    ref =Image.open(ref_image_path)
    im =Image.open(image_path)

    scores["ssim"], _ = ssim(np.array(ref.convert("L")), np.array(im.convert("L")), full=True)


    ref_tensor = torch.tensor(np.array(ref.convert('RGB')).transpose(2, 0, 1)).unsqueeze(0)    
    img_tensor = torch.tensor(np.array(im.convert('RGB')).transpose(2, 0, 1)).unsqueeze(0)

    scores["lpips"] = lpips_metric(ref_tensor.float().to(device) / 255.0, img_tensor.float().to(device) / 255.0)
    scores["lpips"] = scores["lpips"].cpu().detach().numpy()

    return scores


import pickle 
with open('ImageNetLabels.pkl', 'rb') as file: 
	GT_data = pickle.load(file) 


def evaluate_image_context(model,processor,label,path_img,vlm):
    
    image = Image.open(f"{path_img}")
    if vlm == 'Llava-7b':
        prompt = f"USER: <image> \nHow would you describe the image without the {label}?\nASSISTANT:"
    if vlm == 'Blip-2':
        prompt = f"Question: How would you describe the image without the {label}?.Answer:"
    if vlm == 'Instruct-blip':
        prompt = f"How would you describe the image without the {label}?"
    if vlm is None:
        prompt = f"How would you describe the image without the {label}?"

    inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        
    model_output = model.generate(**inputs,max_length=1000,output_scores= True,return_dict_in_generate=True)
    return processor.decode(model_output.sequences[0])
