classif_prompts = [
    "USER: <image> How would you label this image with a single descriptor?. ASSISTANT:",
]

caption_prompts = [
    "USER: <image> Offer a short description of the subjects present in this image. ASSISTANT:",
]


possible_prompts = classif_prompts+caption_prompts

import pickle 
with open('ImageNetLabels.pkl', 'rb') as file: 
	GT_data = pickle.load(file) 



class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0
        self.best_image = None

    def __call__(self, loss, image):
        
        if self.best_loss is None:
            self.best_loss = loss
            self.best_image = image
        
        elif loss >= self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.best_image = image
            self.counter = 0

import torch.optim as optim
import torch

def initialize_optimizer(optim_name,image,lr):
    if optim_name in ['adam', 'Adam']:
        optimizer = optim.Adam([image], lr=lr)
    if optim_name in ['adamw', 'AdamW']:
        optimizer = optim.AdamW([image], lr=lr)
    if optim_name in ['SGD', 'sgd']:
        optimizer = optim.SGD([image], lr=lr)
    return optimizer

import torch
import torchvision.transforms as T
import os

def depatchify_fuyu_image(patches, original_image_shape, patch_size) -> "torch.Tensor":
        """
        Convert patchified image into a 3-channeled tensor image.
        (Inverse operation of patchify : https://github.com/huggingface/transformers/blob/main/src/transformers/models/fuyu/image_processing_fuyu.py)
        """

        batch_size, channels, orig_height, orig_width = original_image_shape
        patch_height, patch_width = patch_size

        # Number of patches along height and width
        num_patches_height = orig_height // patch_height
        num_patches_width = orig_width // patch_width
    
        # Reshape the patches to (batch_size, num_patches_height, num_patches_width, patch_height, patch_width, channels)
        patches = patches.view(batch_size, num_patches_height, num_patches_width, patch_height, patch_width, channels)
    
        # Permute to match the original image dimensions: (batch_size, channels, height, width)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed_image = patches.view(batch_size, channels, orig_height, orig_width)
    
        return reconstructed_image

def get_target_patches(image,boxes,w,h,patch_dim,input_ids=None,patch_ids=None):
    list_patches = []
    for index in range((w*h)//(patch_dim**2)):        
        l_grid =  index // (w//patch_dim)
        c_grid = index % (h//patch_dim)
        for box in boxes:
            if l_grid*patch_dim >= box[1] and l_grid*patch_dim < box[3] and c_grid*patch_dim >= box[0] and c_grid < box[2]:
                list_patches.append(index+1)
    if input_ids is not None and patch_ids is not None:
        #In Fuyu both image and text modalities are mixed, we have to further retrieve patch_ids in hidden activations
        #Thus we select patches directly from image idx, no need for +1 of cls token
        list_patches = [idx-1 for idx in list_patches]
        dst_indices = torch.nonzero(patch_ids[0] >= 0, as_tuple=True)[0]
        list_patches = dst_indices[list_patches]
        
    return list_patches




def get_word_index(processor,word):
    return processor.tokenizer.vocab[f"{word}"]


from PIL import ImageDraw,Image

def evaluate_image(model,processor,label,path,path_img,kw_args=None,other_prompts=None,vlm='Llava'):
    
    image = Image.open(f"{path_img}")
    
    with open(f"{path}", 'w') as file:
                        file.write(f"Label: {label}\n\n\n")
                        file.flush()

                        if kw_args is not None:
                            for key in list(kw_args.keys()):
                                file.write(f"{key}: {kw_args[key]}\n\n\n")
                                file.flush()
                        if vlm == "CLIP":
                                pos_prompt = f"An image of a {label}"
                                neg_prompt = f"Not an image of a {label}"
                                inputs = processor(text = [pos_prompt,neg_prompt], images = image, return_tensors="pt",padding=True).to(model.device)
                                outputs = model(**inputs)
                                logits_per_image = outputs.logits_per_image
                                probs = logits_per_image.softmax(dim=1)
                                yes_proba = probs[0][0]
                                no_proba = probs[0][1]
                            
                                file.write(f"Yes proba:{yes_proba}\n")
                                file.write(f"No proba:{no_proba}\n")
                                file.flush()
                        else:
                            prompt = f"USER: <image> \nIs there any {label} in the image?\nASSISTANT:"
                            if vlm == 'instruct_blip':
                                prompt = f"Is there any {label} apparent in the image?"  

                            if vlm == 'Fuyu':
                                prompt = f"Is there any {label} in this image?\n"
                            if vlm == 'Blip-2':
                                prompt = f"Question: are there any {label} in this image? Answer:"
                                
                            inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
                            try:
                                inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
                            except:
                                pass
                            model_output = model.generate(**inputs,max_length=1000,output_scores= True,return_dict_in_generate=True)
                            file.write(f"{processor.decode(model_output.sequences[0])}\n\n\n")
                            file.flush()
                            proba_scores = torch.nn.functional.softmax(model_output.scores[0][0],dim=-1)
                            try:
                                yes_proba = proba_scores[get_word_index(processor,"▁Yes")]
                                yes_proba += proba_scores[get_word_index(processor,"▁yes")]
                                no_proba = proba_scores[get_word_index(processor,"▁no")]
                                no_proba += proba_scores[get_word_index(processor,"▁No")]
                        
                                yes_proba += proba_scores[get_word_index(processor,"Yes")] #In instruct blip the prediction is not at the begening of the word
                                yes_proba += proba_scores[get_word_index(processor,"yes")]
                                no_proba += proba_scores[get_word_index(processor,"no")]
                                no_proba += proba_scores[get_word_index(processor,"No")]
                            except:
                                yes_proba = proba_scores[get_word_index(processor,"Yes")] #In instruct blip the prediction is not at the begening of the word
                                yes_proba += proba_scores[get_word_index(processor,"yes")]
                                no_proba = proba_scores[get_word_index(processor,"no")]
                                no_proba += proba_scores[get_word_index(processor,"No")]
                            file.write(f"proba of saying yes: {yes_proba}\nproba of saying no: {no_proba} \n\n")
                            file.flush()

                            prompt = f"USER: <image>.\nASSISTANT:"
                            if vlm == 'instruct_blip':
                                prompt = prompt.replace('USER: <image> ','').replace('ASSISTANT:','').replace("\n","")
                            if vlm == 'Fuyu':
                                prompt = "Offer a short description of the subjects present in this image.\n"
                            inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
                            try:
                                inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
                            except:
                                pass
                            model_output = model.generate(**inputs,max_length=200,output_scores= True,return_dict_in_generate=True)
                            file.write(f"{processor.decode(model_output.sequences[0])}\n\n\n")
                            file.flush()
                            
                            if other_prompts is not None and vlm != "Fuyu":
                                for p in other_prompts:
                                    if vlm == 'instruct_blip':
                                        p = p.replace('USER: <image> ','').replace('ASSISTANT:','').replace("\n","")
                                    if vlm == "Fuyu":
                                        p = p.replace('USER: <image> ','').replace('ASSISTANT:','')
                                    if vlm == 'Blip-2':
                                        p = p.replace('USER: <image> ','Question:').replace('ASSISTANT:','Answer:').replace("\n","")

                                    inputs = processor(text = p, images = image, return_tensors="pt").to(model.device)
                                    try:
                                        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
                                    except: 
                                        pass
                                    model_output = model.generate(**inputs,max_length=1000)
                                    file.write(f"{processor.decode(model_output[0])}\n\n\n")
                                    file.flush()        
import torchvision

def save_image(image,path,normalized=False,processor=None,patchified=False,original_image_shape=None,patch_size=(30,30)):
    if patchified:
        image = depatchify_fuyu_image(image.clone(), original_image_shape, patch_size)
    if not normalized:
        torchvision.utils.save_image(image,f"{path}")
    if normalized and processor is not None:
        image = torch.clone(image)
        
        means = processor.image_processor.image_mean
        stds = processor.image_processor.image_std

        if not isinstance(means, (list, tuple)):
            means = [means, means, means]
        if not isinstance(stds, (list, tuple)):
            stds = [stds, stds, stds]
   
        for c in range(3):
            image[0,c,:] *= stds[c]
            image[0,c,:] += means[c]
            torchvision.utils.save_image(image,f"{path}")


from matplotlib import pyplot as plt
def plot_losses(losses,save_loss=True,path = None):
    for loss_id in list(losses.keys()):
        plt.plot(range(len(losses[loss_id])),losses[loss_id])
        plt.xlabel('epochs')
        plt.ylabel(loss_id)
        if save_loss and path is not None:
            plt.savefig(f"{path}_{loss_id}.png")
        plt.show()

def check_model_recognition(model,processor,image,label,vlm):

    if vlm == "CLIP":
            pos_prompt = f"An image of a {GT_data[label[0]]}"
            neg_prompt = f"Not an image of a {GT_data[label[0]]}"
            inputs = processor(text = [pos_prompt,neg_prompt], images = image, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            yes_proba = probs[0][0]
            no_proba = probs[0][1]
        
            return yes_proba > no_proba
                           
    prompt = f"USER: <image> \nIs there any {GT_data[label[0]]} in the image?\nASSISTANT:"
    
    if vlm == 'instruct_blip':
        prompt = f"Is there any {GT_data[label[0]]} apparent in the image?"
    elif vlm =='Blip-2':
        prompt = f"Question: are there any {GT_data[label[0]]} in this image? Answer:"
    elif vlm == "Fuyu":
        prompt = f"Is there any {GT_data[label[0]]} in this image?\n"


    inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
    model_output = model.generate(**inputs,max_new_tokens=1)
    model_output = processor.decode(model_output[0])
    return "Yes" in model_output or "yes" in model_output

from torchvision import transforms

def check_attack_convergence(model,processor,image,label,vlm,id=None,patchified=False,original_image_shape=None,patch_size=(30,30)):
    
    pil_image = torch.clone(image)
    
    save_image(pil_image,f"./{id}.png",normalized=True,processor=processor,patchified=patchified,original_image_shape=original_image_shape,patch_size=patch_size)
    
    pil_image = Image.open(f"./{id}.png")
    
    if vlm == "CLIP":
            pos_prompt = f"An image of a {GT_data[label[0]][0]}"
            neg_prompt = f"Not an image of a {GT_data[label[0]][0]}"
            inputs = processor(text = [pos_prompt,neg_prompt], images = pil_image, return_tensors="pt", padding=True).to(model.device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            yes_proba = probs[0][0]
            no_proba = probs[0][1]
        
            return yes_proba < no_proba

    prompt = f"USER: <image> \nIs there any {GT_data[label[0]]} in the image?\nASSISTANT:"
    if vlm == 'instruct_blip':
        prompt = f"Is there any {GT_data[label[0]]} apparent in the image?"
    elif vlm =='Blip-2':
        prompt = f"Question: are there any {GT_data[label[0]]} in this image? Answer:"
    elif vlm =='Fuyu':
        prompt = f"Are there any {GT_data[label[0]]} in this image?\n"
    
    inputs = processor(text = prompt, images = pil_image, return_tensors="pt").to(model.device)
    model_output = model.generate(**inputs,max_new_tokens =1)
    model_output = processor.decode(model_output[0])
    bool_no = "No" in model_output or "no" in model_output or "</s>No" in model_output or "</s>no" in model_output
    bool_yes = "Yes" in model_output or "yes" in model_output
    os.remove(f"./{id}.png")
    return  bool_no or (not bool_yes)
    
from losses import *
from tqdm import tqdm
import time

def generate_adv_image_(image,label,boxes,model,processor,optimizer,lr,p_budget=1,targeted_block=-1,lambda_a=1,lambda_e=0,lambda_n=0,lambda_p=0,w=336,h=336,patch_dim=14,steps=1000,num_heads=None,checkpoint=100,path='./',img_name='test',att='mean',early_stop=5,check_convergence_rate=100,vlm='Llava-7b'):
    original_image_shape = (1,3,image.size[1],image.size[0])
    encoder = encoder_QKV(vlm,model)
    
    if vlm =='Fuyu':
        inputs = processor(text = "An image of:\n", images = image).to(device)
        
        im = torch.nn.Parameter(inputs.pop('image_patches')[0].to(model.device), requires_grad=True)
        patchified = True
        input_ids = inputs["input_ids"]
        patch_ids = inputs["image_patches_indices"]
        im_id = "image_patches"

    else:
        patchified = False
        inputs = processor(text = "USER: <image>\nASSISTANT:", images = image, return_tensors="pt").to(model.device)
        im = torch.nn.Parameter(inputs.pop("pixel_values").to(model.device), requires_grad=True)        
        input_ids = None
        patch_ids = None
        im_id = "pixel_values"

    list_patches = list(get_target_patches(image,boxes,w,h,patch_dim,input_ids,patch_ids))
    means = processor.image_processor.image_mean
    if not isinstance(means, (list, tuple)):
        means = [means, means, means]
    stds = processor.image_processor.image_std
    if not isinstance(stds, (list, tuple)):
        stds = [stds, stds, stds]

    optimizer = initialize_optimizer(optimizer,im,lr)    
    
    loss_hist = []
    loss_hist_a = []
    loss_hist_e = []
    loss_hist_n = []
    loss_hist_p = []
    
    if att == 'mean':
        att_loss = CustomMHAttentionLoss(list_patches)
    
    entropy_loss = CustomEntropyLoss(target_token_indices=list_patches)

    start = time.time()
    early_stopping = EarlyStopping(patience=early_stop, delta=0.001)

    save_image(im,f"{path}/adv_img/{img_name}_step_{0}.png",normalized=True,processor=processor,patchified = patchified,original_image_shape=original_image_shape,patch_size=(patch_dim,patch_dim))
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/predictions/{img_name}_step_{0}.txt",f"{path}/adv_img/{img_name}_step_{0}.png",vlm=vlm)
    init_im = torch.clone(im)

    
    for step in tqdm(range(steps)):
            epoch_loss = 0
            optimizer.zero_grad()            
            model_activations = {}    
                        
            def get_activation(name):
                def hook_fn(module, input, output):
                    model_activations[name]=output
                return hook_fn
                
            def get_input(name):
                def hook_fn(module, input, output):
                    model_activations[name]=input
                return hook_fn
            loss_a = torch.zeros(1)
            loss_e = torch.zeros(1)
            loss_n = torch.zeros(1)
            loss_p = torch.zeros(1)
            list_hooks = []
            for l in range(targeted_block):
                for l_key in list(encoder.keys()):
                    list_hooks.append(encoder[l_key][l].register_forward_hook(get_activation(f"{l_key}_{l}")))
                    list_hooks.append(encoder[l_key][l].register_forward_hook(get_input(f"{l_key}_{l}_input")))
                    
            if vlm == "Fuyu":
                model_output = model(**inputs,image_patches=im)
            else:
                model_output = model(**inputs,pixel_values=im)
                
            for i in range(len(list_hooks)):
                list_hooks[i].remove()                
            
            for l in range(targeted_block):
                block_acti = {}
                for l_key in list(encoder.keys()):
                    try:
                        block_acti[l_key] = model_activations[f"{l_key}_{l}"]
                        block_acti[f"{l_key}_input"] = model_activations[f"{l_key}_{l}_input"]
                    except:
                        pass
                if lambda_a!=0:
                            loss_a += att_loss(block_acti,num_heads,dropout_rate=0, vlm = vlm, model = model, inputs = inputs).to(loss_a.device)
                if 'V' in list(block_acti.keys()):
                    layer_output_v = block_acti["V"]
                else:
                    query,key,layer_output_v = extract_q_k_v_from_qkv(block_acti["qkv"],num_heads = num_heads,vlm=vlm)
                
                if vlm in ['Fuyu','instruct_blip']:
                    layer_output_v = layer_output_v.permute(0,2,1,3)
                
                elif lambda_n != 0:
                    print(layer_output_v[0,list_patches].shape)
                    raise Exception("check if value is correctly formatted!")
                
                if lambda_e !=0:
                            loss_e += entropy_loss(layer_output_v).to(loss_a.device)
                if lambda_n !=0:
                                loss_n += layer_output_v[0,list_patches].norm(dim=-1).mean().to(loss_a.device)
                        
                        
            if lambda_p !=0:
                loss_p += (init_im - im).norm()
                        
            loss = lambda_a*loss_a + lambda_e*loss_e + lambda_n*loss_n + lambda_p*loss_p 
            
            loss.backward(retain_graph=True)
            
            optimizer.step()
            if im.grad is None:
                raise Exception("Image grad is None")
            
            epoch_loss += loss.item()

            for channel in range(3):
                im[:,channel,:].data.clamp_((0-means[channel])/stds[channel], (1-means[channel])/stds[channel])
            if p_budget is not None:
                im.data.clamp_(init_im - p_budget, init_im + p_budget)
            loss_hist.append(epoch_loss)
            loss_hist_a.append(loss_a.item())
            loss_hist_e.append(loss_e.item())
            loss_hist_n.append(loss_n.item())
            loss_hist_p.append(loss_p.item())

            early_stopping(epoch_loss, image)
            if early_stopping.early_stop:
                print("early_stopping")
                break
            
            if (step+1) % checkpoint == 0 :
                end = time.time()
                    
                kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
                save_image(im,f"{path}/adv_img/{img_name}_step_{step+1}.png",normalized=True,processor=processor,patchified = patchified,original_image_shape=original_image_shape,patch_size=(patch_dim,patch_dim))
                
                evaluate_image(model,processor,GT_data[label[0]],f"{path}/predictions/{img_name}_step_{step+1}.txt",f"{path}/adv_img/{img_name}_step_{step+1}.png",kw_args,vlm=vlm)

                loss_dict = {"overall":loss_hist}
                if lambda_a !=0:
                    loss_dict["att"]=loss_hist_a
                if lambda_e !=0:
                    loss_dict["entropy"]=loss_hist_e
                if lambda_n !=0:
                    loss_dict["norm"]=loss_hist_n
                if lambda_p !=0:
                    loss_dict["p"]=loss_hist_p
                
                plot_losses(loss_dict,save_loss=True,path = f"{path}/hist/{img_name}")
                    
                start -= time.time()-end
            if (step+1) % check_convergence_rate == 0 :
                if check_attack_convergence(model,processor,im.clone(),label,vlm,id=os.getpid(),patchified = patchified,original_image_shape=original_image_shape,patch_size=(patch_dim,patch_dim)):
                    break
            
    best_image = early_stopping.best_image
    end = time.time()
    optimizer.zero_grad()
    
    del model_activations
    del block_acti

    kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
    save_image(im,f"{path}/best/{img_name}_best.png",normalized=True,processor=processor,patchified = patchified,original_image_shape=original_image_shape,patch_size=(patch_dim,patch_dim))
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/best/{img_name}_best.txt",f"{path}/best/{img_name}_best.png",kw_args,possible_prompts,vlm=vlm)
    return best_image, end-start, im
