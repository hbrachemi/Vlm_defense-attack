classif_prompts = [
    "USER: <image> How would you label this image with a single descriptor?. ASSISTANT:",
    "USER: <image> image of  \nASSISTANT:",
    "USER: <image> If you were a classification model, which label would you attribute to the image? ASSISTANT:",
]

caption_prompts = [
    "USER: <image> Elaborate on the elements present in this image. ASSISTANT:",
    "USER: <image> Relate the main components of this picture in words. ASSISTANT:",
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



def get_target_patches(image,boxes,w,h,patch_dim):
    list_patches = []
    for index in range((w*h)//(patch_dim**2)):        
        l_grid =  index // (w//patch_dim)
        c_grid = index % (h//patch_dim)
        for box in boxes:
            if l_grid*patch_dim >= box[1] and l_grid*patch_dim <= box[3] and c_grid*patch_dim >= box[0] and c_grid <= box[2]:
                list_patches.append(index+1)
    return list_patches


def get_word_index(processor,word):
    return processor.tokenizer.vocab[f"▁{word}"]


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
                        
                        prompt = f"USER: <image> \nIs there any {label} apparent in the image?\nASSISTANT:"
                        if vlm == 'instruct_blip':
                            prompt = prompt.replace('USER: <image> ','').replace('ASSISTANT:','')
                        
                        inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
                        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        
                        model_output = model.generate(**inputs,max_length=1000,output_scores= True,return_dict_in_generate=True)
                        file.write(f"{processor.decode(model_output.sequences[0])}\n\n\n")
                        file.flush()
                        proba_scores = torch.nn.functional.softmax(model_output[1][0][0],dim=-1)
                        yes_proba = proba_scores[get_word_index(processor,"Yes")]
                        no_proba = proba_scores[get_word_index(processor,"No")]
                        file.write(f"proba of saying yes: {yes_proba}\nproba of saying no: {no_proba} \n\n")
                        file.flush()

                        prompt = f"USER: <image>.\nASSISTANT:"
                        if vlm == 'instruct_blip':
                            prompt = prompt.replace('USER: <image> ','').replace('ASSISTANT:','').replace("\n","")

                        inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
                        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        
                        model_output = model.generate(**inputs,max_length=1000,output_scores= True,return_dict_in_generate=True)
                        file.write(f"{processor.decode(model_output.sequences[0])}\n\n\n")
                        file.flush()

                        if other_prompts is not None:
                            for p in other_prompts:
                                if vlm == 'instruct_blip':
                                    p = p.replace('USER: <image> ','').replace('ASSISTANT:','').replace("\n","")
                                inputs = processor(text = p, images = image, return_tensors="pt").to(model.device)
                                inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
                                model_output = model.generate(**inputs,max_length=1000)
                                file.write(f"{processor.decode(model_output[0])}\n\n\n")
                                file.flush()
        
import torchvision

def save_image(image,path,normalized=False,processor=None):
    if not normalized:
        torchvision.utils.save_image(image,f"{path}")
    if normalized and processor is not None:
        image = torch.clone(image)
        for c in range(3):
            image[0,c,:] *= processor.image_processor.image_std[c]
            image[0,c,:] += processor.image_processor.image_mean[c]
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
    if vlm == 'instruct_blip':
        prompt = f"Is there any {GT_data[label[0]]} apparent in the image?"
    else:
        prompt = f"USER: <image> \nIs there any {GT_data[label[0]]} apparent in the image?\nASSISTANT:"
    inputs = processor(text = prompt, images = image, return_tensors="pt").to(model.device)
    model_output = model.generate(**inputs,max_new_tokens=1)
    model_output = processor.decode(model_output[0])

    return "Yes" in model_output

def check_attack_convergence(model,processor,image,label,vlm):
    
    if vlm == 'instruct_blip':
        prompt = f"Is there any {GT_data[label[0]]} apparent in the image?\n"
    else:
        prompt = f"USER: <image> \nIs there any {GT_data[label[0]]} apparent in the image?\nASSISTANT:"
    inputs = processor(text = prompt, images = torch.zeros((3,224,224)), return_tensors="pt").to(model.device)
    inputs["pixel_values"] = image
    model_output = model.generate(**inputs,max_new_tokens =1)
    model_output = processor.decode(model_output[0])

    return "No" in model_output or "no" in model_output
    
from losses import *
from tqdm import tqdm
import time

def generate_adv_image_(image,label,boxes,model,processor,optimizer,lr,target_layers,lambda_a=1,lambda_e=0,lambda_n=0,lambda_p=0,lambda_pre_proj=0,w=336,h=336,patch_dim=14,steps=1000,checkpoint=100,path='./',img_name='test',att='mean',early_stop=5,check_convergence_rate=100,vlm='Llava-7b'):
    
    if not target_layers["qkv_state"]: #In some architectures qkv are projected using separated layers such as Llava
        target_layers_q = target_layers["q"]
        target_layers_k = target_layers["k"]
        target_layers_v = target_layers["v"]
        target_layers_proj = target_layers["proj"]  
    
    if target_layers["qkv_state"]: #In some architectures such as Insctruct-Blip qkv projections are done using one projection layer, the projections are thus concatenated and needs to be separated for future use
        target_layers_qkv = target_layers["qkv"]
        target_layers_proj = target_layers["proj"]
        

    
    list_patches = get_target_patches(image,boxes,w,h,patch_dim)
    means = processor.image_processor.image_mean
    stds = processor.image_processor.image_std


    inputs = processor(text = "USER: <image>\nASSISTANT:", images = image, return_tensors="pt").to(model.device)
    im = torch.nn.Parameter(inputs.pop('pixel_values').to(model.device), requires_grad=True)
    optimizer = initialize_optimizer(optimizer,im,lr)    
    
    loss_hist = []
    loss_hist_a = []
    loss_hist_e = []
    loss_hist_n = []
    loss_hist_p = []
    loss_hist_proj = []
    
    if att == 'mean':
        att_loss = CustomMHAttentionLoss(list_patches,target_layers["qkv_state"])
    if att in ["ce","CE"]:
        att_loss = CustomMHCEAttentionLoss(list_patches,target_layers["qkv_state"])
    
    entropy_loss = CustomEntropyLoss(target_token_indices=list_patches)

    start = time.time()
    early_stopping = EarlyStopping(patience=early_stop, delta=0.001)

    save_image(im,f"{path}/adv_img/{img_name}_step_{0}.png",normalized=True,processor=processor)
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/predictions/{img_name}_step_{0}.txt",f"{path}/adv_img/{img_name}_step_{0}.png",vlm=vlm)
    init_im = torch.clone(im)

    
    for step in tqdm(range(steps)):
            epoch_loss = 0
            optimizer.zero_grad()            
            activations = {}    
                        
            def get_activation(name):
                def hook_fn(module, input, output):
                    activations[name]=output
                return hook_fn
            def get_input(name):
                def hook_fn(module, input, output):
                    activations[name]=input
                return hook_fn
            loss_a = torch.zeros(1).to(model.device)
            loss_e = torch.zeros(1).to(model.device)
            loss_n = torch.zeros(1).to(model.device)
            loss_p = torch.zeros(1).to(model.device)
            loss_proj = torch.zeros(1).to(model.device)


        
            for l in range(len(target_layers_proj)):
                    if not target_layers["qkv_state"]:
                        hook_handle_v = target_layers_v[l].register_forward_hook(get_activation('V'))
                        hook_handle_k = target_layers_k[l].register_forward_hook(get_activation('K'))
                        hook_handle_q = target_layers_q[l].register_forward_hook(get_activation('Q'))
                        hook_handle_proj = target_layers_proj[l].register_forward_hook(get_input('proj'))

                        model_output = model(**inputs,pixel_values=im)
                    
                        layer_output_v = activations['V']
                        layer_output_k = activations['K']
                        layer_output_q = activations['Q']
                        layer_proj = activations['proj'][0]

                        hook_handle_v.remove()                
                        hook_handle_k.remove()                
                        hook_handle_q.remove()                
                        hook_handle_proj.remove()   
                
                        if lambda_a!=0:
                            loss_a += att_loss(layer_output_q, layer_output_k)
                        if lambda_e !=0:
                            loss_e += entropy_loss(layer_output_v)
                        if lambda_n !=0:
                            loss_n += layer_output_v[list_patches].norm(dim=1).mean()
                        if lambda_pre_proj!=0:
                            loss_proj += layer_proj[0][list_patches].norm(dim=1).mean()
                        
                        
                    if target_layers["qkv_state"]:
                        hook_handle_qkv = target_layers_qkv[l].register_forward_hook(get_activation('qkv'))
                        hook_handle_proj = target_layers_proj[l].register_forward_hook(get_input('proj'))
                        
                        model_output = model(**inputs,pixel_values=im)

                        layer_output_qkv = activations['qkv']
                        layer_proj = activations['proj'][0]
                        _ , _ , value = extract_q_k_v_from_qkv(layer_output_qkv)
                        
                        hook_handle_qkv.remove()   
                        hook_handle_proj.remove() 

                        if lambda_a!=0:
                            loss_a += att_loss(None, None,layer_output_qkv)
                        if lambda_e !=0:
                            loss_e += entropy_loss(value)
                        if lambda_n !=0:
                            loss_n += value[:,:,list_patches,:].norm()
                        if lambda_pre_proj!=0:
                            loss_proj += layer_proj[0][list_patches].norm(dim=1).mean()
                
                        
            if lambda_p !=0:
                loss_p += (init_im - im).norm()
                        
            loss = lambda_a*loss_a + lambda_e*loss_e + lambda_n*loss_n + lambda_p*loss_p + lambda_pre_proj*loss_proj

            loss.backward(retain_graph=True)
            optimizer.step()
            
            if im.grad is None:
                raise Exception("Image grad is None")
            
            epoch_loss += loss.item()

            for channel in range(3):
                im[:,channel,:].data.clamp_((0-means[channel])/stds[channel], (1-means[channel])/stds[channel])
    
            loss_hist.append(epoch_loss)
            loss_hist_a.append(loss_a.item())
            loss_hist_e.append(loss_e.item())
            loss_hist_n.append(loss_n.item())
            loss_hist_p.append(loss_p.item())
            loss_hist_proj.append(loss_proj.item())

            early_stopping(epoch_loss, image)
            if early_stopping.early_stop:
                print("early_stopping")
                break
            
            if (step+1) % checkpoint == 0 :
                end = time.time()

                kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
                save_image(im,f"{path}/adv_img/{img_name}_step_{step+1}.png",normalized=True,processor=processor)
                
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
                if lambda_pre_proj !=0:
                    loss_dict["proj"]=loss_hist_proj
                
                plot_losses(loss_dict,save_loss=True,path = f"{path}/hist/{img_name}")
                    
                start -= time.time()-end
            if (step+1) % check_convergence_rate == 0 :
                if check_attack_convergence(model,processor,im,label,vlm):
                    break
            
    best_image = early_stopping.best_image
    end = time.time()
    kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
    save_image(im,f"{path}/best/{img_name}_best.png",normalized=True,processor=processor)
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/best/{img_name}_best.txt",f"{path}/best/{img_name}_best.png",kw_args,possible_prompts,vlm=vlm)
    return best_image, end-start, im
