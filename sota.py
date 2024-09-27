from attack_utils import *
import torchvision.transforms as T
from tqdm import tqdm

def generate_sota_targeted(image,p,model,processor,target_label=None,optimizer='adam',lr=1e-3,steps=100,early_stop=5):

    inputs = processor(text = p, images = image, return_tensors="pt").to(model.device)
    im = torch.nn.Parameter(inputs.pop('pixel_values').to(model.device), requires_grad=True)
    optimizer = initialize_optimizer(optimizer,im,lr)
    
    means = processor.image_processor.image_mean
    stds = processor.image_processor.image_std

    device = model.device
    
    with tqdm(range(steps), total=steps) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            model_output = model(**inputs,pixel_values=im)
            word_logits = model_output.logits[0][0].unsqueeze(0)
        
            if target_label is None:
                eos = processor.tokenizer.special_tokens_map['eos_token'] 
                target_label = torch.tensor([get_word_index(processor,eos)]).long().to(device)
            loss = torch.nn.CrossEntropyLoss()(word_logits,target_label)
        
            loss.backward()
        
            if im.grad is None:
                raise Exception("image is not being updated, check gradients...")
            optimizer.step()
            for channel in range(3):
                im[:,channel,:].data.clamp_((0-means[channel])/stds[channel], (1-means[channel])/stds[channel])
            
            pbar.set_postfix(loss=loss.item())  
            
    im = im.detach().clone()
    for c in range(3):
            im[0,c,:] *= processor.image_processor.image_std[c]
            im[0,c,:] += processor.image_processor.image_mean[c]
       
    return T.ToPILImage()(im[0])



def generate_sota_untargeted_PRM(image,boxes,label,p,model,vlm,processor,patch_dim=14,optimizer='adam',lr=1e-3,steps=100,early_stop=5,perturbation_budget=None,path='./',checkpoint=100,img_name='test.png'):    
    inputs = processor(text = p, images = image, return_tensors="pt").to(model.device)
    im = torch.nn.Parameter(inputs.pop('pixel_values').to(model.device), requires_grad=True)
    optimizer = initialize_optimizer(optimizer,im,lr)

    list_patches = get_target_patches(image,boxes,im.shape[-2],im.shape[-1],14)
    means = processor.image_processor.image_mean
    stds = processor.image_processor.image_std

    device = model.device

    start = time.time()
    early_stopping = EarlyStopping(patience=early_stop, delta=0.001)
    
    #register clean features:
    encoder = encoder_QKV(vlm,model)
   
    clean_features = {}
    def get_clean_activation(name):
                def hook_fn(module, input, output):
                    clean_features[name]=output.detach()
                return hook_fn
        
    list_hooks = []
    for idx,layer in enumerate(encoder["encoder_mlp_norm"]):
        list_hooks.append(layer.register_forward_hook(get_clean_activation(str(idx))))

    model_output = model(**inputs,pixel_values=im)
    
    for i in range(len(encoder["encoder_mlp_norm"])):
        list_hooks[i].remove()
    loss_hist = []

    with tqdm(range(steps), total=steps) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            
            adv_features = {}
            def get_adv_activation(name):
                def hook_fn(module, input, output):
                    adv_features[name]=output.requires_grad_(True)
                return hook_fn
        
            list_hooks = []
            for idx,layer in enumerate(encoder["encoder_mlp_norm"]):
                list_hooks.append(layer.register_forward_hook(get_adv_activation(str(idx))))
            
            model_output = model(**inputs,pixel_values=im)
    
            for i in range(len(encoder["encoder_mlp_norm"])):
                list_hooks[i].remove()
            
            for i in range(len(encoder["encoder_mlp_norm"])):
                loss+= torch.nn.CosineEmbeddingLoss()(clean_features[str(i)][0,list_patches],adv_features[str(i)][0,list_patches],-torch.ones(len(list_patches)).to(device))
                #loss += F.cosine_similarity(clean_features[str(i)][0,list_patches],adv_features[str(i)][0,list_patches]).mean()
            loss.backward()
        
            if im.grad is None:
                raise Exception("image is not being updated, check gradients...")

            if perturbation_budget is not None:
                im.grad.clamp_(-perturbation_budget,perturbation_budget)
            
            optimizer.step()
            
            for channel in range(3):
                im[:,channel,:].data.clamp_((0-means[channel])/stds[channel], (1-means[channel])/stds[channel])
            
            loss_hist.append(loss.item())
            pbar.set_postfix(loss=loss.item())  

            early_stopping(loss.item(), image)
            if early_stopping.early_stop:
                print("early_stopping")
                loss_dict = {"overall":loss_hist}
                plot_losses(loss_dict,save_loss=True,path = f"{path}/hist/{img_name}")

                break
            
            if (step+1) % checkpoint == 0 :
                loss_dict = {"overall":loss_hist}
                plot_losses(loss_dict,save_loss=True,path = f"{path}/hist/{img_name}")

    im = im.detach().clone()
    best_image = early_stopping.best_image
    end = time.time()
    kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
    save_image(im,f"{path}/best/{img_name}_best.png",normalized=True,processor=processor)
    
    f = open(f"{path}/best/{img_name}_best.txt", "a")
    for key in list(kw_args.keys()):
        f.write(f"{key}:{kw_args[key]}\n")
    f.close()
    
    return best_image,end-start
