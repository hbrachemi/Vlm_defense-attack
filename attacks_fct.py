from losses import *
from tqdm import tqdm
import time
from attack_utils import *

def generate_adv_image_rollout(image,label,boxes,model,processor,optimizer,lr,p_budget=None,targeted_block=-1,w=336,h=336,patch_dim=14,steps=1000,checkpoint=100,path='./',
                               img_name='test',head_reduction='mean',early_stop=5,check_convergence_rate=100,vlm='Llava-7b'):
        
    
    inputs = processor(text = "USER: <image>\nASSISTANT:", images = image, return_tensors="pt").to(model.device)
    im = torch.nn.Parameter(inputs.pop("pixel_values"), requires_grad=True)        
    
    list_patches = list(get_target_patches(image,boxes,w,h,patch_dim,input_ids=None,patch_ids=None))
    list_patches = [p-1 for p in list_patches]
    means = processor.image_processor.image_mean
    if not isinstance(means, (list, tuple)):
        means = [means, means, means]
    stds = processor.image_processor.image_std
    if not isinstance(stds, (list, tuple)):
        stds = [stds, stds, stds]

    optimizer = initialize_optimizer(optimizer,im,lr)    
    
    loss_hist = []
    
    start = time.time()
    early_stopping = EarlyStopping(patience=early_stop, delta=0.001)

    save_image(im,f"{path}/adv_img/{img_name}_step_{0}.png",normalized=True,processor=processor,patchified = False,original_image_shape=None,patch_size=(patch_dim,patch_dim))
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/predictions/{img_name}_step_{0}.txt",f"{path}/adv_img/{img_name}_step_{0}.png",vlm=vlm)
    init_im = torch.clone(im.clone())

    
    for step in tqdm(range(steps)):
            epoch_loss = 0
            optimizer.zero_grad()            

            if vlm == "Llava":
                outputs = model.vision_tower(im, output_attentions=True)
            else:
                outputs = model.vision_model(im, output_attentions=True)
            attention_maps = outputs.attentions
            attention_rollout = torch.eye(attention_maps[0].size(-1)).to(attention_maps[0].device)
            discard_ratio = 0
            for attention in attention_maps[:targeted_block]:
                if head_reduction == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_reduction == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_reduction == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                attention_rollout = torch.matmul(a, attention_rollout)
                mask = attention_rollout[0, 0 , 1 :]
                mask = mask / torch.max(mask)
                
            loss = mask[list_patches].mean()
        
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
            early_stopping(epoch_loss, image)
            
            if early_stopping.early_stop:
                print("early_stopping")
                break
            
            if (step+1) % checkpoint == 0 :
                end = time.time()
                    
                kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
                save_image(im,f"{path}/adv_img/{img_name}_step_{step+1}.png",normalized=True,processor=processor,patchified = False,original_image_shape=None,patch_size=(patch_dim,patch_dim))
                
                evaluate_image(model,processor,GT_data[label[0]],f"{path}/predictions/{img_name}_step_{step+1}.txt",f"{path}/adv_img/{img_name}_step_{step+1}.png",kw_args,vlm=vlm)

                loss_dict = {"overall":loss_hist}
                
                plot_losses(loss_dict,save_loss=True,path = f"{path}/hist/{img_name}")
                    
                start -= time.time()-end
            if (step+1) % check_convergence_rate == 0 :
                if check_attack_convergence(model,processor,im.clone(),label,vlm,id=os.getpid(),patchified = False,original_image_shape=None,patch_size=(patch_dim,patch_dim)):
                    break
            
    best_image = early_stopping.best_image
    end = time.time()
    optimizer.zero_grad()
    

    kw_args = {"exec_time":end-start,"num_patches":len(list_patches)}
    save_image(im,f"{path}/best/{img_name}_best.png",normalized=True,processor=processor,patchified = False,original_image_shape=None,patch_size=(patch_dim,patch_dim))
    evaluate_image(model,processor,GT_data[label[0]],f"{path}/best/{img_name}_best.txt",f"{path}/best/{img_name}_best.png",kw_args,possible_prompts,vlm=vlm)
    return best_image, end-start, im



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
        im = torch.nn.Parameter(inputs.pop("pixel_values"), requires_grad=True)        
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
    init_im = torch.clone(im.clone())

    
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
            elif vlm == "Blip-2":
                inputs["decoder_input_ids"] = inputs["input_ids"]
                model_output = model(**inputs,pixel_values=im)
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
                
                if vlm in ['Fuyu','instruct_blip','Blip-2']:
                    layer_output_v = layer_output_v.permute(0,2,1,3)
                
                elif lambda_n != 0 and vlm != 'Llava-7b':
                    print(layer_output_v.shape)
                    print(len(list_patches))
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
