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
