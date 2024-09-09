""" Random functions used during experimentations """

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def wrap_text(draw, text, font, max_width):
    """ preprocesses text so that it fits in an image """
    lines = []
    words = text.split()
    while words:
        line = ''
        while words and draw.textbbox((0, 0), line + words[0], font=font)[2] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines

def text_to_image(text, font_path='arial.ttf', font_size=20, image_size=(224, 224), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """ converts text to image """
    img = Image.new('RGB', image_size, color=bg_color)

    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    max_width = image_size[0] - 20  
    lines = wrap_text(draw, text, font, max_width)
    
    total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)

    y = (image_size[1] - total_text_height) // 2  
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        x = (image_size[0] - line_width) // 2  
        draw.text((x, y), line, fill=text_color, font=font)
        y += draw.textbbox((0, 0), line, font=font)[3]

    return img

def generate_embeddings(model,processor,prompts,layer=0,decoder_call=0,modality='txt',context=False,image = None):
    """ generates hidden states of the model using hf provided function"""
    prompts_actii = {}
    for key in prompts.keys():
        prompts_list = prompts[key]
        list_acti = []
        for p in prompts_list:
            if modality == 'img':
                image = text_to_image(p)
                p = f"USER:<image> the image provides a dialogue. You are assumed to be the assistant, complete the provided dialogue. ASSISTANT:"
            if context:
                p.replace("ASSISTANT:","Answer only with 'I am sorry I cannot assist you' if you cannot answer with no further explanation. ASSISTANT:") 
            inputs = processor(text=p,images =image ,return_tensors="pt")
            for input_key in inputs.keys():
                if inputs[input_key] is not None:
                    inputs[input_key] = inputs[input_key].to(model.device) 
            output = model.generate(**inputs,output_hidden_states=True,return_dict_in_generate=True,max_new_tokens=1)
            list_acti.append(output.hidden_states[decoder_call][layer])
        prompts_actii[key] = list_acti
    return prompts_actii


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

def visualize_embeddings(embeddings_dictionary,reduce='mean',algo='PCA',binary_map=None):
    """ visualizes hidden states of the model, function expects a dictionnary with categories as keys e.g. toxic_embeddings, normal_embeddings """
    embeddings_dictionary = embeddings_dictionary.copy()
    if reduce == 'mean':
        for key in embeddings_dictionary.keys():
            embeddings_dictionary[key] = np.concatenate([np.array(emb).mean(1) for emb in embeddings_dictionary[key]])
    elif reduce == 'flatten':
        for key in embeddings_dictionary.keys():
            embeddings_dictionary[key] = np.concatenate([np.array(emb).reshape(-1, 4096) for emb in embeddings_dictionary[key]])
    
    all_embeddings = np.vstack(list(embeddings_dictionary.values()))
    
    if algo in ['PCA','pca']:
        pca = PCA(n_components=2)
        pca.fit(all_embeddings)
        color = iter(cm.rainbow(np.linspace(0, 1, len(embeddings_dictionary.keys()))))
        for key in embeddings_dictionary.keys():
            c = next(color)
            pca_data = pca.transform(embeddings_dictionary[key])
            if binary_map is None or key =='regular':
                plt.scatter(pca_data[:, 0], pca_data[:, 1], color=c,label=str(key),marker='.')
            else:
                pca_data_aligned = pca_data[binary_map[key]==1]
                pca_data_not_aligned = pca_data[binary_map[key] == 0]
                plt.scatter(pca_data_aligned[:, 0], pca_data_aligned[:, 1], color='red',label=str(key),marker='.')
                plt.scatter(pca_data_not_aligned[:, 0], pca_data_not_aligned[:, 1], color='blue',label=str(key),marker='.')
    elif algo in ['TSNE','tsne']:
        tsne = TSNE(n_components=2).fit_transform(all_embeddings)
        color = iter(cm.rainbow(np.linspace(0, 1, len(embeddings_dictionary.keys()))))
        counter = 0
        for key in embeddings_dictionary.keys():
            c = next(color)
            if binary_map is None or key =='regular':
                plt.scatter(tsne[counter:counter+len(embeddings_dictionary[key])][:, 0], tsne[counter:counter+len(embeddings_dictionary[key])][:, 1], color=c,label=str(key),marker='.')
                counter += len(embeddings_dictionary[key])
            else:
                data =tsne[counter:counter+len(embeddings_dictionary[key])]
                data_aligned = data[binary_map[key]==1]
                data_not_aligned = data[binary_map[key] == 0]
                plt.scatter(data_aligned[:, 0], data_aligned[:, 1], color='red',label=str(key),marker='.')
                plt.scatter(data_not_aligned[:, 0], data_not_aligned[:, 1], color='blue',label=str(key),marker='.')
            
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f"{algo} scatter plot of Benign and Toxic Embeddings")
    plt.legend(bbox_to_anchor=(1.12, 0.5), loc='center')
    plt.show()


def visualize_QKV(embeddings_dictionary,reduce='mean',algo='PCA',colors='sep'):
    """ visualize QKV hidden states of the model, function expects a dicitionnary with Q,K,V,O as keys"""
    init_embeddings_dictionary = embeddings_dictionary.copy()
    r_embeddings_dictionary = {}
    keys = ['Q','K','V','O']
    f, axes = plt.subplots(1, len(keys), figsize=(10, 10))
    axes = axes.flatten()
    for idx, (ax, i) in enumerate(zip(axes, keys)):
        if reduce == 'mean':
            for key in init_embeddings_dictionary.keys():
                r_embeddings_dictionary[key] = np.concatenate([np.array(emb[i]).mean(1) for emb in init_embeddings_dictionary[key]])
        elif reduce == 'flatten':
            for key in embeddings_dictionary.keys():
                r_embeddings_dictionary[key] = np.concatenate([np.array(emb[i]).reshape(-1, 4096) for emb in init_embeddings_dictionary[key]])
    
        all_embeddings = np.vstack(list(r_embeddings_dictionary.values()))
    
        if algo in ['PCA','pca']:
            pca = PCA(n_components=2)
            pca.fit(all_embeddings)
            color = iter(cm.rainbow(np.linspace(0, 1, len(r_embeddings_dictionary.keys()))))
            for key in r_embeddings_dictionary.keys():
                if colors == 'sep':
                    c = next(color)
                else:
                    if key == 'regular':
                        c = 'blue'
                    else:
                        c = 'red'
                
                pca_data = pca.transform(r_embeddings_dictionary[key])
                ax.scatter(pca_data[:, 0], pca_data[:, 1], color=c,label=str(key),marker='^')
    
        elif algo in ['TSNE','tsne']:
            tsne = TSNE(n_components=2).fit_transform(all_embeddings)
            color = iter(cm.rainbow(np.linspace(0, 1, len(r_embeddings_dictionary.keys()))))
            counter = 0
            for key in init_embeddings_dictionary.keys():
                if colors == 'sep':
                    c = next(color)
                else:
                    if key == 'regular':
                        c = 'blue'
                    else:
                        c = 'red'
                ax.scatter(tsne[counter:counter+len(r_embeddings_dictionary[key])][:, 0], tsne[counter:counter+len(r_embeddings_dictionary[key])][:, 1], color=c,label=str(key),marker='.')
                counter += len(r_embeddings_dictionary[key])
        ax.set_xlabel(f"{i} ")
        ax.set_ylabel(f"{i} ")

    plt.title(f"{algo} scatter plot of Benign and Toxic Embeddings")
    plt.legend(bbox_to_anchor=(1.12, 0.5), loc='center')
    plt.show()

def generate_QKV(model,processor,prompts,decoder_block=0,modality='txt',context=False):
""" Generates QKV hidden states from the decoder using hooks instead of provided function by hf, this function only works for Llava1.5 with the 7b vicuna decoder, 
returns corresponding Q,K,V,O dictionnary"""
    submodule = model.language_model.get_submodule('model').get_submodule('layers')
    leaf_nodes = [module for module in submodule.modules() if len(list(module.children())) == 0]

    Q = [leaf_nodes[0+i*11] for i in range(32)]
    K = [leaf_nodes[1+i*11] for i in range(32)]
    V = [leaf_nodes[2+i*11] for i in range(32)]
    O = [leaf_nodes[3+i*11] for i in range(32)]
    
    activations = {}
    def get_activation(name):
        def hook_fn(module, input, output):
            activations[name]=output
        return hook_fn
    
    image = None
    prompts_actii = {}
    for key in prompts.keys():
        prompts_list = prompts[key]
        list_acti = []
        for p in prompts_list:
            activations = {}   
            ###Q
            hook_handle_q = Q[decoder_block].register_forward_hook(get_activation('Q'))
            ###K
            hook_handle_k = K[decoder_block].register_forward_hook(get_activation('K'))
            ###V
            hook_handle_v = V[decoder_block].register_forward_hook(get_activation('V'))
            ###O
            hook_handle_o = O[decoder_block].register_forward_hook(get_activation('O'))

            if modality == 'img':
                image = text_to_image(p)
                p = f"USER:<image> the image provides a dialogue. You are assumed to be the assistant, complete the provided dialogue. ASSISTANT:"
            if context:
                p.replace("ASSISTANT:","Answer only with 'I am sorry I cannot assist you' if you cannot answer with no further explanation. ASSISTANT:")   
        
            inputs = processor(text=p,images =image ,return_tensors="pt")
            output = model(**inputs)

            hook_handle_q.remove()    
            activations['Q'] = [act.cpu().detach().numpy() for act in activations['Q']]
    
            hook_handle_k.remove()
            activations['K'] = [act.cpu().detach().numpy() for act in activations['K']]
    
            hook_handle_v.remove()
            activations['V'] = [act.cpu().detach().numpy() for act in activations['V']]
    
            hook_handle_o.remove()
            activations['O'] = [act.cpu().detach().numpy() for act in activations['O']]
    
            list_acti.append(activations)
        
        prompts_actii[key] = list_acti
    return prompts_actii


def generate_llava_embeddings_using_hooks(model,processor,prompts,layer=0,modality='txt',context=False,input_output='output'):
    """ Generates a layer's hidden state from the decoder using hooks instead of provided function by hf, this function only works for Llava1.5 with the 7b vicuna decoder"""

    
    image = None
    submodule = model.get_submodule("language_model")
    leaf_nodes = [module for module in submodule.modules()
                  if len(list(module.children())) == 0]
    
    target_layer = leaf_nodes[layer]
    print(f"extracting {input_output} from layer: {target_layer}") 
    prompts_actii = {}
    
    for key in prompts.keys():
        prompts_list = prompts[key]
        list_acti = []
        
        for idx,p in enumerate(tqdm(prompts_list)):
            
            if modality == 'img':
                image = text_to_image(p)
                p = f"USER:<image> the image provides a dialogue. You are assumed to be the assistant, complete the provided dialogue. ASSISTANT:"
            
            if context:
                p.replace("ASSISTANT:","Answer only with 'I am sorry I cannot assist you' if you cannot answer with no further explanation. ASSISTANT:")   

            if input_output == 'input':
                activations = []
                def hook_fn(module, input, output):
                    activations.append(input)
            else:
                activations = []
                def hook_fn(module, input, output):
                    activations.append(output)
            
            hook_handle = target_layer.register_forward_hook(hook_fn)
            
            inputs = processor(text=p,images =image ,return_tensors="pt")
            output = model(**inputs)
            hook_handle.remove()
            activations_np = [act[0].cpu().detach().numpy() for act in activations]           
            list_acti.append(activations_np)
        prompts_actii[key] = list_acti
    
    return prompts_actii

    
