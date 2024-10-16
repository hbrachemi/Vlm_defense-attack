import torch
import torch.nn.functional as F
from model_utils import encoder_QKV

def extract_q_k_v_from_qkv(layer_qkv_output,num_heads = 16,vlm='instruct_blip'):
    if vlm == 'instruct_blip':
        batch_size, seq_len, concat_embed_dim = layer_qkv_output.shape
        embed_dim = concat_embed_dim //3
        head_dim = embed_dim // num_heads
        layer_qkv_output = layer_qkv_output.view(batch_size, seq_len, 3, num_heads, embed_dim // num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = layer_qkv_output[0], layer_qkv_output[1], layer_qkv_output[2]
    if vlm == 'Fuyu':
        batch_size, seq_len, concat_embed_dim = layer_qkv_output.shape
        head_dim = concat_embed_dim // (3*num_heads)        
        layer_qkv_output = layer_qkv_output.view(batch_size, seq_len, num_heads, 3, head_dim)
        query, key, value = layer_qkv_output[..., 0, :], layer_qkv_output[..., 1, :], layer_qkv_output[..., 2, :]

    return query, key, value



    
from transformers.models.persimmon.modeling_persimmon import *

def self_attention_MH(activations={}, num_heads=16, dropout_rate=0, vlm = None, model = None, inputs = None):
    """Using multi-head official implementation"""
    if vlm == "Fuyu":
        query,key,value = extract_q_k_v_from_qkv(activations["qkv"],num_heads = num_heads,vlm='Fuyu')
                
        query = activations["q_layernorm"].transpose(1, 2)
        value = value.transpose(1, 2)
        key = activations["k_layernorm"].transpose(1, 2)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        position_ids = torch.arange(0,inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        batch_size, _ ,seq_len, head_dim = query.size()

        rotary_emb = model.language_model.model.layers[0].self_attn.rotary_emb
        cos, sin = rotary_emb(activations["encoder_input_ln_input"][0], position_ids)    
        
        causal_mask = model.language_model.model._update_causal_mask(inputs["attention_mask"], inputs_embeds, position_ids[0], DynamicCache(), model.config.output_attentions)
        value_states = value
        rotary_ndims = int(head_dim * model.config.partial_rotary_factor)

        query_rot, query_pass = (
            query[..., : rotary_ndims],
            query[..., rotary_ndims :],
        )
        key_rot, key_pass = (
            key[..., : rotary_ndims],
            key[..., rotary_ndims :],
        )

        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos.to(query_rot.device), sin.to(query_rot.device))

        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        attn_weights = attn_weights + causal_mask.to(attn_weights.device)

        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)
        attn_weights = model.language_model.model.layers[0].self_attn.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, num_heads, seq_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, model.config.hidden_size)
        
        proj_result, attention_weights = attn_output, attn_weights
        
    if vlm == "Llava-7b" or vlm == "CLIP":
        layer_output_v = activations['V']
        layer_output_k = activations['K']
        layer_output_q = activations['Q']
        
        # Determine batch size, sequence length, and embed_dim from the layer_output_q shape
        batch_size, seq_len, embed_dim = layer_output_q.size()

        # Calculate head_dim
        head_dim = embed_dim // num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads.")
        scale = head_dim ** -0.5

        # Reshape query, key, value to (batch_size, num_heads, seq_len, head_dim)
        layer_output_q = layer_output_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        layer_output_k = layer_output_k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        if layer_output_v is not None:
            layer_output_v = layer_output_v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            value = layer_output_v.contiguous().view(batch_size * num_heads, seq_len, head_dim)
        # Reshape them into (batch_size * num_heads, seq_len, head_dim)
        query = layer_output_q.contiguous().view(batch_size * num_heads, seq_len, head_dim)
        key = layer_output_k.contiguous().view(batch_size * num_heads, seq_len, head_dim)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) * scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply dropout (if applicable)
        attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)
        if layer_output_v is None:
            return None, attention_weights.view(batch_size,num_heads,seq_len,seq_len)
        # Compute the attention output: (batch_size * num_heads, seq_len, head_dim)
        proj_result = torch.bmm(attention_weights, value)
        # Reshape the output back to (batch_size, seq_len, embed_dim)
        proj_result = proj_result.view(batch_size, num_heads, seq_len, head_dim)
        proj_result = proj_result.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attention_weights = attention_weights.view(batch_size,num_heads,seq_len,seq_len)

    if vlm == 'instruct_blip':
        layer_qkv = activations['qkv']

        batch_size, seq_len, concat_embed_dim = layer_qkv.size()
        embed_dim = concat_embed_dim //3
        head_dim = embed_dim // num_heads
        layer_qkv = layer_qkv.view(batch_size, seq_len, 3, num_heads, embed_dim // num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = layer_qkv[0], layer_qkv[1], layer_qkv[2]
    
        scale = head_dim ** -0.5
    
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply dropout (if applicable)
        attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)
        proj_result = torch.matmul(attention_weights, value).permute(0, 2, 1, 3)
        new_proj_result_shape = proj_result.size()[:-2] + (embed_dim,)
        proj_result = proj_result.contiguous().view(new_proj_result_shape)
    
    
    return proj_result, attention_weights

def redistribute_probabilities(A, tokens):
    """
    Redistribute attention in matrix A such that the columns corresponding
    to the indices in `tokens` are set to 0, but the sum of each column remains 1.

    Args:
        A (Batch_size x heads x n x n): the multihead attention matrix where each column sums to 1.
        tokens: A list of column indices that should be set to 0.

    Returns:
        A matrix with the modified attention distribution.
    """
    A = torch.clone(A)  # Clone the tensor to avoid in-place modifications
    n = A.shape[-1]     # n is the number of columns (sequence length)

    for head in range(A.shape[1]):  # Loop over attention heads
        # Sum up attention for the specified tokens
        prob_sum = torch.sum(A[:, head, :, tokens], dim=2, keepdim=True)  # Sum along token columns
        
        # Set the attention values for these tokens to 0
        A[:, head, :, tokens] = 0

        # Get the indices of columns not in tokens
        other_columns = [i for i in range(n) if i not in tokens]

        # Redistribute the summed attention to the other columns
        A[:, head, :, other_columns] += prob_sum / len(other_columns)  # Distribute equally
    
    return A
    
def extract_target_hidden_states(model,inputs,im,vlm,tokens,target_layer,num_heads=16):
    """
     Computes the hidden states if attention coefficients given to ROI patches were 0.

    Args:
        model: Victim VLM model.
        target_layer: layer we want to extract hidden states from.
        

    Returns:
        A vector with the new target hidden states.
    """
    
    encoder = encoder_QKV(vlm,model)
    K = encoder["K"]
    Q = encoder["Q"]
    V = encoder["V"]
    proj = encoder["proj"]
    
    #Extract first QKV
    activations = {}
    
    def get_activation(name):
                def hook_fn(module, input, output):
                    activations[name]=output
                return hook_fn
        
    hook_handle_v = V[0].register_forward_hook(get_activation('V'))
    hook_handle_k = K[0].register_forward_hook(get_activation('K'))
    hook_handle_q = Q[0].register_forward_hook(get_activation('Q'))
    
    model_output = model(**inputs,pixel_values=im)
    
    hook_handle_v.remove()                
    hook_handle_k.remove()                
    hook_handle_q.remove()                

    layer_output_v = activations['V']
    layer_output_k = activations['K']
    layer_output_q = activations['Q']
                
    
    #Compute first attention matrix
    _, attention_weights = self_attention_MH(layer_output_q, layer_output_k, None)

    #Redistribute attention
    A = redistribute_probabilities(attention_weights, tokens)
    
    #Compute new hidden states if attention of ROI was 0
    batch_size, seq_len, embed_dim = layer_output_q.size()
    head_dim = embed_dim // num_heads
    
    layer_output_v = layer_output_v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = layer_output_v.contiguous().view(batch_size * num_heads, seq_len, head_dim)
    proj_result = torch.bmm(A, value)
    # Reshape the output back to (batch_size, seq_len, embed_dim)
    proj_result = proj_result.view(batch_size, num_heads, seq_len, head_dim)
    proj_result = proj_result.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)


    def create_replace_input_hook(new_input):
        def replace_input_hook(module, input):
            return new_input
        return replace_input_hook
    
    hook_handle = proj[0].register_forward_pre_hook(create_replace_input_hook(proj_result))

    
    activations_target = {}
    
    def get_activation(name):
                def hook_fn(module, input, output):
                    activations_target[name]=output.detach()
                return hook_fn
        
    hook_handle_v = V[target_layer].register_forward_hook(get_activation('V'))
    hook_handle_k = K[target_layer].register_forward_hook(get_activation('K'))
    hook_handle_q = Q[target_layer].register_forward_hook(get_activation('Q'))
    hook_handle_proj = proj[target_layer].register_forward_hook(get_activation('proj'))
    
    model_output = model(**inputs,pixel_values=im)
    
    hook_handle.remove()
    hook_handle_v.remove()                
    hook_handle_k.remove()                
    hook_handle_q.remove()   
    hook_handle_proj.remove()

    return activations_target
   
