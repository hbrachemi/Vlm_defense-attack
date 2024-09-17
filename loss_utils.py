import torch

def self_attention_MH(layer_output_q, layer_output_k, layer_output_v, num_heads=16, dropout_rate=0):
    """Using multi-head official implementation"""
    # Determine batch size, sequence length, and embed_dim from the layer_output_q shape
    batch_size, seq_len, embed_dim = layer_output_q.size()
    
    # Calculate head_dim
    head_dim = embed_dim // num_heads
    if embed_dim % num_heads != 0:
        raise ValueError("Embedding dimension must be divisible by the number of heads.")
    
    # Reshape query, key, value to (batch_size, num_heads, seq_len, head_dim)
    layer_output_q = layer_output_q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    layer_output_k = layer_output_k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    if layer_output_v is not None:
        layer_output_v = layer_output_v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Reshape them into (batch_size * num_heads, seq_len, head_dim)
    query = layer_output_q.contiguous().view(batch_size * num_heads, seq_len, head_dim)
    key = layer_output_k.contiguous().view(batch_size * num_heads, seq_len, head_dim)
    if layer_output_v is not None:
        value = layer_output_v.contiguous().view(batch_size * num_heads, seq_len, head_dim)

    # Scaling factor to prevent large dot products
    scale = head_dim ** -0.5
    
    # Compute attention scores: (batch_size * num_heads, seq_len, seq_len)
    attention_scores = torch.bmm(query, key.transpose(1, 2)) * scale
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # Apply dropout (if applicable)
    attention_weights = F.dropout(attention_weights, p=dropout_rate, training=True)
    if layer_output_v is None:
        return None, attention_weights
        
    # Compute the attention output: (batch_size * num_heads, seq_len, head_dim)
    proj_result = torch.bmm(attention_weights, value)
    
    # Reshape the output back to (batch_size, seq_len, embed_dim)
    proj_result = proj_result.view(batch_size, num_heads, seq_len, head_dim)
    proj_result = proj_result.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    return proj_result, attention_weights

