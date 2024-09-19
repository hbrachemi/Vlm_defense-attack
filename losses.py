import torch
import torch.nn.functional as F
from loss_utils import *



class CustomMHAttentionLoss(torch.nn.Module):
    def __init__(self, target_token_indices,qkv_state=False):
        super(CustomMHAttentionLoss, self).__init__()
        
        self.target_token_indices = target_token_indices  
        self.qkv_state = qkv_state
        
    def forward(self, Q, K, QKV = None ,num_heads=16):
        if not self.qkv_state:
            batch_size, seq_len, embed_dim = Q.size()
            head_dim = embed_dim // num_heads
        
            if embed_dim % num_heads != 0:
                raise ValueError("Embedding dimension must be divisible by the number of heads.")
    
            _, attention_weights = self_attention_MH(Q,K,None,None)

        if self.qkv_state:
            batch_size, seq_len, concat_embed_dim = QKV.size()
            embed_dim = concat_embed_dim // 3
            head_dim = embed_dim // num_heads
            
            if embed_dim % num_heads != 0:
                raise ValueError("Embedding dimension must be divisible by the number of heads.")
            
            _, attention_weights = self_attention_MH(None,None,None,QKV)
        target_attention_weights = attention_weights[:,:,:, self.target_token_indices]  
        
        if target_attention_weights.shape[-1]!= len(self.target_token_indices):
                raise ValueError("Q,K or QKV must be of shape Batch_size x num_heads x len_seq x len_seq")

        loss = target_attention_weights.mean()  

        return loss

class CustomMHCEAttentionLoss(torch.nn.Module):
    def __init__(self, target_token_indices,qkv_state=False):
        super(CustomMHCEAttentionLoss, self).__init__()
        self.target_token_indices = target_token_indices  
        self.qkv_state = qkv_state

    def forward(self, Q, K, QKV = None ,num_heads=16):
        if not self.qkv_state:
            device = Q.device
            batch_size, seq_len, embed_dim = Q.size()
            head_dim = embed_dim // num_heads
        
            if embed_dim % num_heads != 0:
                raise ValueError("Embedding dimension must be divisible by the number of heads.")
    
            _, attention_weights = self_attention_MH(Q,K,None,None)

        if self.qkv_state:
            device = QKV.device
            batch_size, seq_len, concat_embed_dim = QKV.size()
            embed_dim = concat_embed_dim // 3
            head_dim = embed_dim // num_heads
            
            if embed_dim % num_heads != 0:
                raise ValueError("Embedding dimension must be divisible by the number of heads.")
    
            _, attention_weights = self_attention_MH(None,None,None,QKV)


        y = torch.ones(attention_weights.size()).to(device)
        y[:,:,:,self.target_token_indices] = 0 

        loss = torch.nn.CrossEntropyLoss()(attention_weights.flatten(),y.flatten())
        

        return loss


class CustomEntropyLoss(torch.nn.Module):
    def __init__(self, target_token_indices):
        super(CustomEntropyLoss, self).__init__()
        self.target_token_indices = target_token_indices  

    def forward(self, V):
        activation_probs = torch.nn.functional.softmax(V, dim=-1)

        log_activation_probs = torch.log(activation_probs + 1e-7) 

        entropy = -torch.sum(activation_probs * log_activation_probs, dim=-1)  
        loss = entropy[self.target_token_indices].mean()
        return loss



