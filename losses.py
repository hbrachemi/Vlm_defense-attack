import torch

class CustomAttentionLoss(torch.nn.Module):
    def __init__(self, target_token_indices):
        super(CustomAttentionLoss, self).__init__()
        self.target_token_indices = target_token_indices  

    def forward(self, Q, K):
        
        d_k = Q.size(-1)  
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        target_attention_weights = attention_weights[:, self.target_token_indices]  
        loss = target_attention_weights.mean()  

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


class CustomCEAttentionLoss(torch.nn.Module):
    def __init__(self, target_token_indices):
        super(CustomCEAttentionLoss, self).__init__()
        self.target_token_indices = target_token_indices  

    def forward(self, Q, K):
        d_k = Q.size(-1)  
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        y = torch.ones(attention_weights.size()).to(Q.device)
        y[:, self.target_token_indices] = 0 
        
        loss = torch.nn.CrossEntropyLoss()(attention_weights.flatten(),y.flatten())
        

        return loss

class CustomPreSoftmaxAttentionLoss(torch.nn.Module):
    def __init__(self, target_token_indices):
        super(CustomPreSoftmaxAttentionLoss, self).__init__()
        self.target_token_indices = target_token_indices  

    def forward(self, Q, K):
        d_k = Q.size(-1)  
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        target_attention_weights = attention_scores[:, self.target_token_indices]  
        loss = target_attention_weights.mean()  

        return loss
