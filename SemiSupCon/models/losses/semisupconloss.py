import torch
import torch.nn as nn
import matplotlib.pyplot as plt

    
class SSNTXent(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.first_run = True
        self.sim_function = nn.CosineSimilarity(2)
        
    def get_similarities(self, features, temperature = None):
        if temperature is None:
            temperature = self.temperature  
        return self.sim_function(features.unsqueeze(1),features.unsqueeze(0))/temperature
        
    def forward(self,features, positive_mask, negative_mask):
        
        ## features shape extended_batch, d_model
        ## mask shape extended_batch,extended_batch
        
        ## add zeros to negative and positive masks to prevent self-contrasting
        
        
        self_contrast = (~(torch.eye(positive_mask.shape[0], device = features.device).bool())).int()
        
        
        positive_mask = positive_mask * self_contrast
        negative_mask = negative_mask * self_contrast
        
    
        original_cosim = self.get_similarities(features=features)    
        
        original_cosim = torch.exp(original_cosim)   ## remove this when reverting
         
    
        pos = torch.sum( original_cosim * positive_mask, dim = 1)
        neg = torch.sum( original_cosim * negative_mask, dim = 1)
        
        log_prob = torch.log(pos / (pos + neg))
        log_prob = log_prob / positive_mask.sum(1)
        
        loss = -(torch.mean(log_prob))    
        
        # loss = - log_prob.sum()/MN ## not directly mean() because size of matrix is MN squared
        
        # loss = - log_prob.mean()
        return loss
    