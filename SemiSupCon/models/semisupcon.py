import pytorch_lightning as pl
from SemiSupCon.models.losses.semisupconloss import SSNTXent
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim

class SemiSupCon(pl.LightningModule):
    
    def __init__(self, encoder, temperature = 0.1):
        super().__init__()
        self.loss = SSNTXent(temperature = temperature)
        self.encoder = encoder
        self.proj_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
        )
        
        
    def freeze(self):
        self.freeze()
        self.eval()
        
    def forward(self,x):
        
        wav = x['wav']
        labels = x['labels']
        
        
        # x is of shape [B,N_augmentations,T]:
        B, N_augmentations, T = wav.shape

        x = x.contiguous().view(-1,x.shape[-1]) ## [B*N_augmentations,T]
        
        semisl_contrastive_matrix,ssl_contrastive_matrix, sl_contrastive_matrix = self.get_contrastive_matrices(B,N_augmentations,T,labels)
        
        encoded = self.encoder(x)
        projected = self.proj_head(encoded)
        
        return {
            'projected':projected,
            'semisl_contrastive_matrix':semisl_contrastive_matrix, ## this is the matrix that we will use for the ssl loss
            'ssl_contrastive_matrix':ssl_contrastive_matrix,
            'sl_contrastive_matrix':sl_contrastive_matrix,
            'labels':labels,
            'encoded':encoded
        }
        
    def training_step(self, batch, batch_idx):
            
        x = batch
        out_ = self(x)
        
        labeled = x['labels'].sum(dim=-1) > 0
        
        semisl_loss = self.loss(out_['projected'],out_['semisl_contrastive_matrix'])
        
        self.logging(out_,labeled)
        
        return semisl_loss
    
    
    def validation_step(self,batch,batch_idx):
        x = batch
        out_ = self(x)
        
        semisl_loss = self.loss(out_['projected'],out_['semisl_contrastive_matrix'])
        
        return semisl_loss
    
    def logging(self,out_, labeled): 
        
        semisl_loss = self.loss(out_['projected'],out_['semisl_contrastive_matrix'])
        sl_loss = self.loss(out_['projected'][labeled,:],out_['sl_contrastive_matrix'][labeled, labeled])
        ssl_loss = self.loss(out_['projected'][~labeled,:],out_['ssl_contrastive_matrix'][~labeled, ~labeled])
        
        # get similarities
        semisl_sim = self.loss.get_similarities(out_['projected'])
        ssl_sim = self.loss.get_similarities(out_['projected'][~labeled,:])
        sl_sim = self.loss.get_similarities(out_['projected'][labeled,:])
        
        if self.logger:
            
            self.log('semisl_loss',semisl_loss)
            self.log('sl_loss',sl_loss)
            self.log('ssl_loss',ssl_loss)
            
            if self.global_step % 100 == 0:
                self.log_similarity(semisl_sim,'semisl_sim')
                self.log_similarity(ssl_sim,'ssl_sim')
                self.log_similarity(sl_sim,'sl_sim')
                
                fig, ax = plt.subplots(1, 1)
                ax.imshow(out_['semisl_contrastive_matrix'].detach(
                ).cpu().numpy(), cmap="plasma")
                self.logger.log_image(
                    'target_contrastive_matrix', [wandb.Image(fig)])
                plt.close(fig)
    
    def log_similarity(self,similarity,name):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(similarity.detach(
        ).cpu().numpy(), cmap="plasma")
        self.logger.log_image(
            name, [wandb.Image(fig)])
        plt.close(fig)
    
        
        
    def get_contrastive_matrices(self,B,N,T,labels):
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        ssl_contrastive_matrix = self.get_ssl_contrastive_matrix(B,N,T,device = labels.device)
        sl_contrastive_matrix = self.get_sl_contrastive_matrix(B,N,T,labels, device = labels.device)
        
        # this is if we want to have a different N_aug for ssl and sl but it makes the code more complicated
        # new_contrastive_matrix = torch.zeros(B*(N_ssl+N_sl),B*(N_ssl+N_sl),device = labels.device)
         # new_contrastive_matrix[:B*N_ssl,:B*N_ssl] = ssl_contrastive_matrix
        # new_contrastive_matrix[B*N_ssl:,B*N_ssl:] = sl_contrastive_matrix
        
        new_contrastive_matrix = ((ssl_contrastive_matrix + sl_contrastive_matrix) > 0).int()
        
        return new_contrastive_matrix,ssl_contrastive_matrix, sl_contrastive_matrix
        
        
    def get_ssl_contrastive_matrix(self,B,N,device):
        
        contrastive_matrix = torch.zeros(B*N,B*N,device = device)
        indices = torch.arange(0, B * N, 1, device=device)

        i_indices, j_indices = torch.meshgrid(indices, indices)
        mask = (i_indices // N) == (j_indices // N)
        contrastive_matrix[i_indices[mask], j_indices[mask]] = 1
        contrastive_matrix[j_indices[mask], i_indices[mask]] = 1

        return contrastive_matrix
    
    def get_sl_contrastive_matrix(self,B,N,labels, device):
        
        ## labels is of shape [B,N_augmentations,n_classes]
        ## labels is a one_hot encoding of the labels
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        indices = torch.arange(0, B * N, 1, device=device)
        
        i_indices, j_indices = torch.meshgrid(indices, indices)
        
        # if the label is -1 then there is no corresponding class in the batch
        
        
        ## the simplest solution with multilabel is to consider that if any of the labels are the same, then the
        ## examples are similar.
        ## the other solution is to consider that if all the labels are the same, then the examples are similar.
        ## the latter is more strict and will probably lead to better results, but the former is more robust to
        ## label noise.
        ## the last solution is to weight the similarity by the number of labels that are the same.
        
        
        x = (labels[i_indices] == labels[j_indices])*(labels[i_indices]==1)
        contrastive_matrix = x.any(dim=-1).int()
        
        return contrastive_matrix
    
    
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer