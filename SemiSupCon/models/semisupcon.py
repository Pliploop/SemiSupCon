import pytorch_lightning as pl
from SemiSupCon.models.losses.semisupconloss import SSNTXent
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable

class SemiSupCon(pl.LightningModule):
    
    def __init__(self, encoder, 
        optimizer: OptimizerCallable = None, temperature = 0.1):
        super().__init__()
        self.loss = SSNTXent(temperature = temperature)
        self.encoder = encoder
        self.proj_head = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
        )
        
        self.optimizer = optimizer
        
    def forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels']
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
        
        
        # x is of shape [B,N_augmentations,T]:
        B, N_augmentations,_, T = wav.shape

        wav = wav.contiguous().view(-1,1,wav.shape[-1]) ## [B*N_augmentations,T]
        labels = labels.contiguous().view(-1,labels.shape[-1]) ## [B*N_augmentations,n_classes]
        
        semisl_contrastive_matrix,ssl_contrastive_matrix, sl_contrastive_matrix = self.get_contrastive_matrices(B,N_augmentations,T,labels)
        
        encoded = self.encoder(wav)
        projected = self.proj_head(encoded)
        
        return {
            'projected':projected,
            'semisl_contrastive_matrix':semisl_contrastive_matrix, ## this is the matrix that we will use for the ssl loss
            'ssl_contrastive_matrix':ssl_contrastive_matrix,
            'sl_contrastive_matrix':sl_contrastive_matrix,
            'labels':labels,
            'encoded':encoded
        }
        
    def finetune_forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels']
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
        
        encoded = self.encoder(wav)
        
        return {
            'encoded':encoded,
            'labels':labels
        }
        
    def training_step(self, batch, batch_idx):
            
        x = batch
        out_ = self(x)
        
        labeled = batch['labeled']
        labeled = labeled.contiguous().view(-1)
        # invert the semi-supervised contrastive matrix
        
        positive_mask = out_['semisl_contrastive_matrix']
        negative_mask = torch.ones_like(positive_mask)
        
        semisl_loss = self.loss(out_['projected'],positive_mask,negative_mask)
        
        self.logging(out_,labeled)
        
        return semisl_loss
    
    
    def validation_step(self,batch,batch_idx):
        x = batch
        out_ = self(x)
        
        positive_mask = out_['semisl_contrastive_matrix']
        negative_mask = torch.ones_like(positive_mask)
        
        semisl_loss = self.loss(out_['projected'],positive_mask,negative_mask)
        self.log('val_semisl_loss',semisl_loss,on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        
        return semisl_loss
    
    def logging(self,out_, labeled): 
        
        labeled = labeled.bool()
        
        semisl_loss = self.loss(out_['projected'],out_['semisl_contrastive_matrix'], torch.ones_like(out_['semisl_contrastive_matrix']))
        sl_loss = self.loss(out_['projected'][labeled,:],out_['sl_contrastive_matrix'][labeled, labeled],torch.ones_like(out_['sl_contrastive_matrix'][labeled, labeled]))
        ssl_loss = self.loss(out_['projected'][~labeled,:],out_['ssl_contrastive_matrix'][~labeled, ~labeled], torch.ones_like(out_['ssl_contrastive_matrix'][~labeled, ~labeled]))
        
        # get similarities
        semisl_sim = self.loss.get_similarities(out_['projected'])
        ssl_sim = self.loss.get_similarities(out_['projected'][~labeled,:])
        sl_sim = self.loss.get_similarities(out_['projected'][labeled,:])
        
        self.log('semisl_loss',semisl_loss,on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log('sl_loss',sl_loss, on_step = True, on_epoch = True, sync_dist = True)
        self.log('ssl_loss',ssl_loss, on_step = True, on_epoch = True, sync_dist = True)
        
        if self.logger:
            
            if self.global_step % 200 == 0:
                self.log_similarity(semisl_sim,'semisl_sim')
                self.log_similarity(ssl_sim,'ssl_sim')
                self.log_similarity(sl_sim,'sl_sim')
                
                fig, ax = plt.subplots(1, 1)
                
                semisl_contrastive_matrix = out_['semisl_contrastive_matrix']
                semisl_contrastive_matrix[torch.eye(semisl_contrastive_matrix.shape[0],device = semisl_contrastive_matrix.device).bool()] = 0
                
                ax.imshow(semisl_contrastive_matrix.detach(
                ).cpu().numpy(), cmap="plasma")
                self.logger.log_image(
                    'target_contrastive_matrix', [wandb.Image(fig)])
                plt.close(fig)
    
    def log_similarity(self,similarity,name):
        fig, ax = plt.subplots(1, 1)
        # remove diagonal
        similarity[torch.eye(similarity.shape[0],device = similarity.device).bool()] = 0
        ax.imshow(similarity.detach(
        ).cpu().numpy(), cmap="plasma")
        self.logger.log_image(
            name, [wandb.Image(fig)])
        plt.close(fig)
    
        
        
    def get_contrastive_matrices(self,B,N,T,labels):
        
        ## returns a matrix of shape [B*N_augmentations,B*N_augmentations] with 1s where the labels are the same
        ## and 0s where the labels are different
        
        ssl_contrastive_matrix = self.get_ssl_contrastive_matrix(B,N,device = labels.device)
        sl_contrastive_matrix = self.get_sl_contrastive_matrix(B,N,labels, device = labels.device)
        
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
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer