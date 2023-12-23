import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable
from SemiSupCon.models.semisupcon import SemiSupCon

class FinetuneSemiSupCon(pl.LightningModule):
    
    def __init__(self, encoder, 
        optimizer: OptimizerCallable = None, freeze_encoder = True, checkpoint_path = None):
        super().__init__()
        
        self.semisupcon = SemiSupCon(encoder)
        
        self.mlp = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
        )
        
        self.optimizer = optimizer
        
        self.freeze_encoder = freeze_encoder
        self.checkpoint_path = checkpoint_path
        
        if self.checkpoint_path:
            self.load_encoder_weights_from_checkpoint(self.checkpoint_path)
            
        if self.freeze_encoder:
            self.freeze_encoder()
        
        
    def load_encoder_weights_from_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.semisupcon.load_state_dict(checkpoint['state_dict'], strict = False)
        
        
    def freeze_encoder(self):
        self.semisupcon.freeze()
        
        # assert that the encoder is frozen
        for param in self.semisupcon.parameters():
            assert param.requires_grad == False
        print('encoder frozen')
        
    def forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels']
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
        
        
        # x is of shape [B,T]:
        
        encoded = self.semisupcon.encoder(wav)
        projected = self.mlp(encoded)
        
        return {
            'projected':projected,
            'labels':labels,
            'encoded':encoded
        }
        
        
    def training_step(self, batch, batch_idx):
            
        x = batch
        out_ = self(x)
        
        
        self.logging(out_)
        
        return 0
    
    
    def validation_step(self,batch,batch_idx):
        x = batch
        out_ = self(x)
        
        
        self.logging(out_)
        
        return 0
    
    def logging(self,out_): 
        
        # metrics and losses are logged here
        pass
    
    
        
        
    
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer