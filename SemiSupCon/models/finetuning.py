import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable
from SemiSupCon.models.semisupcon import SemiSupCon
from torchmetrics.functional import auroc, average_precision

class FinetuneSemiSupCon(pl.LightningModule):
    
    def __init__(self, encoder, 
        optimizer: OptimizerCallable = None,
        freeze_encoder = True,
        checkpoint = None,
        mlp_head = True,
        checkpoint_head = None,
        task = 'mtat_top50'):
        super().__init__()
        
        self.task = task
        if self.task == 'mtat_top50':
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.n_classes = 50
            
        self.semisupcon = SemiSupCon(encoder)
        self.optimizer = optimizer
        
        self.freeze_encoder = freeze_encoder
        self.checkpoint = checkpoint
        self.checkpoint_head = checkpoint_head
        
        if self.checkpoint:
            self.load_encoder_weights_from_checkpoint(self.checkpoint)
            
        if self.checkpoint_head:
            self.mlp.load_state_dict(torch.load(self.checkpoint_head))
            
        if self.freeze_encoder:
            self.semisupcon.freeze()
            self.semisupcon.eval()
            
            
        self.agg_preds = []
        self.agg_ground_truth = []
        
        
            
        if mlp_head:
            self.head = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Linear(512, self.n_classes, bias=False),
            )
        else:
            self.head = nn.Linear(512, self.n_classes, bias=False)
            
        
        
    def load_encoder_weights_from_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.semisupcon.load_state_dict(checkpoint['state_dict'], strict = False)
        
        
    def forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels'].squeeze(1)
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
        
        
        # x is of shape [B,T]:
        
        encoded = self.semisupcon(wav)['encoded']
        projected = self.head(encoded)
        
        return {
            'projected':projected,
            'labels':labels,
            'encoded':encoded
        }
        
        
    def training_step(self, batch, batch_idx):
            
        x = batch
        out_ = self(x)
        
        logits = out_['projected']
        labels = out_['labels']
        
        loss = self.loss_fn(logits,labels.float())
        
        #get metrics
        preds = torch.sigmoid(logits)
        aurocs = auroc(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        ap_score = average_precision(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        
        self.log('train_loss',loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log('train_auroc',aurocs, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log('train_ap',ap_score, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        x = batch
        out_ = self(x)
        
        logits = out_['projected']
        labels = out_['labels']
        
        loss = self.loss_fn(logits,labels.float())
    
        #get metrics
        preds = torch.sigmoid(logits)
        aurocs = auroc(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        ap_score = average_precision(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        
        self.log('val_loss',loss, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log('val_auroc',aurocs, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        self.log('val_ap',ap_score, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)
        
        return loss
    
    def test_step(self,batch,batch_idx):
        x = batch
        
        x['audio'] = x['audio'].squeeze(0).unsqueeze(1).unsqueeze(1)
        x['labels'] = x['labels'].squeeze(0)
        
        out_ = self(x)
        
        
        logits = out_['projected']
        labels = out_['labels']
        
        logits = logits.mean(0).unsqueeze(0)
        labels = labels[0].unsqueeze(0)
        
        self.agg_ground_truth.append(labels)
        self.agg_preds.append(logits)
        
        loss = self.loss_fn(logits,labels.float())
        return loss
    
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.agg_preds,0)
        ground_truth = torch.cat(self.agg_ground_truth,0)
        
        preds = torch.sigmoid(preds)
        
        loss = self.loss_fn(preds,ground_truth.float())
        aurocs = auroc(preds,ground_truth,task = 'multilabel',num_labels = self.n_classes)
        ap_score = average_precision(preds,ground_truth,task = 'multilabel',num_labels = self.n_classes)
        
        self.log('test_loss',loss, on_step = False, on_epoch = True, prog_bar = False, sync_dist = True)
        self.log('test_auroc',aurocs, on_step = False, on_epoch = True, prog_bar = False, sync_dist = True)
        self.log('test_ap',ap_score, on_step = False, on_epoch = True, prog_bar = False, sync_dist = True)
        
        self.agg_preds = []
        self.agg_ground_truth = []
    
    
        
        
    
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer
    
    def on_checkpoint_save(self, checkpoint):
        checkpoint['state_dict'] = self.head.state_dict()