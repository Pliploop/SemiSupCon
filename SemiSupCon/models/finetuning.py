import pytorch_lightning as pl
import torch
from torch import nn
import matplotlib.pyplot as plt
import wandb
from torch import optim
from pytorch_lightning.cli import OptimizerCallable
from SemiSupCon.models.semisupcon import SemiSupCon
from torchmetrics.functional import auroc, average_precision
from SemiSupCon.models.utils import confusion_matrix
import numpy as np
from SemiSupCon.models.task_metrics import *

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
        
        task_dict = {
            'mtat_top50': {'n_classes': 50, 'loss_fn': nn.BCEWithLogitsLoss()},
            'mtat_all': {'n_classes': 188, 'loss_fn': nn.BCEWithLogitsLoss()},
            'giantsteps': {'n_classes': 24, 'loss_fn': nn.CrossEntropyLoss()},
            'nsynth_pitch': {'n_classes': 112, 'loss_fn': nn.CrossEntropyLoss()},
            'nsynth_instr_family': {'n_classes': 11, 'loss_fn': nn.CrossEntropyLoss()},
            'gtzan': {'n_classes': 10, 'loss_fn': nn.CrossEntropyLoss()},
            'mtg_top50': {'n_classes': 50, 'loss_fn': nn.BCEWithLogitsLoss()},
            'mtg_genre': {'n_classes': 87, 'loss_fn': nn.BCEWithLogitsLoss()},
            'mtg_mood': {'n_classes': 56, 'loss_fn': nn.BCEWithLogitsLoss()},
            'mtg_instr': {'n_classes': 40, 'loss_fn': nn.BCEWithLogitsLoss()},
            'emomusic': {'n_classes': 2, 'loss_fn': nn.MSELoss()},
            'vocalset_technique': {'n_classes': 17, 'loss_fn': nn.CrossEntropyLoss()},
            'vocalset_singer': {'n_classes': 20, 'loss_fn': nn.CrossEntropyLoss()},
            'medleydb': {'n_classes': 20, 'loss_fn': nn.CrossEntropyLoss()}
        }

        if self.task in task_dict:
            self.n_classes = task_dict[self.task]['n_classes']
            self.loss_fn = task_dict[self.task]['loss_fn']
        else:
            raise ValueError(f"Invalid task: {self.task}")
            
        self.semisupcon = SemiSupCon(encoder)
        self.optimizer = optimizer
        
        self.freeze_encoder = freeze_encoder
        self.checkpoint = checkpoint
        self.checkpoint_head = checkpoint_head
        
        if self.checkpoint:
            self.load_encoder_weights_from_checkpoint(self.checkpoint)
            
            
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
            
        device = next(self.semisupcon.parameters()).device
        
        if self.checkpoint_head:
            print(f'Loading head weights from checkpoint {self.checkpoint_head}')
            print(torch.load(self.checkpoint_head, map_location=device)['state_dict'].keys())
            self.load_state_dict(torch.load(self.checkpoint_head, map_location=device)['state_dict'], strict = False)
            print(f'Loaded head weights from checkpoint {self.checkpoint_head}')
            
        
        if self.freeze_encoder:
            self.semisupcon.freeze()
            self.semisupcon.eval()
            print('Encoder is frozen')
            
        
            
        self.idx2class = None
        self.class_names = None
        
        self.get_metrics = eval(f'{self.task}_metrics')
        
    def set_idx2class(self,idx2class):
        self.idx2class = idx2class    
    
    def set_class_names(self,class_names):
        self.class_names = class_names
    
    def log_metrics(self,metrics, stage = 'train'):
        # metrics is a dictionary containing the metric name and the value
        for k,v in metrics.items():
            if stage == 'train' or stage == 'val':
                self.log(f'{stage}_{k}',v, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
            else:
                self.log(f'{stage}_{k}',v, on_step = False, on_epoch = True, prog_bar = True, sync_dist = True)

    def load_encoder_weights_from_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.semisupcon.load_state_dict(checkpoint['state_dict'], strict = False)
        
    def load_head_weights_from_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'], strict = False)
    
    def load_checkpoint(self,checkpoint_path):
        # get the device of the model
        device = next(self.parameters()).device
        checkpoint = torch.load(checkpoint_path,map_location=device)
        self.load_state_dict(checkpoint['state_dict'], strict = True)
        
    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels'].squeeze(1)
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
        
        
        # x is of shape [B,T]:
        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.semisupcon(wav)['encoded']
        else:
            encoded = self.semisupcon(wav)['encoded']
            
        projected = self.head(encoded)
        
        return {
            'projected':projected,
            'labels':labels,
            'encoded':encoded
        }
        
    def test_forward(self,x):
        
        if isinstance(x,dict):
            wav = x['audio']
            labels = x['labels'].squeeze(1)
        else:
            wav = x
            labels = torch.zeros(wav.shape[0]*wav.shape[1],10)
            
        # x is of shape [B,T]:
        
        encoded = self.semisupcon(wav)['encoded']
        projected = self.head(encoded.mean(0)).unsqueeze(0)
        
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
        # preds = torch.sigmoid(logits)
        # aurocs = auroc(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        # ap_score = average_precision(preds,labels,task = 'multilabel',num_labels = self.n_classes)
        
        train_metrics = self.get_metrics(logits,labels,self.idx2class,self.class_names,self.n_classes)
        
        self.log('train_loss',loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        
        self.log_metrics(train_metrics,stage = 'train')
        
        return loss
    
    
    def validation_step(self,batch,batch_idx):
        x = batch
        out_ = self(x)
        
        logits = out_['projected']
        labels = out_['labels']
        
        loss = self.loss_fn(logits,labels.float())
    
        
        #get metrics
        self.log('val_loss',loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        
        
        val_metrics = self.get_metrics(logits,labels,self.idx2class,self.class_names,self.n_classes)
        
        
        self.log_metrics(val_metrics,stage = 'val')
        
        return loss
    
    def test_step(self,batch,batch_idx):
        x = batch
        
        x['audio'] = x['audio'].squeeze(0).unsqueeze(1).unsqueeze(1)
        if x['audio'].shape[0] > 64:
            x['audio'] = x['audio'][:64]
        x['labels'] = x['labels'].squeeze(0)
        
        out_  = self.test_forward(x)
        
        
        logits = out_['projected']
        labels = out_['labels']
        
        logits = logits.mean(0).unsqueeze(0)
        labels = labels[0].unsqueeze(0)
        
        
        self.agg_ground_truth.append(labels.detach().cpu())
        self.agg_preds.append(logits.detach().cpu())
        
        loss = self.loss_fn(logits,labels.float())
        return loss
    
    
    def on_test_epoch_end(self):
        preds = torch.cat(self.agg_preds,0)
        ground_truth = torch.cat(self.agg_ground_truth,0)
        
        
        loss = self.loss_fn(preds,ground_truth.float())
        
        test_metrics = self.get_metrics(preds,ground_truth,self.idx2class,self.class_names,self.n_classes)
        
        
        self.log('test_loss',loss, on_step = False, on_epoch = True, prog_bar = False, sync_dist = True)
    
        self.log_metrics(test_metrics,stage = 'test')
        
        # make a pretty matplotlib heatmap of the confusion matrix
        if self.task in ['mtat_top50','giantsteps','medleydb']:
            cmat = confusion_matrix(preds,ground_truth,self.n_classes).cpu().numpy()
        # normalize the cmatrix row-wise
            cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        
            fig, ax = plt.subplots(figsize=(30,30))
            im = ax.imshow(cmat)
            
            # We want to show all ticks...
            ax.set_xticks(np.arange(len(self.class_names)))
            ax.set_yticks(np.arange(len(self.class_names)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(self.class_names)
            ax.set_yticklabels(self.class_names)
            
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                        rotation_mode="anchor")
            
            # Loop over data dimensions and create text annotations.
            for i in range(len(self.class_names)):
                
                for j in range(len(self.class_names)):
                    # round to 2 decimal places
                    leg = round(cmat[i,j],2)  
                    text = ax.text(j, i, leg,
                                ha="center", va="center", color="w")
                    
            ax.set_title("Confusion matrix")
            fig.tight_layout()
            if self.logger:
                self.logger.log_image(
                    'confusion_matrix', [wandb.Image(fig)])
            
            
                wandb.log({"Confusion Matrix":self.custom_wandb_confusion_matrix(cmat)})
        
        self.agg_preds = []
        self.agg_ground_truth = []
    
    
    def custom_wandb_confusion_matrix(self,confusion_matrix):
        data = []
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                data.append([self.class_names[i],self.class_names[j],confusion_matrix[i,j]])
            
        fields = {
            'target': 'target',
            'prediction': 'prediction',
            'value': 'value'
        }
        
        return wandb.plot_table(
            "Confusion matrix",
            wandb.Table(data=data, columns=["target", "prediction", "value"]),
            fields,
            {'title': 'Confusion matrix'}
        )
                
        
    
    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = self.optimizer(self.parameters())
            
        return optimizer