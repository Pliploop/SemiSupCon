## boilerplate code for a pytorchlightning datamodule please

import os
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Union
from torch import Tensor
from torch.utils.data import random_split

from SemiSupCon.dataloading import *

class MixedDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, annotations = None, target_length = 2.6, target_sample_rate = 22050, n_augmentations= 2, transform = True, n_classes = 50, batch_size = 32, num_workers = 8, val_split = 0.1, supervised_data_p = 1, intrabatch_supervised_p = 0.5) -> None:
        
        super().__init__()
        
        self.data_dir = data_dir
        self.annotations = annotations
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.n_augmentations = n_augmentations
        self.target_samples = int(self.target_length * self.target_sample_rate)
        self.global_target_samples = self.target_samples * self.n_augmentations
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.supervised_dataset_percentage = supervised_data_p
        self.in_batch_supervised_percentage = intrabatch_supervised_p
        
        self.augentations = None
        
        
        
    def setup(self):
        
        self.supervised_dataset = SupervisedDataset(self.data_dir, self.annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.augmentations,  self.n_classes, train = True)
        self.self_supervised_dataset = SelfSupervisedDataset(self.data_dir, self.annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.augmentations, self.n_classes, train = True )
        
        if self.val_split > 0:
            train_supervised_dataset, val_supervised_dataset = self.split_datasets(self.supervised_dataset)
            train_self_supervised_dataset, val_self_supervised_dataset =  self.split_datasets(self.self_supervised_dataset)
            val_supervised_dataset.transform = False
            val_self_supervised_dataset.transform = False
        else:
            train_supervised_dataset = self.supervised_dataset
            train_self_supervised_dataset = self.self_supervised_dataset
            val_supervised_dataset = None
            val_self_supervised_dataset = None
            

        if self.in_batch_supervised_percentage == 0:
            train_supervised_dataset = None
            val_supervised_dataset = None
        if self.in_batch_supervised_percentage == 1:
            train_self_supervised_dataset = None
            val_self_supervised_dataset = None
            
        self.train_dataloader = MixedDataLoader(train_supervised_dataset, train_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, self.batch_size, self.num_workers)
        self.val_dataloader = MixedDataLoader(val_supervised_dataset, val_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, self.batch_size, self.num_workers)
        
    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        if self.val_split > 0:
            return self.val_dataloader
        else:
            return None    
        
    def split_datasets(self, dataset):
        
        # split datasets intro train and validation
        train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * (1 - self.val_split)), int(len(dataset) * self.val_split)])
        return train_dataset, val_dataset