## a mixed dataset will take a supervised dataset and an unsupervised dataset
## as well as a percentage of the supervised dataset to use for training
## and a percentage of the batch to be supervised data
## it should fetch items from both datasets and then combine them
## it should be implemented as a DataLoder to be able to control the in_batch proportions

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
import numpy as np
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter


class CustomDataLoderIter(_BaseDataLoaderIter):
    def __init__(self, supervised_dataloader, unsupervised_dataloader, *args, **kwargs):
        super().__init__(loader = unsupervised_dataloader)
        self.supervised_loader = supervised_dataloader
        self.unsupervised_loader = unsupervised_dataloader
        self.supervised_iter = iter(supervised_dataloader)
        self.unsupervised_iter = iter(unsupervised_dataloader)
        
        # infinite cycle the smaller dataset
         
        
    def __next__(self):
        try:
            supervised_batch = next(self.supervised_iter)
        except StopIteration:
            self.supervised_iter._reset(self.supervised_loader)
            supervised_batch = next(self.supervised_iter)

        try:
            unsupervised_batch = next(self.unsupervised_iter)
        except StopIteration:
            self.unsupervised_iter._reset(self.unsupervised_loader)
            unsupervised_batch = next(self.unsupervised_iter)      
        ## collate dictionaries into a single dictionary
        
        batch = {}
        for key in supervised_batch:
            batch[key] = torch.cat([supervised_batch[key], unsupervised_batch[key]])
        
        return batch

class MixedDataLoader(DataLoader):
    
    def __init__(self, supervised_dataset, unsupervised_dataset, supervised_dataset_percentage = 1, in_batch_supervised_percentage = 0.5, *args, **kwargs):
        super().__init__(dataset = None, *args, **kwargs)
        
        self.supervised_dataset = supervised_dataset
        self.unsupervised_dataset = unsupervised_dataset
        self.supervised_dataset_percentage = supervised_dataset_percentage
        self.in_batch_supervised_percentage = in_batch_supervised_percentage
        self.supervised_batch_size = 0
        
        # if self.supervised_dataset is None:
        #     self.supervised_dataset_size = 0
        # else:
        #     self.supervised_dataset_size = len(self.supervised_dataset)
        # # shuffle and randomly keep supervised_dataset_percentage of the supervised dataset
        
        
        # self.supervised_dataset_indices = np.random.permutation(self.supervised_dataset_size)[:int(self.supervised_dataset_size * self.supervised_dataset_percentage)]
        # self.supervised_dataset = torch.utils.data.Subset(self.supervised_dataset, self.supervised_dataset_indices)
        
        # self.supervised_batch_size = int(self.batch_size * self.in_batch_supervised_percentage)
        # self.unsupervised_batch_size = self.batch_size - self.supervised_batch_size
        
        # self.supervised_dataloader = DataLoader(self.supervised_dataset, batch_size=self.supervised_batch_size, shuffle=True)
        # self.unsupervised_dataloader = DataLoader(self.unsupervised_dataset, batch_size=self.unsupervised_batch_size, shuffle=True)
        
        # replicate above code but deal with the cases where either dataset is none
        
        if self.supervised_dataset is None:
            self.supervised_dataloader = None
        else:
            self.supervised_dataset_size = len(self.supervised_dataset)
            self.supervised_dataset_indices = np.random.permutation(self.supervised_dataset_size)[:int(self.supervised_dataset_size * self.supervised_dataset_percentage)]
            self.supervised_dataset = torch.utils.data.Subset(self.supervised_dataset, self.supervised_dataset_indices)
            self.supervised_batch_size = int(self.batch_size * self.in_batch_supervised_percentage)
            self.supervised_dataloader = DataLoader(self.supervised_dataset, batch_size=self.supervised_batch_size, shuffle=True, num_workers=self.num_workers//2)
            
        if self.unsupervised_dataset is None:
            self.unsupervised_dataloader = None
        else:
            self.unsupervised_batch_size = self.batch_size - self.supervised_batch_size
            self.unsupervised_dataloader = DataLoader(self.unsupervised_dataset, batch_size=self.unsupervised_batch_size, shuffle=True, num_workers=self.num_workers//2)
        
    def __iter__(self) -> _BaseDataLoaderIter:
        if self.unsupervised_dataset is None:
            return iter(self.supervised_dataloader)
        if self.supervised_dataset is None:
            return iter(self.unsupervised_dataloader)
        return CustomDataLoderIter(self.supervised_dataloader, self.unsupervised_dataloader)
    
    def __len__(self):
        if self.supervised_dataset is None:
            return len(self.unsupervised_dataloader)
        if self.unsupervised_dataset is None:
            return len(self.supervised_dataloader)
        else:
            return max(len(self.supervised_dataloader), len(self.unsupervised_dataloader))