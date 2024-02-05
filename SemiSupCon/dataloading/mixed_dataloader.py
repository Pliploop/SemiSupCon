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
from tqdm import tqdm


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
    
    def __init__(self, supervised_dataset, unsupervised_dataset, supervised_dataset_percentage = 1, in_batch_supervised_percentage = 0.5, n_classes=10, *args, **kwargs):
        super().__init__(dataset = None, *args, **kwargs)
        
        self.supervised_dataset = supervised_dataset
        self.unsupervised_dataset = unsupervised_dataset
        self.supervised_dataset_percentage = supervised_dataset_percentage
        self.in_batch_supervised_percentage = in_batch_supervised_percentage
        self.supervised_batch_size = 0
        self.unsupervised_batch_size = 0
        self.n_classes= n_classes
        
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
            self.supervised_dataset_size = 0
        else:
            self.supervised_dataset_size = len(self.supervised_dataset)
            self.supervised_dataset_indices = np.random.permutation(self.supervised_dataset_size)[:int(self.supervised_dataset_size * self.supervised_dataset_percentage)]
            self.supervised_dataset = torch.utils.data.Subset(self.supervised_dataset, self.supervised_dataset_indices)
            self.supervised_batch_size = int(self.batch_size * self.in_batch_supervised_percentage)
            
        if self.unsupervised_dataset is None:
            self.unsupervised_dataloader = None
            self.unsupervised_dataset_size = 0
        else:
            self.unsupervised_batch_size = self.batch_size - self.supervised_batch_size
            self.unsupervised_dataset_size = len(self.unsupervised_dataset)

        
        
        # create some samplers for the supervised and unsupervised dataloaders so that they have the same length
        # number of batches should be the same for both dataloaders
        
        # if self.supervised_dataset is None:
        #     self.supervised_dataset_size = 0
        # else:
        #     self.supervised_dataset_size = len(self.supervised_dataset)
        # if self.unsupervised_dataset is None:
        #     self.unsupervised_dataset_size = 0
        # else:
        #     self.unsupervised_dataset_size = len(self.unsupervised_dataset)
            
        # max_dataset_size = max(self.supervised_dataset_size, self.unsupervised_dataset_size)
        if self.supervised_dataset is None:
            target_unsupervised_dataset_size = self.unsupervised_dataset_size
            target_supervised_dataset_size = 0
            
            
        elif self.unsupervised_dataset is None:
            target_supervised_dataset_size = self.supervised_dataset_size
            target_unsupervised_dataset_size = 0
        else:
            max_dataloader_size = max(self.supervised_dataset_size//self.supervised_batch_size + 1, self.unsupervised_dataset_size//self.unsupervised_batch_size +1)
            target_supervised_dataset_size = max_dataloader_size * self.supervised_batch_size
            target_unsupervised_dataset_size = max_dataloader_size * self.unsupervised_batch_size
        
        if self.supervised_dataset is not None:
            supervised_sampler = torch.utils.data.RandomSampler(self.supervised_dataset, replacement=True, num_samples=target_supervised_dataset_size)
            self.supervised_dataloader = DataLoader(self.supervised_dataset, batch_size=self.supervised_batch_size, sampler=supervised_sampler, num_workers=self.num_workers//2, drop_last=True)
            max_dataloader_size = len(self.supervised_dataloader)
        if self.unsupervised_dataset is not None:
            unsupervised_sampler = torch.utils.data.RandomSampler(self.unsupervised_dataset, replacement=True, num_samples=target_unsupervised_dataset_size)
            self.unsupervised_dataloader = DataLoader(self.unsupervised_dataset, batch_size=self.unsupervised_batch_size, sampler=unsupervised_sampler, num_workers=self.num_workers//2, drop_last=True)
            max_dataloader_size = len(self.unsupervised_dataloader)
            
        # do some logging for sanity checking
        print(f"supervised dataset size: {self.supervised_dataset_size}")
        print(f"unsupervised dataset size: {self.unsupervised_dataset_size}")
        print(f"supervised batch size: {self.supervised_batch_size}")
        print(f"unsupervised batch size: {self.unsupervised_batch_size}")
        print(f"max dataloader size: {max_dataloader_size}")
        print(f"target supervised dataset size: {target_supervised_dataset_size}")
        print(f"target unsupervised dataset size: {target_unsupervised_dataset_size}")
        # print(f"supervised dataloader size: {len(self.supervised_dataloader)}")
        # print(f"unsupervised dataloader size: {len(self.unsupervised_dataloader)}")
        
        
        
    def __iter__(self) -> _BaseDataLoaderIter:
        if self.unsupervised_dataset is None or len(self.unsupervised_dataset) == 0:
            return iter(self.supervised_dataloader)
        if self.supervised_dataset is None or len(self.supervised_dataset) == 0:
            return iter(self.unsupervised_dataloader)
        return CustomDataLoderIter(self.supervised_dataloader, self.unsupervised_dataloader)
    
    def __len__(self):
        if self.supervised_dataset is None:
            return len(self.unsupervised_dataloader)
        if self.unsupervised_dataset is None:
            return len(self.supervised_dataloader)
        else:
            return min(len(self.supervised_dataloader), len(self.unsupervised_dataloader))
        
        
    def get_prototypes(self, model, n = None):
        # get a prototype for each class in the form of the centroid of the embeddings of the samples of that class

        # send model to gpu if available
        if torch.cuda.is_available():
            model.to('cuda')

        # get the d_model by checking the shape of the last layer of {model_layer}
        d_model = model.proj_head[-1].weight.shape[0]
        model.eval()
        model.freeze()

        prototypes = torch.zeros(self.n_classes, d_model).to(model.device)  # Initialize the prototypes
        class_counts = torch.zeros(self.n_classes).to(model.device)  # Initialize the counts of each class

        if n is None:
            n = len(self)
        else:
            # n is the number of samples wanted to take batch size into account
            n = int(np.ceil(n / self.batch_size))

        for idx in tqdm(range(n)):  # Iterate over batches from the DataLoader
            data = next(iter(self))
            audio = data['audio'].to(model.device)
            labels = data['labels'].to(model.device)

            with torch.no_grad():
                out = model(audio)

            encoded = out['projected']

            for i in range(self.n_classes):
                # if class i is in the labels then add encoded to prototypes[i]
                # print(labels.shape)
                # mask = (labels[:, i] == 1).any(dim = 1)  # Find where class i is in the labels
                mask = (labels[:, 0, i] == 1)
                prototypes[i] += encoded[mask].sum(dim=0)  # Sum all the encoded vectors where class i is present
                class_counts[i] += mask.float().sum()  # Count how many times class i is present

        return prototypes / class_counts.unsqueeze(1)