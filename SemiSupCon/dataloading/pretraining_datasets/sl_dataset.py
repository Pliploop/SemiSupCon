from torch.utils.data import Dataset
import torch
from SemiSupCon.dataloading.utils.loading_utils import load_audio_and_split_in_chunks, load_random_audio_chunk, load_full_audio
import os
from tqdm import tqdm
from torch.utils.data import Subset
import pandas as pd




class SupervisedDataset(Dataset):
    
    def __init__(self, data_dir, annotations = None, target_length = 2.7, target_sample_rate = 22050, n_augmentations= 2, transform = True, augmentations = None, train = True, n_classes = 50, allow_overlapping = False, idx2class = None) -> None:
        
        self.data_dir = data_dir
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.n_augmentations = n_augmentations
        self.target_samples = int(self.target_length * self.target_sample_rate)
        self.global_target_samples = self.target_samples * self.n_augmentations
        self.train = train
        self.n_classes = n_classes
        self.transform = transform
        self.augmentations = augmentations
        self.allow_overlapping = allow_overlapping
        self.idx2class = idx2class # dictionary mapping class indices to label names
        
        self.annotations = annotations
        
    def get_labels(self, index):
        return torch.tensor(self.annotations.iloc[index]['labels'])
    
    def __len__(self):
        return len(self.annotations)
                    
    def __getitem__(self, index):
        
        path = os.path.join(self.data_dir, self.annotations.iloc[index]['file_path'])
        audio = load_random_audio_chunk(path, self.global_target_samples, self.target_sample_rate, allow_overlapping = self.allow_overlapping, n_augmentations=self.n_augmentations)
        if audio is None:
            return self[index + 1]
        
        audio = self.split_and_augment(audio)
        
        labeled = torch.tensor(1)
        labels = self.get_labels(index)
        
        #add n_augmentation dimension as the first dimension
        labels = labels.unsqueeze(0).repeat(self.n_augmentations,1)
        labeled = labeled.unsqueeze(0).repeat(self.n_augmentations)
        
        
        return {
            "audio": audio,
            "labels": labels,
            "labeled": labeled
        }
        
    
    def split_and_augment(self,audio):
        
        waveform = torch.cat(torch.split(
            audio, self.target_samples, dim=1)).unsqueeze(1)
        
        if self.augmentations is not None and self.transform:
            waveform = self.augmentations(waveform)
            
        return waveform
    
    def get_representations_for_reduction(self, model, n = None, bottleneck = 'encoded'):
        
        # get the d_model by checking the shape of the last layer of {model_layer}
        
        if bottleneck == 'encoded':
            d_model = model.proj_head[-1].weight.shape[1]
        else:
            d_model = model.proj_head[-1].weight.shape[0]
            
        #pretty print the d_model and the bottleneck
        print(f'd_model: {d_model}, bottleneck: {bottleneck}')
        model.eval()
        model.freeze()
        
        # get model device 
        device = next(model.parameters()).device
        
        
        if n is None:
            n = len(self)
        
        representations = torch.zeros(n, d_model)
        labels = torch.zeros(n, self.n_classes)
        rows = []
            
        for idx in tqdm(range(n)):
                
                data = self[idx]
                audio = data['audio'].to(device)
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
                if audio.dim() == 3:
                    audio = audio.unsqueeze(1)
                label = data['labels']
                
                if audio.shape[0] > 64:
                    audio = audio[:64]
                
                with torch.no_grad():
                    out = model(audio)
                
                encoded = out[bottleneck].mean(0).unsqueeze(0)
                representations[idx,:] = encoded
                labels[idx,:] = label
                # labels are one-hot encoded, get the index of the class
                labels_idx = torch.argmax(labels, dim = 1).tolist()
                if self.idx2class is not None:
                    labels_names  = [self.idx2class[i] for i in labels_idx]
                else:
                    labels_names = None
                row = self.annotations.iloc[idx]
                rows.append(row)
                
        dataframe_out = pd.DataFrame(rows)
        return {
            'representations': representations,
            'labels': labels,
            'labels_idx': labels_idx,
            'labels_names': labels_names,
            'dataframe': self.annotations # for potential dual class indexing
        }
    
    def get_prototypes(self,model,n = None):
        # get a prototype for each class in the form of the centroid of the embeddings of the samples of that class
        
        
        # get the d_model by checking the shape of the last layer of {model_layer}
        
        d_model = model.proj_head[-1].weight.shape[0]
        model.eval()
        model.freeze()
        
        prototypes = torch.zeros(self.n_classes, d_model)
        
        if n is None:
            n = len(self)
        
        
        class_counts = torch.zeros(self.n_classes)
        
        for idx in tqdm(range(n)):
            
            data = self[idx]
            audio = data['audio'].unsqueeze(0)
            labels = data['labels']
            
            
            
            with torch.no_grad():
                out = model(audio)
            
            encoded = out['projected']
            
            for i in range(self.n_classes):
                # consider batch size of 1 for now
                # if class i is in the labels then add encoded to prototypes[i]
                if labels[0][i] == 1:
                    prototypes[i] += encoded[0]
                class_counts[i] += labels[0][i]
                    
        return prototypes / class_counts.unsqueeze(1)
        
  
class SupervisedTestDataset(SupervisedDataset):
    
    def __init__(self, data_dir, annotations = None, target_length = 2.7, target_sample_rate = 22050, n_augmentations= 1, transform = True, augmentations = None, train = True, n_classes = 50, idx2class = None) -> None:
        
        super().__init__(data_dir, annotations, target_length, target_sample_rate, n_augmentations, transform, augmentations, train, n_classes, idx2class = idx2class)
        
        
    def __getitem__(self, index):
        
        path = os.path.join(self.data_dir, self.annotations.iloc[index]['file_path'])
        
        
        audio = load_audio_and_split_in_chunks(path, self.target_samples, self.target_sample_rate)
        if audio is None:
            return self[index + 1]
        labeled = torch.tensor(1)
        labels = self.get_labels(index)
        
        if self.transform and self.augmentations is not None:
            audio = audio.unsqueeze(1)
            audio = self.augmentations(audio)
            audio = audio.squeeze(1)
        
        #add n_augmentation dimension as the first dimension
        labels = labels.unsqueeze(0).repeat(self.n_augmentations,1)
        labeled = labeled.unsqueeze(0).repeat(self.n_augmentations)
        
        
        
        return {
            "audio": audio,
            "labels": labels,
            "labeled": labeled
        }

class DummySupervisedDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        audio = torch.rand(1, 16000)
        # labels should be a one-hot vector with num_classes = 10
        labels = torch.zeros(10)
        # random selection of one of the 10 classes
        label = torch.rand(1, 10)
        labels[label.int()] = 1
        
        labeled = torch.tensor(1)
        
        return {
            "audio": audio,
            "labels": labels,
            "labeled": labeled
                }