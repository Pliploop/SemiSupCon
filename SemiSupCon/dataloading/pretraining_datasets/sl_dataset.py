from torch.utils.data import Dataset
import torch
from SemiSupCon.dataloading.utils.loading_utils import load_audio_and_split_in_chunks, load_random_audio_chunk
import os




class SupervisedDataset(Dataset):
    
    def __init__(self, data_dir, annotations = None, target_length = 2.7, target_sample_rate = 22050, n_augmentations= 2, transform = True, augmentations = None, train = True, n_classes = 50) -> None:
        
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
        
        self.annotations = annotations
        
    def get_labels(self, index):
        return torch.tensor(self.annotations.iloc[index]['labels'])
    
    def __len__(self):
        return len(self.annotations)
                    
    def __getitem__(self, index):
        
        path = os.path.join(self.data_dir, self.annotations.iloc[index]['file_path'])
        audio = load_random_audio_chunk(path, self.global_target_samples, self.target_sample_rate)
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
        
        if self.augmentations is not None and self.transform and self.train:
            waveform = self.augmentations(waveform)
            
        return waveform
            
        

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