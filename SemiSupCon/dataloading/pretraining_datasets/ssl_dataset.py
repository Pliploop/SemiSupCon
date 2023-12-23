from torch.utils.data import Dataset
import torch
import os
from SemiSupCon.dataloading.utils.loading_utils import load_full_audio, load_random_audio_chunk
import numpy as np



class SelfSupervisedDataset(Dataset):
    
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
        self.allow_overlapping = False
        
        self.annotations = annotations
        
    def __len__(self):
        return len(self.annotations)
                    
    def __getitem__(self, index):
        
        path = os.path.join(self.data_dir, self.annotations.iloc[index]['file_path'])
        if self.allow_overlapping == False:
            audio = load_random_audio_chunk(path, self.global_target_samples, self.target_sample_rate)
        else:
            audio = load_full_audio(path, self.target_samples, self.target_sample_rate)
        
        
        if audio is None:
            return self[index + 1]
        
        audio = self.split_and_augment(audio)
        
        labeled = torch.tensor(0)
        labels = torch.zeros(self.n_classes)
        
        
        #add n_augmentation dimension as the first dimension
        labels = labels.unsqueeze(0).repeat(self.n_augmentations,1)
        labeled = labeled.unsqueeze(0).repeat(self.n_augmentations)
        
        return {
            "audio": audio,
            "labels": labels,
            "labeled": labeled
        }
    
    def split_and_augment(self,audio):
        
        if self.allow_overlapping == False:
        
            waveform = torch.cat(torch.split(
                audio, self.target_samples, dim=1)).unsqueeze(1)
        
        else:
            # sample N_augmentations random target_samples samples from the audio
            for i in range(self.n_augmentations):
                start_idx = np.random.randint(low=0, high=audio.shape[1] - self.target_samples)
                waveform = audio[:,start_idx:start_idx + self.target_samples].unsqueeze(0)
                if i == 0:
                    waveform = waveform
                else:
                    waveform = torch.cat([waveform, waveform])
            
            
        
        if self.augmentations is not None and self.transform and self.train:
            waveform = self.augmentations(waveform)
            
        return waveform
            
        


class DummyUnsupervisedDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        audio = torch.rand(1, 16000)
        labeled = torch.tensor(0)
        labels = torch.zeros(10)
        
        return {
            "audio": audio,
            "labels": labels,
            "labeled": labeled
                }