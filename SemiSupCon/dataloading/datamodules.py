## boilerplate code for a pytorchlightning datamodule please

import os
import pytorch_lightning as pl
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from torch_audiomentations import *

from SemiSupCon.dataloading import *
from SemiSupCon.dataloading.custom_augmentations import *


class DataModuleSplitter:
    """generates annotation files for the MixedDataModule. Takes a task as argument
    and returns a pandas dataframe with three columns : file_path, split, labels.
    labels are one-hot encoded vectors of size n_classes or None if the dataset is unsupervised
    
    if supervised_data_p is 1, then all available supervised data is used. If it is 0, then no supervised data is used.
    if fully_supervised is True, then only supervised data is used (e.g supervised contrastive learning or finetuning).
    if use_test_set is false, then the test set is used as part of the training set.
    """
    
    def __init__(self,
                 data_dir,
                 task=None,
                 supervised_data_p = 1,
                 val_split=0.1,
                 test_split = 0.1,
                 use_test_set = False,
                 fully_supervised = False,
                 extension = 'wav'
                 ):
        
        self.task = task
        self.data_dir = data_dir
        self.supervised_data_p = supervised_data_p
        self.fully_supervised = fully_supervised
        self.val_split = val_split
        self.test_split = test_split
        self.use_test_set = use_test_set
        if self.use_test_set==False:
            self.test_split = 0
            
        
        if self.task is None:
            self.fetch_function = self.get_default_annotations
        else:
            self.fetch_function = eval(f"self.get_{self.task}_annotations")
            
        
        self.n_classes = 0
        self.annotations = self.get_annotations()
        
        if self.use_test_set==False:
            # change the split column to train where it is 'test'
            self.annotations.loc[self.annotations['split'] == 'test', 'split'] = 'train'
            
        
        
        # build new column for annotations dataframe that indicates whether the data is supervised or not
        self.annotations['supervised'] = 1
        self.annotations.loc[self.annotations['labels'].isna(), 'supervised'] = 0
        
        if self.fully_supervised:
            self.annotations = self.annotations[self.annotations['supervised'] == 1]
            
        #replace the last 3 characters of every file path with extension
        self.annotations['file_path'] = self.annotations['file_path'].str[:-3] + extension
            
    def get_annotations(self):
        return self.fetch_function()
    
    def get_mtat_top50_annotations(self):
        csv_path = '/import/c4dm-datasets/MagnaTagATune/annotations_final.csv'
        annotations = pd.read_csv(csv_path, sep='\t')
        labels = annotations.drop(columns=['mp3_path', 'clip_id'])

        top_50_labels = labels.sum(axis=0).sort_values(ascending=False).head(50).index
        labels = labels[top_50_labels]

        label_sums = labels.sum(axis=1)
        unsupervised_annotations = annotations[label_sums == 0]
        annotations = annotations[label_sums > 0]
        labels = labels[label_sums > 0]

        annotations['labels'] = labels.values.tolist()
        annotations = annotations[['mp3_path', 'labels']]
        unsupervised_annotations['labels'] = None
        unsupervised_annotations = unsupervised_annotations[['mp3_path', 'labels']]

        val_folders = ['c/']
        test_folders = ['d/','e/', 'f/']

        annotations['split'] = 'train'
        annotations.loc[annotations['mp3_path'].str[:2].isin(val_folders), 'split'] = 'val'
        annotations.loc[annotations['mp3_path'].str[:2].isin(test_folders), 'split'] = 'test'

        n_supervised = int(len(annotations) * self.supervised_data_p)
        shuffle =  np.random.permutation(len(annotations))
        unsupervised_indices = shuffle[n_supervised:]
        temp_labels = annotations['labels']
        temp_labels.iloc[unsupervised_indices] = None
        annotations['labels'] = temp_labels

        unsupervised_annotations['split'] = 'train'
        annotations = pd.concat([annotations, unsupervised_annotations])
        
        ## rename the columns to match the default annotations
        annotations = annotations.rename(columns={'mp3_path':'file_path'})

        return annotations
    
    def get_mtat_all_annotations(self):
        
        csv_path = '/import/c4dm-datasets/MagnaTagATune/annotations_final.csv'
        annotations = pd.read_csv(csv_path, sep='\t')
        labels = annotations.drop(columns=['mp3_path', 'clip_id'])
        
        annotations['labels'] = labels.values.tolist()
        val_folders = ['c/']
        test_folders = ['d/','e/', 'f/']
        
        annotations['split'] = 'train'
        annotations.loc[annotations['mp3_path'].str[:2].isin(val_folders), 'split'] = 'val'
        annotations.loc[annotations['mp3_path'].str[:2].isin(test_folders), 'split'] = 'test'
        
        n_supervised = int(len(annotations) * self.supervised_data_p)
        shuffle =  np.random.permutation(len(annotations))
        unsupervised_indices = shuffle[n_supervised:]
        temp_labels = annotations['labels']
        temp_labels.iloc[unsupervised_indices] = None
        annotations.loc[:,'labels'] = temp_labels
        annotations = annotations[['mp3_path', 'labels','split']]
        annotations = annotations.rename(columns={'mp3_path':'file_path'})
        
        
        self.n_classes = len(labels.columns)
        
        return annotations
        
        
    def get_default_annotations(self):
        # read through data_dir, fetch any audio files, and random split train and val according to self.val_split
        # labels are None, or nan in the pandas dataframe
        
        file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav") or file.endswith(".mp3"):
                    # add the file path to file_list but exclude the data_dir
                    file_list.append(os.path.join(root, file)[len(self.data_dir) + 1:])
                    
        annotations = pd.DataFrame(file_list, columns = ['file_path'])
        annotations.loc[:,'split'] = 'train'
        annotations.loc[:,'labels'] = None
        
        
        
        if self.val_split > 0:
            train_len = int(len(annotations) * (1 - self.val_split))
            train_annotations, val_annotations = random_split(annotations, [train_len, len(annotations) - train_len])
            # turn train_annotations and val_annotations back into dataframes
            train_annotations = annotations.iloc[train_annotations.indices]
            val_annotations = annotations.iloc[val_annotations.indices]
            train_annotations.loc[:,'split'] = 'train'
            val_annotations.loc[:,'split'] = 'val'
            annotations = pd.concat([train_annotations, val_annotations])
            
        return annotations
        
        
    
    
    
class MixedDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 data_dir,
                 task = None,
                 target_length = 2.7,
                 target_sample_rate = 22050,
                 n_augmentations= 2,
                 transform = True,
                 n_classes = 50,
                 batch_size = 32,
                 num_workers = 16,
                 val_split = 0.1,
                 test_split = 0,
                 use_test_set = False,
                 supervised_data_p = 1,
                 fully_supervised = False,
                 intrabatch_supervised_p = 0.5) -> None:
        
        super().__init__()
        
        self.data_dir = data_dir
        self.task = task
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
        self.transform = transform
        self.test_split = test_split
        self.use_test_set = use_test_set
        self.fully_supervised = fully_supervised
        
        self.supervised_augmentations = Compose(
                [
                    Gain(
                        min_gain_in_db=-15.0,
                        max_gain_in_db=5.0,
                        p=0.4,
                        sample_rate=self.target_sample_rate
                    ),
                    PolarityInversion(p=0.6, sample_rate=self.target_sample_rate),
                    AddColoredNoise(p=0.6, sample_rate=self.target_sample_rate),
                    OneOf([
                        BandPassFilter(p=0.3, sample_rate = self.target_sample_rate),
                        BandStopFilter(p=0.3, sample_rate = self.target_sample_rate),
                        HighPassFilter(p=0.3, sample_rate = self.target_sample_rate),
                        LowPassFilter(p=0.3, sample_rate = self.target_sample_rate),
                    ]),
                    PitchShift(p=0.6, sample_rate = self.target_sample_rate),
                    Delay(p = 0.6, sample_rate = self.target_sample_rate),
                ],
                p=0.8,
            )
        self.self_supervised_augmentations = self.supervised_augmentations
        
        if self.fully_supervised:
            self.supervised_dataset_percentage = 1
            self.in_batch_supervised_percentage = 1
        
        self.splitter = DataModuleSplitter(self.data_dir, self.task, self.supervised_dataset_percentage, self.val_split, self.test_split, self.use_test_set, self.fully_supervised)
        self.annotations = self.splitter.annotations
        
        
        
        
    def setup(self, stage = 'fit'):
        
        
        supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 1)]
        supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 1)]
        supervised_test_annotations = self.annotations[(self.annotations['split'] == 'test') & (self.annotations['supervised'] == 1)]
        self_supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 0)]
        self_supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 0)]
        
        train_supervised_dataset = SupervisedDataset(self.data_dir, supervised_train_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  True, self.n_classes)
        val_supervised_dataset = SupervisedDataset(self.data_dir, supervised_val_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  False, self.n_classes)
        test_supervised_dataset = SupervisedTestDataset(self.data_dir, supervised_test_annotations, self.target_length, self.target_sample_rate, 1, False, None,  False, self.n_classes)
        train_self_supervised_dataset = SelfSupervisedDataset(self.data_dir, self_supervised_train_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.self_supervised_augmentations, True, self.n_classes)
        val_self_supervised_dataset = SelfSupervisedDataset(self.data_dir, self_supervised_val_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.self_supervised_augmentations, False, self.n_classes)
        
        
        # testing only makes sense if there is a supervised dataset and for fine-tuning
        
        if self.in_batch_supervised_percentage == 0 or len(train_supervised_dataset) == 0:
            train_supervised_dataset = None
            val_supervised_dataset = None
            test_supervised_dataset = None
        if self.in_batch_supervised_percentage == 1 or len(train_self_supervised_dataset) == 0:
            train_self_supervised_dataset = None
            val_self_supervised_dataset = None
        
        self.train_supervised_dataset = train_supervised_dataset
        self.val_supervised_dataset = val_supervised_dataset
        self.test_supervised_dataset = test_supervised_dataset
        self.train_self_supervised_dataset = train_self_supervised_dataset
        self.val_self_supervised_dataset = val_self_supervised_dataset
        
            
    def train_dataloader(self):
        return MixedDataLoader(self.train_supervised_dataset, self.train_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, batch_size = self.batch_size,  num_workers = self.num_workers)
        
    def val_dataloader(self):
        return MixedDataLoader(self.val_supervised_dataset, self.val_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return MixedDataLoader(self.test_supervised_dataset, None, 1, 1, batch_size = 1, num_workers = self.num_workers)