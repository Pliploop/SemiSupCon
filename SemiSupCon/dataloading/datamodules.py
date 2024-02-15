## boilerplate code for a pytorchlightning datamodule please

import os
import pytorch_lightning as pl
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from torch_audiomentations import *
import json

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
                 ssl_task= None,
                 sl_task= None,
                 supervised_data_p = 1,
                 val_split=0.1,
                 test_split = 0.1,
                 use_test_set = False,
                 fully_supervised = False,
                 extension = 'wav'
                 ):
        
        self.task = task
        self.ssl_task = ssl_task
        self.sl_task = sl_task
        self.data_dir = data_dir
        self.supervised_data_p = supervised_data_p
        self.fully_supervised = fully_supervised
        self.val_split = val_split
        self.test_split = test_split
        self.use_test_set = use_test_set
        if self.use_test_set==False:
            self.test_split = 0
            
        
        self.n_classes = 0
            
        if self.ssl_task == self.sl_task:
            print('what the hekk')
            if self.ssl_task is None:
                fetch_function = self.get_default_annotations
            else:
                fetch_function = eval(f"self.get_{self.ssl_task}_annotations")
            annotations, idx2class = fetch_function()
            self.annotations = self.filter_supervised_annotations(annotations, self.supervised_data_p)
            self.annotations['task'] = self.ssl_task
            
            
        else:
            if self.ssl_task is None:
                ssl_fetch_function = self.get_default_annotations
            else:
                ssl_fetch_function = eval(f"self.get_{self.ssl_task}_annotations")
            if self.sl_task is None:
                sl_fetch_function = self.get_default_annotations
            else:
                sl_fetch_function = eval(f"self.get_{self.sl_task}_annotations")
                
            ssl_annotations, _ = ssl_fetch_function()
            sl_annotations, idx2class = sl_fetch_function()
            
            # raise ValueError('not implemented')
            
            
            ssl_annotations.loc[:,'labels'] = None
            sl_annotations = self.filter_supervised_annotations(sl_annotations, self.supervised_data_p, drop = True)
            ssl_annotations['task'] = self.ssl_task
            sl_annotations['task'] = self.sl_task
            
            self.annotations = pd.concat([ssl_annotations, sl_annotations])
       
        self.idx2class = idx2class
            
            
        
        
        # self.annotations = self.get_annotations()
        
        if self.use_test_set==False:
            # change the split column to train where it is 'test'
            self.annotations.loc[self.annotations['split'] == 'test', 'split'] = 'train'
            
        
        
        # build new column for annotations dataframe that indicates whether the data is supervised or not
        self.annotations['supervised'] = 1
        self.annotations.loc[self.annotations['labels'].isna(), 'supervised'] = 0
        
        if self.fully_supervised:
            self.annotations = self.annotations[self.annotations['supervised'] == 1]
            
        # #replace the last 3 characters of every file path with extension
        # self.annotations['file_path'] = self.annotations['file_path'].str[:-3] + extension
            
    def get_annotations(self):
        return self.fetch_function()
    
    def get_fma_annotations(self):
        path = 'data/fma_medium.csv' # just because for some weird reason it takes forever to read the files
        annotations = pd.read_csv(path)
        annotations.loc[:,'split'] = 'train'
        annotations.loc[:,'labels'] = None
        
        annotations = annotations[['file_path']]
        
        if self.val_split > 0:
            train_len = int(len(annotations) * (1 - self.val_split))
            train_annotations, val_annotations = random_split(annotations, [train_len, len(annotations) - train_len])
            # turn train_annotations and val_annotations back into dataframes
            train_annotations = annotations.iloc[train_annotations.indices]
            val_annotations = annotations.iloc[val_annotations.indices]
            train_annotations.loc[:,'split'] = 'train'
            val_annotations.loc[:,'split'] = 'val'
            annotations = pd.concat([train_annotations, val_annotations])
            
        return annotations,None
    
    def filter_supervised_annotations(self, annotations, supervised_data_p, drop = False):
        supervised_annotations = annotations[annotations.labels.notna()]
        unsupervised_annotations = annotations[annotations.labels.isna()]
        
        n_supervised = int(len(supervised_annotations) * supervised_data_p)
        shuffle =  np.random.permutation(len(supervised_annotations))
        unsupervised_indices = shuffle[n_supervised:]
        temp_labels = supervised_annotations['labels']
        temp_labels.iloc[unsupervised_indices] = None
        annotations.loc[:,'labels'] = temp_labels
        annotations = pd.concat([supervised_annotations, unsupervised_annotations])
        
        # annotations = annotations[['file_path', 'labels','split']]
        
        if drop :
            annotations = annotations[annotations['labels'].notna()]
        
        return annotations
        
    
    def get_mtat_top50_annotations(self):
        csv_path = '/import/c4dm-datasets/MagnaTagATune/annotations_final.csv'
        annotations = pd.read_csv(csv_path, sep='\t')
        labels = annotations.drop(columns=['mp3_path', 'clip_id'])

        top_50_labels = labels.sum(axis=0).sort_values(ascending=False).head(50).index
        labels = labels[top_50_labels]
        
        
        ## rename the columns to match the default annotations
        annotations = annotations.rename(columns={'mp3_path':'file_path'})

        label_sums = labels.sum(axis=1)
        unsupervised_annotations = annotations[label_sums == 0]
        annotations = annotations[label_sums > 0]
        labels = labels[label_sums > 0]

        annotations['labels'] = labels.values.tolist()
        annotations = annotations[['file_path', 'labels']]
        unsupervised_annotations['labels'] = None
        unsupervised_annotations = unsupervised_annotations[['file_path', 'labels']]
        unsupervised_annotations['split'] = 'train'

        val_folders = ['c/']
        test_folders = ['d/','e/', 'f/']

        annotations['split'] = 'train'
        annotations.loc[annotations['file_path'].str[:2].isin(val_folders), 'split'] = 'val'
        annotations.loc[annotations['file_path'].str[:2].isin(test_folders), 'split'] = 'test'


        annotations = pd.concat([annotations, unsupervised_annotations])
        
        class2idx = {c: i for i, c in enumerate(labels.columns)}
        idx2class = {i: c for i, c in enumerate(labels.columns)}
        
        #replace .mp3 with .wav
        annotations['file_path'] = annotations['file_path'].str[:-3] + 'wav'
        self.n_classes= 50

        return annotations, idx2class
    
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
        annotations = annotations.rename(columns={'mp3_path':'file_path'})
        self.n_classes = len(labels.columns)
        
        class2idx = {c: i for i, c in enumerate(labels.columns)}
        idx2class = {i: c for i, c in enumerate(labels.columns)}
        
        
        annotations['file_path'] = annotations['file_path'].str[:-3] + 'wav'
        
        return annotations, idx2class
        
        
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
            
        return annotations, None
        
    def get_gtzan_annotations(self):
        audio_path = "/import/c4dm-datasets/gtzan_torchaudio/genres"
        # annotations = pd.read_csv("data/gtzan_annotations.csv")
        #read txt files into dataframes
        
        train_annotations = pd.read_csv("data/gtzan/train_filtered.txt", sep = ' ', header = None)
        val_annotations = pd.read_csv("data/gtzan/val_filtered.txt", sep = ' ', header = None)
        test_annotations = pd.read_csv("data/gtzan/test_filtered.txt", sep = ' ', header = None)
        
        train_annotations['split'] = 'train'
        val_annotations['split'] = 'val'
        test_annotations['split'] = 'test'
        
        annotations = pd.concat([train_annotations,val_annotations,test_annotations])
        print(annotations.head())
        annotations.columns = ['file_path','split']
        
        annotations['genre'] = annotations['file_path'].apply(lambda x:x.split('/')[0])

        self.n_classes = len(annotations['genre'].unique())
        
        annotations['file_path'] = audio_path + '/' + annotations['file_path']
        
        class2idx = {c: i for i, c in enumerate(annotations['genre'].unique())}
        idx2class = {i: c for i, c in enumerate(annotations['genre'].unique())}
        annotations['labels'] = annotations['genre'].apply(lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(annotations['labels']).values.astype(int).tolist()
        
        return annotations, idx2class
    
    def get_giantsteps_annotations(self):
        test_audio_path = "/homes/jpmg86/giantsteps-key-dataset/audio"
        test_annotations_path = "/homes/jpmg86/giantsteps-key-dataset/annotations/key"

        # every file in the annotations path is of shape {filename}.LOFI.key
        # and when read contains the key as a string.
        # build the annotations file with the audio files in audio_path and the keys in the annotations_path
        test_annotations = pd.DataFrame(os.listdir(test_audio_path), columns=['file_path'])
        test_annotations['split'] = 'test'
        test_annotations['labels'] = None
        test_annotations['annotation_file'] = test_annotations_path + '/' + test_annotations['file_path'].str[:-4] + '.key'
        test_annotations['file_path'] = test_audio_path + '/' + test_annotations['file_path']

        for idx, row in test_annotations.iterrows():
            with open(row['annotation_file'], 'r') as f:
                key = f.read()
                test_annotations.loc[idx, 'key'] = key

        # do a random split of the data into train, val and test, put this into the dataframe as a column "split"

        # test_annotations["split"] = np.random.choice(["train", "val"], size=len(test_annotations), p=[
        #                                         1-self.val_split-self.val_split])
        test_classes = test_annotations['key'].unique()

        train_audio_path = "/homes/jpmg86/giantsteps-mtg-key-dataset/audio"
        train_annotations_txt = '/homes/jpmg86/giantsteps-mtg-key-dataset/annotations/annotations.txt'

        train_annotations = pd.read_csv(train_annotations_txt, sep='\t')
        train_annotations = train_annotations.iloc[:, :3]
        train_annotations.columns = ['file_path', 'key', 'confidence']
        train_annotations = train_annotations[train_annotations['confidence'] == 2]

        # train_annotations = pd.DataFrame(os.listdir(train_audio_path), columns = ['file_path'])
        train_annotations['split'] = 'train'
        train_annotations['labels'] = None
        train_annotations['file_path'] = train_audio_path + '/' + train_annotations['file_path'].astype(str) + '.LOFI.mp3'

        # train_annotations['key'] = train_annotations['key'].apply(lambda x: x.split('/')[0].strip())
        train_annotations = train_annotations[~train_annotations['key'].str.contains('/')]
        train_annotations = train_annotations[train_annotations['key'].notna()]
        train_annotations = train_annotations[train_annotations['key'] != '-']

        enharmonic = {
            "C#": "Db",
            "D#": "Eb",
            "F#": "Gb",
            "G#": "Ab",
            "A#": "Bb",
        }


        # train_annotations = train_annotations[train_annotations['key'].isin(test_classes)]
        train_annotations['split'] = np.random.choice(["train", "val"], size=len(train_annotations),
                                                      p=[1 - self.val_split, self.val_split])

       
        

        annotations = pd.concat([train_annotations, test_annotations])

        annotations['key'] = annotations['key'].replace(enharmonic, regex=True)
        annotations['key'] = annotations['key'].apply(lambda x:x.strip())


        class2idx = {c: i for i, c in enumerate(annotations['key'].unique())}
        idx2class = {i: c for i, c in enumerate(annotations['key'].unique())}

        annotations['labels'] = annotations['key'].apply(lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(annotations['labels']).values.astype(int).tolist()

        self.n_classes = len(annotations['key'].unique())
        print(self.n_classes)

        return annotations, idx2class
        
        
    
    def get_emomusic_annotations(self):
        
        raise NotImplementedError('not implemented')
    
    def get_nsynth_instr_family_annotations(self):
        return self.get_nsynth_annotations('instrument_family')
    
    def get_nsynth_instr_annotations(self):
        return self.get_nsynth_annotations('instrument')
    
    def get_nsynth_pitch_annotations(self):
        return self.get_nsynth_annotations('pitch')
    
    def get_nsynth_qualities_annotations(self):
        annotations = self.get_nsynth_annotations('instrument_family')
        annotations['labels'] = annotations["qualities"]
        self.n_classes = len(annotations['labels'][0])
        return annotations
    
    def get_nsynth_annotations(self,class_name):
        all_data = {}
        path_dir = '/import/c4dm-datasets/nsynth/nsynth'
        for split in 'train', 'valid', 'test':
            path = os.path.join(path_dir+'-'+split, 'examples.json')
            with open(path, 'r') as f:
                data = json.load(f)
                # add split, data is of shape {sample_key : dict}. We want dict[split] = split
                for key in data.keys():
                    data[key]['split'] = split
                    
                all_data.update(data)
                
        annotations = pd.DataFrame(list(all_data.values()))
        annotations['file_path'] = path_dir +'-' + annotations['split'] + '/audio/' + annotations['note_str'] + '.wav'
        # replace 'split' with 'train', 'val', 'test'
        annotations['split'] = annotations['split'].apply(lambda x: 'train' if x == 'train' else 'val' if x == 'valid' else 'test')
    
    
        # get the number of classes for column "instrument_family"
        self.n_classes = len(annotations[class_name].unique())
        # pretty print the number of classes
        print(f'Number of classes: {self.n_classes}')
        # add 'labels' to the dataframe as a one-hot of classes to int
        
        class2idx = {c: i for i, c in enumerate(annotations[class_name].unique())}
        idx2class = {i: c for i, c in enumerate(annotations[class_name].unique())}
        annotations['labels'] = annotations[class_name].apply(lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(annotations['labels']).values.astype(int).tolist()
        
        return annotations, idx2class
        
            
    def get_vocalset_singer_annotations(self):
        annotations = pd.DataFrame(columns=['file_path', 'labels'])
        data_dir = '/import/c4dm-datasets/VocalSet1-2/data_by_singer'
        raise NotImplementedError('not implemented')
        
    def get_vocalset_technique_annotations(self):
        
        data_dir = '/import/c4dm-datasets/VocalSet1-2'
        train_singers = pd.read_csv(os.path.join(data_dir, 'train_singers_technique.txt'))
        test_singers = pd.read_csv(os.path.join(data_dir, 'test_singers_technique.txt'))
        train_singers.columns = ['id']
        test_singers.columns = ['id']
        
        train_singers['split'] = 'train'
        test_singers['split'] = 'test'
        
        id_to_split = pd.concat([train_singers, test_singers])
        id_to_split['id'] = id_to_split.id.apply(lambda x: x[0]+x[-1])
        
        annotations = pd.DataFrame(columns=['file_path', 'labels', 'split'])
        
        for root, dirs, files in os.walk(os.path.join(data_dir, 'data_by_technique')):
            for file in files:
                if file.endswith(".wav"):
                    singer_id = file.split('_')[0]
                    technique = root.split('/')[-1]
                    split = annotations[annotations['id'] == singer_id]['split']
                    annotations = annotations.append({'file_path': os.path.join(root, file), 'label_name': technique, 'split': split}, ignore_index=True)
        
        class2idx = {c: i for i, c in enumerate(annotations['label_name'].unique())}
        idx2class = {i: c for i, c in enumerate(annotations['label_name'].unique())}
        annotations['labels'] = annotations['label_name'].apply(lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(annotations['labels']).values.astype(int).tolist()
        # split the train data into train and val
        annotations['split'] = np.random.choice(["train", "val"], size=len(annotations), p=[1-self.val_split, self.val_split])
        
        
        return annotations, idx2class
    
    def get_mtg_annotations(self,path,audio_path):
        
        annotations = []
        
        class2idx = {}
        for split in ['train', 'validation', 'test']:
            data = open(path.replace("split.tsv",f"{split}.tsv"), "r").readlines()
            all_paths = [line.split('\t')[3] for line in data[1:]]
            all_tags = [line[:-1].split('\t')[5:] for line in data[1:]]
            annotations.append(pd.DataFrame({"file_path":all_paths, "tags":all_tags, "split":split}))            
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2idx:
                        class2idx[tag] = len(class2idx)
                       
        idx2class = {i: c for i, c in enumerate(class2idx.keys())}
        annotations = pd.concat(annotations) 
        
        #replace mp3 extensions with wav in path columns
        annotations["split"] = annotations["split"].str.replace("validation", "val")
        
        self.n_classes = len(class2idx)
        print(f'Number of classes: {self.n_classes}')
        
        # the "labels" column is a list of tags, we need to one-hot encode it
        annotations['idx'] = annotations['tags'].apply(lambda x: [class2idx[tag] for tag in x])
        # now "labels" is a list of indices, we need to one-hot encode it into one on-hot vector per example
        annotations['labels'] = annotations['idx'].apply(lambda x: np.sum(np.eye(len(class2idx))[x], axis=0).astype(int).tolist())
        annotations['file_path'] = audio_path + '/' + annotations['file_path']
        
        return annotations,idx2class
    
    def get_mtg_top50_annotations(self):
        
        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_top50tags-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"
        
        return self.get_mtg_annotations(path,audio_path)
        
        
        
    
    def get_mtg_instr_annotations(self):
        
        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"
        
        return self.get_mtg_annotations(path,audio_path)
        
    
    def get_mtg_genre_annotations(self):
        
        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_genre-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"
        
        return self.get_mtg_annotations(path,audio_path)
        
    def get_mtg_mood_annotations(self):
        
        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"
        
        return self.get_mtg_annotations(path,audio_path)
    
    def get_openmic_annotations(self):
        
        raise NotImplementedError('not implemented')
    
class MixedDataModule(pl.LightningDataModule):
    
    def __init__(self,
                 data_dir,
                 task = None,
                 sl_task = None,
                 ssl_task = None,
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
                 intrabatch_supervised_p = 0.5,
                 aug_list = []) -> None:
        
        super().__init__()
        
        self.data_dir = data_dir
        self.task = task
        self.sl_task = sl_task
        self.ssl_task = ssl_task
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
        self.aug_list = aug_list
        
        #build a dict with augmentation name and the corresponding function with parameters as specified below
        self.augmentations = {
            'gain': Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.4, sample_rate=self.target_sample_rate),
            'polarity_inversion': PolarityInversion(p=0.6, sample_rate=self.target_sample_rate),
            'add_colored_noise': AddColoredNoise(p=0.6, sample_rate=self.target_sample_rate),
            'filtering': OneOf([
                BandPassFilter(p=0.3, sample_rate = self.target_sample_rate),
                BandStopFilter(p=0.3, sample_rate = self.target_sample_rate),
                HighPassFilter(p=0.3, sample_rate = self.target_sample_rate),
                LowPassFilter(p=0.3, sample_rate = self.target_sample_rate),
            ]),
            'pitch_shift': PitchShift(p=0.6, sample_rate = self.target_sample_rate),
            'delay': Delay(p = 0.6, sample_rate = self.target_sample_rate),
        }
        
        # build an augmentation pipeline with the above augmentations if they are in self.aug_list
        self.supervised_augmentations = Compose(
            [
                self.augmentations[aug] for aug in self.aug_list
            ],
            p=0.8,
        )
        
        self.self_supervised_augmentations = self.supervised_augmentations
        
        if self.fully_supervised:
            self.supervised_dataset_percentage = 1
            self.in_batch_supervised_percentage = 1
        
        self.splitter = DataModuleSplitter(self.data_dir, self.task, self.ssl_task, self.sl_task, self.supervised_dataset_percentage, self.val_split, self.test_split, self.use_test_set, self.fully_supervised)
        self.annotations = self.splitter.annotations
        self.n_classes = self.splitter.n_classes
        if self.splitter.idx2class is not None:
            self.idx2class = self.splitter.idx2class
            self.class_names = list(self.idx2class.values())
        else:
            self.idx2class = None
            self.class_names = None
        
        print(self.annotations.groupby('split').count())
        
        
    def setup(self, stage = 'fit'):
        
        
        supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 1)]
        supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 1)]
        supervised_test_annotations = self.annotations[(self.annotations['split'] == 'test') & (self.annotations['supervised'] == 1)]
        self_supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 0)]
        self_supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 0)]
        
        train_supervised_dataset = SupervisedDataset(self.data_dir, supervised_train_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  True, self.n_classes)
        val_supervised_dataset = SupervisedDataset(self.data_dir, supervised_val_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  True, self.n_classes)
        test_supervised_dataset = SupervisedTestDataset(self.data_dir, supervised_test_annotations, self.target_length, self.target_sample_rate, 1, False, None,  False, self.n_classes)
        train_self_supervised_dataset = SelfSupervisedDataset(self.data_dir, self_supervised_train_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.self_supervised_augmentations, True, self.n_classes)
        val_self_supervised_dataset = SelfSupervisedDataset(self.data_dir, self_supervised_val_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.self_supervised_augmentations, True, self.n_classes)
        
        
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
        return MixedDataLoader(self.train_supervised_dataset, self.train_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, batch_size = self.batch_size,  num_workers = self.num_workers, n_classes = self.n_classes)
        
    def val_dataloader(self):
        return MixedDataLoader(self.val_supervised_dataset, self.val_self_supervised_dataset, self.supervised_dataset_percentage, self.in_batch_supervised_percentage, batch_size = self.batch_size, num_workers = self.num_workers, n_classes = self.n_classes)
    
    def test_dataloader(self):
        return MixedDataLoader(self.test_supervised_dataset, None, 1, 1, batch_size = 1, num_workers = self.num_workers, n_classes = self.n_classes)