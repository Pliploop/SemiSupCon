
import os
from torch.utils.data import random_split
import pandas as pd
import numpy as np
from torch_audiomentations import *
import json
import librosa
import hashlib
import gzip
import csv
import pathlib
import yaml
from sklearn.model_selection import train_test_split



def compute_checksum(path_or_bytes, algorithm="sha256", gunzip=False, chunk_size=4096):
    """Computes checksum of target path.

    Parameters
    ----------
    path_or_bytes : :class:`pathlib.Path` or bytes
    Location or bytes of file to compute checksum for.
    algorithm : str, optional
    Hash algorithm (from :func:`hashlib.algorithms_available`); default ``sha256``.
    gunzip : bool, optional
    If true, decompress before computing checksum.
    chunk_size : int, optional
    Chunk size for iterating through file.

    Raises
    ------
    :class:`FileNotFoundError`
    Unknown path.
    :class:`IsADirectoryError`
    Path is a directory.
    :class:`ValueError`
    Unknown algorithm.

    Returns
    -------
    str
    Hex representation of checksum.
    """
    if algorithm not in hashlib.algorithms_guaranteed or algorithm.startswith("shake"):
        raise ValueError("Unknown algorithm")
    computed = hashlib.new(algorithm)
    if isinstance(path_or_bytes, bytes):
        computed.update(path_or_bytes)
    else:
        open_fn = gzip.open if gunzip else open
        with open_fn(path_or_bytes, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                computed.update(data)
    return computed.hexdigest()


def parse_minsec(s):
    s = s.split(".")
    t = float(s[0]) * 60
    if len(s) > 1:
        assert len(s) == 2
        if len(s[1]) == 1:
            s[1] += "0"
        t += float(s[1])
    return t


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
                 ssl_task=None,
                 sl_task=None,
                 supervised_data_p=1,
                 val_split=0.1,
                 test_split=0.1,
                 use_test_set=False,
                 fully_supervised=False,
                 extension='wav'
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
        if self.use_test_set == False:
            self.test_split = 0

        self.n_classes = 0

        if self.ssl_task == self.sl_task:
            if self.ssl_task is None:
                fetch_function = self.get_default_annotations
            else:
                fetch_function = eval(f"self.get_{self.ssl_task}_annotations")
            annotations, idx2class = fetch_function()
            self.annotations = self.filter_supervised_annotations(
                annotations, self.supervised_data_p)
            self.annotations['task'] = self.ssl_task

        else:
            if self.ssl_task is None:
                ssl_fetch_function = self.get_default_annotations
            else:
                ssl_fetch_function = eval(
                    f"self.get_{self.ssl_task}_annotations")
            if self.sl_task is None:
                sl_fetch_function = self.get_default_annotations
            else:
                sl_fetch_function = eval(
                    f"self.get_{self.sl_task}_annotations")

            ssl_annotations, _ = ssl_fetch_function()
            sl_annotations, idx2class = sl_fetch_function()

            ssl_annotations.loc[:, 'labels'] = None
            sl_annotations = self.filter_supervised_annotations(
                sl_annotations, self.supervised_data_p, drop=True)
            ssl_annotations['task'] = self.ssl_task
            sl_annotations['task'] = self.sl_task

            self.annotations = pd.concat([ssl_annotations, sl_annotations])

        self.idx2class = idx2class

        if self.use_test_set == False:
            # change the split column to train where it is 'test'
            self.annotations.loc[self.annotations['split']
                                 == 'test', 'split'] = 'train'

        self.annotations['supervised'] = 1
        self.annotations.loc[self.annotations['labels'].isna(),
                             'supervised'] = 0

        if self.fully_supervised:
            self.annotations = self.annotations[self.annotations['supervised'] == 1]

    def get_annotations(self):
        return self.fetch_function()

    def get_fma_annotations(self):
        # just because for some weird reason it takes forever to read the files using get default annotations
        path = 'data/fma/fma_medium_wav.csv'
        annotations = pd.read_csv(path)
        annotations.loc[:, 'split'] = 'train'
        annotations.loc[:, 'labels'] = None

        annotations = annotations[['file_path']]

        if self.val_split > 0:
            train_len = int(len(annotations) * (1 - self.val_split))
            train_annotations, val_annotations = random_split(
                annotations, [train_len, len(annotations) - train_len])
            train_annotations = annotations.iloc[train_annotations.indices]
            val_annotations = annotations.iloc[val_annotations.indices]
            train_annotations.loc[:, 'split'] = 'train'
            val_annotations.loc[:, 'split'] = 'val'
            annotations = pd.concat([train_annotations, val_annotations])

        return annotations, None

    def filter_supervised_annotations(self, annotations, supervised_data_p, drop=False):
        supervised_annotations = annotations[annotations.labels.notna()]
        unsupervised_annotations = annotations[annotations.labels.isna()]

        n_supervised = int(len(supervised_annotations) * supervised_data_p)
        shuffle = np.random.permutation(len(supervised_annotations))
        unsupervised_indices = shuffle[n_supervised:]
        temp_labels = supervised_annotations['labels']
        temp_labels.iloc[unsupervised_indices] = None
        annotations.loc[:, 'labels'] = temp_labels
        annotations = pd.concat(
            [supervised_annotations, unsupervised_annotations])

        # annotations = annotations[['file_path', 'labels','split']]

        if drop:
            annotations = annotations[annotations['labels'].notna()]

        return annotations

    def get_mtat_top50_annotations(self):
        csv_path = '/import/c4dm-datasets/MagnaTagATune/annotations_final.csv'
        annotations = pd.read_csv(csv_path, sep='\t')
        labels = annotations.drop(columns=['mp3_path', 'clip_id'])

        top_50_labels = labels.sum(axis=0).sort_values(
            ascending=False).head(50).index
        labels = labels[top_50_labels]

        # rename the columns to match the default annotations
        annotations = annotations.rename(columns={'mp3_path': 'file_path'})

        label_sums = labels.sum(axis=1)
        unsupervised_annotations = annotations[label_sums == 0]
        annotations = annotations[label_sums > 0]
        labels = labels[label_sums > 0]

        annotations['labels'] = labels.values.tolist()
        annotations = annotations[['file_path', 'labels']]
        unsupervised_annotations['labels'] = None
        unsupervised_annotations = unsupervised_annotations[[
            'file_path', 'labels']]
        unsupervised_annotations['split'] = 'train'

        val_folders = ['c/']
        test_folders = ['d/', 'e/', 'f/']

        annotations['split'] = 'train'
        annotations.loc[annotations['file_path'].str[:2].isin(
            val_folders), 'split'] = 'val'
        annotations.loc[annotations['file_path'].str[:2].isin(
            test_folders), 'split'] = 'test'

        annotations = pd.concat([annotations, unsupervised_annotations])

        class2idx = {c: i for i, c in enumerate(labels.columns)}
        idx2class = {i: c for i, c in enumerate(labels.columns)}

        # replace .mp3 with .wav
        annotations['file_path'] = annotations['file_path'].str[:-3] + 'wav'
        self.n_classes = 50

        return annotations, idx2class

    def get_mtat_all_annotations(self):

        csv_path = '/import/c4dm-datasets/MagnaTagATune/annotations_final.csv'
        annotations = pd.read_csv(csv_path, sep='\t')
        labels = annotations.drop(columns=['mp3_path', 'clip_id'])

        annotations['labels'] = labels.values.tolist()
        val_folders = ['c/']
        test_folders = ['d/', 'e/', 'f/']

        annotations['split'] = 'train'
        annotations.loc[annotations['mp3_path'].str[:2].isin(
            val_folders), 'split'] = 'val'
        annotations.loc[annotations['mp3_path'].str[:2].isin(
            test_folders), 'split'] = 'test'
        annotations = annotations.rename(columns={'mp3_path': 'file_path'})
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
                    file_list.append(os.path.join(root, file)[
                                     len(self.data_dir) + 1:])

        annotations = pd.DataFrame(file_list, columns=['file_path'])
        annotations.loc[:, 'split'] = 'train'
        annotations.loc[:, 'labels'] = None

        if self.val_split > 0:
            train_len = int(len(annotations) * (1 - self.val_split))
            train_annotations, val_annotations = random_split(
                annotations, [train_len, len(annotations) - train_len])
            # turn train_annotations and val_annotations back into dataframes
            train_annotations = annotations.iloc[train_annotations.indices]
            val_annotations = annotations.iloc[val_annotations.indices]
            train_annotations.loc[:, 'split'] = 'train'
            val_annotations.loc[:, 'split'] = 'val'
            annotations = pd.concat([train_annotations, val_annotations])

        return annotations, None

    def get_gtzan_annotations(self):
        audio_path = "/import/c4dm-datasets/gtzan_torchaudio/genres"
        # annotations = pd.read_csv("data/gtzan_annotations.csv")
        # read txt files into dataframes

        train_annotations = pd.read_csv(
            "data/gtzan/train_filtered.txt", sep=' ', header=None)
        val_annotations = pd.read_csv(
            "data/gtzan/val_filtered.txt", sep=' ', header=None)
        test_annotations = pd.read_csv(
            "data/gtzan/test_filtered.txt", sep=' ', header=None)

        train_annotations['split'] = 'train'
        val_annotations['split'] = 'val'
        test_annotations['split'] = 'test'

        annotations = pd.concat(
            [train_annotations, val_annotations, test_annotations])
        annotations.columns = ['file_path', 'split']

        annotations['genre'] = annotations['file_path'].apply(
            lambda x: x.split('/')[0])

        self.n_classes = len(annotations['genre'].unique())

        annotations['file_path'] = audio_path + '/' + annotations['file_path']

        class2idx = {c: i for i, c in enumerate(annotations['genre'].unique())}
        idx2class = {i: c for i, c in enumerate(annotations['genre'].unique())}
        annotations['labels'] = annotations['genre'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()

        return annotations, idx2class

    def get_giantsteps_annotations(self):
        test_audio_path = "/homes/jpmg86/giantsteps-key-dataset/audio"
        test_annotations_path = "/homes/jpmg86/giantsteps-key-dataset/annotations/key"

        # every file in the annotations path is of shape {filename}.LOFI.key
        # and when read contains the key as a string.
        # build the annotations file with the audio files in audio_path and the keys in the annotations_path
        test_annotations = pd.DataFrame(os.listdir(
            test_audio_path), columns=['file_path'])
        test_annotations['split'] = 'test'
        test_annotations['labels'] = None
        test_annotations['annotation_file'] = test_annotations_path + \
            '/' + test_annotations['file_path'].str[:-4] + '.key'
        test_annotations['file_path'] = test_audio_path + \
            '/' + test_annotations['file_path']

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
        train_annotations['file_path'] = train_audio_path + '/' + \
            train_annotations['file_path'].astype(str) + '.LOFI.mp3'

        # train_annotations['key'] = train_annotations['key'].apply(lambda x: x.split('/')[0].strip())
        train_annotations = train_annotations[~train_annotations['key'].str.contains(
            '/')]
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
        annotations['key'] = annotations['key'].apply(lambda x: x.strip())

        class2idx = {c: i for i, c in enumerate(annotations['key'].unique())}
        idx2class = {i: c for i, c in enumerate(annotations['key'].unique())}

        annotations['labels'] = annotations['key'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()

        self.n_classes = len(annotations['key'].unique())

        return annotations, idx2class

    def get_emomusic_annotations(self):
        d = "/import/c4dm-datasets/emoMusic45s"

        # prase annotations CSV
        audio_uids = set()
        uid_to_metadata = {}
        for stem in [
            "songs_info",
            "static_annotations",
            "valence_cont_average",
            "valence_cont_std",
            "arousal_cont_average",
            "arousal_cont_std",
        ]:
            with open(pathlib.Path(d, f"{stem}.csv"), "r") as f:
                for row in csv.DictReader(f):
                    row = {k: v.strip() for k, v in row.items()}
                    uid = str(int(row["song_id"])).zfill(4)
                    if stem == "songs_info":
                        assert uid not in uid_to_metadata
                        audio_uid = (row["Artist"], row["Song title"])
                        # NOTE: Only one clip per song in this dataset
                        assert audio_uid not in audio_uids
                        audio_uids.add(audio_uid)
                        clip_start = parse_minsec(
                            row["start of the segment (min.sec)"])
                        clip_end = parse_minsec(
                            row["end of the segment (min.sec)"])
                        clip_dur = clip_end - clip_start
                        assert clip_dur == 45.0
                        uid_to_metadata[uid] = {
                            "split": "test"
                            if row["Mediaeval 2013 set"] == "evaluation"
                            else "train",
                            "clip": {
                                "audio_uid": audio_uid,
                                "audio_duration": clip_end,
                                "clip_idx": 0,
                                "clip_offset": clip_start,
                            },
                            "y": None,
                            "extra": {},
                        }
                    else:
                        assert uid in uid_to_metadata
                    uid_to_metadata[uid]["extra"][stem] = row
                    if stem == "static_annotations":
                        uid_to_metadata[uid]["y"] = [
                            float(row["mean_arousal"]),
                            float(row["mean_valence"]),
                        ]

        # Normalize
        arousals = [
            metadata["y"][0]
            for metadata in uid_to_metadata.values()
            if metadata["split"] == "train"
        ]
        valences = [
            metadata["y"][1]
            for metadata in uid_to_metadata.values()
            if metadata["split"] == "train"
        ]
        arousal_mean = np.mean(arousals)
        arousal_std = np.std(arousals)
        valence_mean = np.mean(valences)
        valence_std = np.std(valences)
        for metadata in uid_to_metadata.values():
            metadata["y"] = [
                (metadata["y"][0] - arousal_mean) / arousal_std,
                (metadata["y"][1] - valence_mean) / valence_std,
            ]

        # split train/valid/test
        ratios = ["train"] * 8 + ["valid"] * 2
        for uid, metadata in uid_to_metadata.items():
            if metadata["split"] == "train":
                artist = metadata["extra"]["songs_info"]["Artist"]
                artist = "".join(
                    [
                        c
                        for c in artist.lower()
                        if (ord(c) < 128 and (c.isalpha() or c.isspace()))
                    ]
                )
                artist = " ".join(artist.split())
                artist_id = int(
                    compute_checksum(artist.encode("utf-8"),
                                     algorithm="sha1"), 16
                )
                split = ratios[artist_id % len(ratios)]
                metadata["split"] = split

        # Yield unique id, metadata, and path (if downloaded) for each audio clip.
        results = []
        for uid, metadata in uid_to_metadata.items():
            # Yield result
            split = metadata["split"]
            mp3_path = pathlib.Path(d, "clips_45seconds", f"{int(uid)}.mp3")
            labels = metadata["y"]

            result = {
                "file_path": mp3_path,
                "split": split,
                "labels": labels,
            }

            results.append(result)
        df = pd.DataFrame(results)
        self.n_classes = 2
        # replace valid with val
        df['split'] = df['split'].replace('valid', 'val')

        return df, None

    def get_nsynth_instr_family_annotations(self):
        return self.get_nsynth_annotations('instrument_family')

    def get_nsynth_instr_annotations(self):
        return self.get_nsynth_annotations('instrument')

    def get_nsynth_pitch_annotations(self):
        annotations, idx2class = self.get_nsynth_annotations('pitch')
        idx2class = {i: librosa.midi_to_note(c) for i, c in idx2class.items()}
        return annotations, idx2class

    def get_nsynth_qualities_annotations(self):
        annotations = self.get_nsynth_annotations('instrument_family')
        annotations['labels'] = annotations["qualities"]
        self.n_classes = len(annotations['labels'][0])
        return annotations

    def get_nsynth_annotations(self, class_name):
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
        annotations['file_path'] = path_dir + '-' + annotations['split'] + \
            '/audio/' + annotations['note_str'] + '.wav'
        # replace 'split' with 'train', 'val', 'test'
        annotations['split'] = annotations['split'].apply(
            lambda x: 'train' if x == 'train' else 'val' if x == 'valid' else 'test')

        # get the number of classes for column "instrument_family"
        self.n_classes = len(annotations[class_name].unique())
        # pretty print the number of classes
        # add 'labels' to the dataframe as a one-hot of classes to int

        class2idx = {c: i for i, c in enumerate(
            annotations[class_name].unique())}
        idx2class = {i: c for i, c in enumerate(
            annotations[class_name].unique())}
        annotations['labels'] = annotations[class_name].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()

        return annotations, idx2class

    def get_vocalset_singer_annotations(self):

        data_dir = '/import/c4dm-datasets/VocalSet1-2'

        # annotations = pd.DataFrame(columns=['file_path', 'labels', 'split'])
        annotations = []

        for root, dirs, files in os.walk(os.path.join(data_dir, 'data_by_singer')):
            for file in files:
                if file.endswith(".wav"):
                    # the singer_id is the first folder after root in dirs
                    singer_id = file.split('_')[0]
                    file_path = os.path.join(root, os.path.join(root, file))
                    annotation = {
                        'singer_id': singer_id,
                        'file_path': file_path,
                    }

                    annotations.append(annotation)

        annotations = pd.DataFrame(annotations, columns=[
                                   'singer_id', 'file_path'])
        annotations['split'] = np.random.choice(["train", "val", "test"], size=len(
            annotations), p=[1 - 2*self.val_split, self.val_split, self.val_split])

        annotations['label_name'] = annotations["singer_id"]

        class2idx = {c: i for i, c in enumerate(
            annotations['label_name'].unique())}
        idx2class = {i: c for i, c in enumerate(
            annotations['label_name'].unique())}
        annotations['labels'] = annotations['label_name'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()
        # split the train data into train and val #but ignore test

        self.n_classes = len(annotations['label_name'].unique())

        return annotations, idx2class

    def get_vocalset_technique_annotations(self):
        data_dir = '/import/c4dm-datasets/VocalSet1-2'
        train_singers = pd.read_csv(os.path.join(
            data_dir, 'train_singers_technique.txt'), header=None)
        test_singers = pd.read_csv(os.path.join(
            data_dir, 'test_singers_technique.txt'), header=None)
        train_singers.columns = ['id']
        test_singers.columns = ['id']

        train_singers['split'] = 'train'
        test_singers['split'] = 'test'

        id_to_split = pd.concat([train_singers, test_singers])

        id_to_split['id'] = id_to_split['id'].apply(
            lambda x: x.replace('emale', '').replace('ale', ''))

        # annotations = pd.DataFrame(columns=['file_path', 'labels', 'split'])
        annotations = []

        for root, dirs, files in os.walk(os.path.join(data_dir, 'data_by_technique')):
            for file in files:
                if file.endswith(".wav"):
                    singer_id = file.split('_')[0]
                    technique = root.split('/')[-1]
                    if singer_id in id_to_split.id.unique():
                        file_path = os.path.join(root, file)

                        split = id_to_split[id_to_split['id']
                                            == singer_id]['split'].values[0]

                        annotations.append(
                            [singer_id, file_path, technique, split])

        annotations = pd.DataFrame(annotations, columns=[
                                   'singer_id', 'file_path', 'technique', 'split'])

        annotations['label_name'] = annotations["technique"]

        class2idx = {c: i for i, c in enumerate(
            annotations['label_name'].unique())}
        idx2class = {i: c for i, c in enumerate(
            annotations['label_name'].unique())}
        annotations['labels'] = annotations['label_name'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()
        # split the train data into train and val #but ignore test

        if self.val_split > 0:
            test_data = annotations[annotations['split'] == 'test']
            train_data = annotations[annotations['split'] == 'train']
            val_data = annotations[annotations['split']
                                   == 'train'].sample(frac=self.val_split)
            train_data = train_data.drop(val_data.index)
            val_data['split'] = 'val'
            annotations = pd.concat([train_data, val_data, test_data])

        self.n_classes = len(annotations['label_name'].unique())

        return annotations, idx2class

    def get_mtg_annotations(self, path, audio_path):

        annotations = []

        class2idx = {}
        for split in ['train', 'validation', 'test']:
            data = open(path.replace(
                "split.tsv", f"{split}.tsv"), "r").readlines()
            all_paths = [line.split('\t')[3] for line in data[1:]]
            all_tags = [line[:-1].split('\t')[5:] for line in data[1:]]
            annotations.append(pd.DataFrame(
                {"file_path": all_paths, "tags": all_tags, "split": split}))
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2idx:
                        class2idx[tag] = len(class2idx)

        idx2class = {i: c for i, c in enumerate(class2idx.keys())}
        annotations = pd.concat(annotations)

        # replace mp3 extensions with wav in path columns
        annotations["split"] = annotations["split"].str.replace(
            "validation", "val")

        self.n_classes = len(class2idx)

        # the "labels" column is a list of tags, we need to one-hot encode it
        annotations['idx'] = annotations['tags'].apply(
            lambda x: [class2idx[tag] for tag in x])
        # now "labels" is a list of indices, we need to one-hot encode it into one on-hot vector per example
        annotations['labels'] = annotations['idx'].apply(lambda x: np.sum(
            np.eye(len(class2idx))[x], axis=0).astype(int).tolist())
        annotations['file_path'] = audio_path + '/' + annotations['file_path']

        return annotations, idx2class

    def get_mtg_top50_annotations(self):

        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_top50tags-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"

        return self.get_mtg_annotations(path, audio_path)

    def get_mtg_instr_annotations(self):

        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_instrument-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"

        return self.get_mtg_annotations(path, audio_path)

    def get_mtg_genre_annotations(self):

        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_genre-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"

        return self.get_mtg_annotations(path, audio_path)

    def get_mtg_mood_annotations(self):

        path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-split.tsv"
        audio_path = "/import/c4dm-datasets/mtg-jamendo-raw/mtg-jamendo-dataset/mp3"

        return self.get_mtg_annotations(path, audio_path)

    def get_medleydb_annotations(self):
        annotations, _ = self.get_medleydb_both_annotations()

        train, test = train_test_split(
            annotations, test_size=self.val_split, stratify=annotations['stem_instrument'])
        train, val = train_test_split(
            train, test_size=self.val_split, stratify=train['stem_instrument'])

        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        annotations = pd.concat([train, val, test])

        annotations['label_name'] = annotations["stem_instrument"]

        idx2class = {i: c for i, c in enumerate(
            annotations['stem_instrument'].unique())}
        class2idx = {c: i for i, c in enumerate(
            annotations['stem_instrument'].unique())}

        annotations['labels'] = annotations['label_name'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()

        self.n_classes = len(annotations['label_name'].unique())

        return annotations, idx2class

    def get_medleydb_raw_annotations(self):

        _, annotations = self.get_medleydb_both_annotations()

        train, test = train_test_split(
            annotations, test_size=self.val_split, stratify=annotations['raw_instrument'])
        train, val = train_test_split(
            train, test_size=self.val_split, stratify=train['raw_instrument'])

        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        annotations = pd.concat([train, val, test])

        annotations['label_name'] = annotations["raw_instrument"]

        idx2class = {i: c for i, c in enumerate(
            annotations['raw_instrument'].unique())}
        class2idx = {c: i for i, c in enumerate(
            annotations['raw_instrument'].unique())}

        annotations['labels'] = annotations['label_name'].apply(
            lambda x: class2idx[x])
        annotations['labels'] = pd.get_dummies(
            annotations['labels']).values.astype(int).tolist()

        self.n_classes = len(annotations['label_name'].unique())

        return annotations, idx2class

    def get_medleydb_both_annotations(self):

        medleydb_path = "data/medleydb"
        medleydb_audio_path = '/import/c4dm-datasets/MedleyDB_V1/V1'
        medleydb_audio_path_v2 = '/import/c4dm-datasets/MedleyDB_V2/V2'

        # get all the folder names in both audio paths with root
        all_paths_1 = []
        all_folders_1 = []
        all_paths_2 = []
        all_folders_2 = []
        for root, dirs, files in os.walk(medleydb_audio_path):
            for dir_ in dirs:
                all_folders_1.append(os.path.join(root, dir_))
                all_paths_1.append(dir_)
        for root, dirs, files in os.walk(medleydb_audio_path_v2):
            for dir_ in dirs:
                all_folders_2.append(os.path.join(root, dir_))
                all_paths_2.append(dir_)

        rows = []

        for song in os.listdir(medleydb_path):
            song_path = os.path.join(medleydb_path, song)
            yaml_dict = yaml.load(open(song_path), Loader=yaml.FullLoader)
            for stem in yaml_dict['stems']:
                stem_instrument = yaml_dict['stems'][stem]['instrument']
                row = {'song_name': song.split('_METADATA.yaml')[
                    0], 'filename': yaml_dict['stems'][stem]['filename'], 'filedir': yaml_dict['stem_dir'], 'stem_instrument': stem_instrument}

                rows.append(row)

        df = pd.DataFrame(rows)

        raw_rows = []

        for song in os.listdir(medleydb_path):
            song_path = os.path.join(medleydb_path, song)
            yaml_dict = yaml.load(open(song_path), Loader=yaml.FullLoader)
            for stem in yaml_dict['stems']:

                stem_instrument = yaml_dict['stems'][stem]['instrument']
                stem_dict = yaml_dict['stems'][stem]
                for raw in stem_dict['raw']:
                    raw_dict = stem_dict['raw'][raw]
                    raw_instrument = raw_dict['instrument']
                    raw_row = {'song_name': song.split('_METADATA.yaml')[
                        0], 'filename': stem_dict['raw'][raw]['filename'], 'filedir': yaml_dict['raw_dir'], 'raw_instrument': raw_instrument}
                    raw_rows.append(raw_row)

        raw_df = pd.DataFrame(raw_rows)

        existing_paths = pd.Series(all_paths_1 + all_paths_2)
        existing_folders = pd.Series(all_folders_1 + all_folders_2)

        paths2folders = pd.DataFrame(
            {'path': existing_paths, 'folder': existing_folders})

        # melt both dataframes : when the instruments are lists we can use the explode function to create a row for each instrument

        df = df.explode('stem_instrument')
        raw_df = raw_df.explode('raw_instrument')

        to_filter = ['fx/processed_sound', 'Unlabeled',
                     'auxiliary percussion', 'Main System']

        df = df[~df['stem_instrument'].isin(to_filter)]
        raw_df = raw_df[~raw_df['raw_instrument'].isin(to_filter)]

        # keep top 20 instruments for df and top 40 for raw_df

        df = df[df['stem_instrument'].isin(
            df['stem_instrument'].value_counts().head(20).index)]
        raw_df = raw_df[raw_df['raw_instrument'].isin(
            raw_df['raw_instrument'].value_counts().head(40).index)]

        df = pd.merge(df, paths2folders, left_on='song_name',
                      right_on='path', how='inner')
        df['file_path'] = df['folder'] + '/' + \
            df['filedir'] + '/' + df['filename']
        raw_df = pd.merge(raw_df, paths2folders,
                          left_on='song_name', right_on='path', how='inner')

        raw_df['file_path'] = raw_df['folder'] + '/' + \
            raw_df['filedir'] + '/' + raw_df['filename']

        return df, raw_df
