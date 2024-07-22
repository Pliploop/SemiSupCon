import pytorch_lightning as pl
from torch_audiomentations import *
from SemiSupCon.dataloading import *
from SemiSupCon.dataloading.custom_augmentations import *

        
    
class MixedDataModule(pl.LightningDataModule):
    
    """
    
    Mixed Data module class that returns a semi-supervised dataloader.
    The dataloader is a mix of supervised and self-supervised data.
    with proportion of supervised data in a batch determined by intrabatch_supervised_p.
    
    
    """
    
    def __init__(self,
                 data_dir = None, # data directory for ssl task : can be None if ssl_task is not None
                 sl_task = None, # supervised learning task, with logic in DatamoduleSplitter class
                 ssl_task = None, # self-supervised learning task, with logic in DatamoduleSplitter class. Can be None if data_dir is not None
                 target_length = 2.7, # in seconds
                 target_sample_rate = 22050, # sr to resample to when loading (on the fly)
                 n_augmentations= 2, # number of augmentations. 2 by default. Samples will be loaded as (batch, n_augmentations, target_samples)
                 transform = True,
                 batch_size = 32,
                 num_workers = 16,
                 val_split = 0.1, # if validation split is not determined in logic in DatamoduleSplitter class
                 test_split = 0, # if test set split is not determined in logic in DatamoduleSplitter class
                 use_test_set = False, # whether to use the test set or during training (leave False by default)
                 supervised_data_p = 1, # proportion of supervised training dataset to use for training
                 fully_supervised = False, # whether or not to use the fully supervised setting (and return labels)
                 intrabatch_supervised_p = 0.5, # proportion of supervised data in a batch
                 severity_modifier = 2, # scales from 0 to 5, only applies if augmentations are included. leave at 2 by default
                 test_transform = False,
                 aug_list = [],
                 sl_kwargs = {} # additional keyword arguments for supervised dataloading
                 ) -> None:
    
        
        super().__init__()
        
        self.data_dir = data_dir
        self.sl_task = sl_task
        self.ssl_task = ssl_task
        
        if self.data_dir is None and self.ssl_task is None:
            assert self.sl_task is not None, "Either data_dir or ssl_task must be provided if sl_task is not None"
        if self.sl_task is None:
            assert self.data_dir is not None or self.ssl_task is not None, "Either data_dir or ssl_task must be provided if sl_task is None"
        
        self.target_length = target_length
        self.target_sample_rate = target_sample_rate
        self.n_augmentations = n_augmentations
        self.target_samples = int(self.target_length * self.target_sample_rate)
        self.global_target_samples = self.target_samples * self.n_augmentations
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
        self.test_transform = test_transform
        
        self.severity_modifier = severity_modifier/2 # scale back to between 0 (unmodified) and 2.5 (max severity) with 1 being "normal severity"
        self.get_aug_chain()
        
        self.self_supervised_augmentations = self.supervised_augmentations # by default augmentations are the same but this can be changed
        
        if self.fully_supervised:
            self.supervised_dataset_percentage = 1
            self.in_batch_supervised_percentage = 1

        #Get annotations based on task
        #df with [filepath,label]
        #df also returns the split for each file path. 
        self.splitter = DataModuleSplitter(
            data_dir=self.data_dir,
            ssl_task=self.ssl_task,
            sl_task=self.sl_task,
            supervised_data_p=self.supervised_dataset_percentage,
            val_split=self.val_split,
            test_split=self.test_split,
            use_test_set=self.use_test_set,
            fully_supervised=self.fully_supervised,
            sl_kwargs=sl_kwargs
            ) # annotations are retrieved at initialization
        self.annotations = self.splitter.annotations
        self.n_classes = self.splitter.n_classes # n_classes are determined by logic in splitter, required for supervised dataloading
        if self.splitter.idx2class is not None:
            self.idx2class = self.splitter.idx2class #idx2class is determined by logic in splitter, required for supervised dataloading
            self.class_names = list(self.idx2class.values())
        else:
            self.idx2class = None
            self.class_names = None
        
        print(self.annotations.groupby('split').count())
        print("n_classes: ", self.n_classes)
        
    def get_aug_chain(self):
        
        if self.severity_modifier > 0: # if severity modifier is 0, no augmentations are applied
        
            self.augmentations = {
                'gain': lambda: Gain(min_gain_in_db=-15.0 * self.severity_modifier, max_gain_in_db=5.0 * self.severity_modifier, p=min(0.7,0.4* self.severity_modifier), sample_rate=self.target_sample_rate),
                'polarity_inversion': lambda: PolarityInversion(p=min(0.8,0.6* self.severity_modifier), sample_rate=self.target_sample_rate),
                'add_colored_noise': lambda: AddColoredNoise(p=min(0.8,0.6* self.severity_modifier), sample_rate=self.target_sample_rate, min_snr_in_db=3 / self.severity_modifier, max_snr_in_db=30 / self.severity_modifier, min_f_decay=-2 * self.severity_modifier, max_f_decay=2 * self.severity_modifier),
                'filtering': lambda: OneOf([
                    BandPassFilter(p=min(0.6,0.3* self.severity_modifier), sample_rate=self.target_sample_rate, min_center_frequency=200, max_center_frequency=4000, min_bandwidth_fraction=0.5 * self.severity_modifier, max_bandwidth_fraction=1.99 ),
                    BandStopFilter(p=min(0.6,0.3* self.severity_modifier), sample_rate=self.target_sample_rate, min_center_frequency=200 , max_center_frequency=4000 , min_bandwidth_fraction=0.5 * self.severity_modifier, max_bandwidth_fraction=1.99 ),
                    HighPassFilter(p=min(0.6,0.3* self.severity_modifier), sample_rate=self.target_sample_rate, min_cutoff_freq=200 * self.severity_modifier , max_cutoff_freq=min(0.5* self.target_sample_rate,2400 * self.severity_modifier)),
                    LowPassFilter(p=min(0.6,0.3* self.severity_modifier), sample_rate=self.target_sample_rate, min_cutoff_freq=max(75,150 / self.severity_modifier), max_cutoff_freq=7500 / max(1,self.severity_modifier) ),
                ]),
                'pitch_shift': lambda: PitchShift(p=min(0.75,0.6* self.severity_modifier), sample_rate=self.target_sample_rate, min_transpose_semitones=-4 * self.severity_modifier, max_transpose_semitones=4 * self.severity_modifier),
                'delay': lambda: Delay(p=min(0.6,0.6* self.severity_modifier), sample_rate=self.target_sample_rate, min_delay_ms=100 / self.severity_modifier, max_delay_ms=500, volume_factor=0.5 * self.severity_modifier, repeats=2 * self.severity_modifier, attenuation=min(1,0.5 * self.severity_modifier)),
                'timestretch': lambda: TimeStretchAudiomentation(p=1, sample_rate=self.target_sample_rate, min_stretch_rate=0.7, max_stretch_rate=1.3),
                'splice': lambda: SpliceOut(p=1, sample_rate=self.target_sample_rate, max_width=100),
                'reverb' : lambda: ReverbAudiomentation(p=1, sample_rate=self.target_sample_rate,room_size = 1, wet_level = 1, dry_level = 1),
                'chorus' : lambda: ChorusAudiomentation(p=1, sample_rate=self.target_sample_rate, mix = 1, rate_hz = 5, depth = 1),
                'distortion' : lambda: DistortionAudiomentation(p=1, sample_rate=self.target_sample_rate, drive_db = 9),
                'compression' : lambda: CompressorAudiomentation(p=1, sample_rate=self.target_sample_rate, threshold_db = -30, ratio = 5),
                'reverse' : lambda: Reverse(p=1, sample_rate=self.target_sample_rate),
                'bitcrush' : lambda: BitcrushAudiomentation(p=1, sample_rate=self.target_sample_rate, bit_depth = 4),
                'mp3' : lambda: MP3CompressorAudiomentation(p=1, sample_rate=self.target_sample_rate, vbr_quality = 9)
            }
            
            self.supervised_augmentations = Compose(
                [
                    self.augmentations[aug]() for aug in self.aug_list
                ],
                p=min(1,0.8 * self.severity_modifier),
            )
        
        else:
            self.augmentations = None
            self.supervised_augmentations = None
            
      
    def setup(self, stage = 'fit'):
        
        
        supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 1)]
        supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 1)]
        supervised_test_annotations = self.annotations[(self.annotations['split'] == 'test') & (self.annotations['supervised'] == 1)]
        self_supervised_train_annotations = self.annotations[(self.annotations['split'] == 'train') & (self.annotations['supervised'] == 0)]
        self_supervised_val_annotations = self.annotations[(self.annotations['split'] == 'val') & (self.annotations['supervised'] == 0)]
        
        train_supervised_dataset = SupervisedDataset(self.data_dir, supervised_train_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  True, self.n_classes, idx2class=self.idx2class)
        val_supervised_dataset = SupervisedDataset(self.data_dir, supervised_val_annotations, self.target_length, self.target_sample_rate, self.n_augmentations, self.transform, self.supervised_augmentations,  True, self.n_classes, idx2class=self.idx2class)
        test_supervised_dataset = SupervisedTestDataset(self.data_dir, supervised_test_annotations, self.target_length, self.target_sample_rate, 1, self.test_transform, self.supervised_augmentations,  False, self.n_classes, idx2class = self.idx2class)
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
