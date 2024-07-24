
<div  align="center">

# Semi-Supervised Contrastive Learning of Musical Representations
[Julien Guinot](https://julienguinot.com/)<sup>1,2</sup>,
[Elio Quinton](https://scholar.google.com/citations?user=IaciybgAAAAJ)<sup>2</sup>,
[Gyorgy Fazekas](http://www.eecs.qmul.ac.uk/~gyorgyf/about.html)<sup>1</sup> <br>
<sup>1</sup>  Queen Mary University of London, <sup>2</sup>  Universal Music Group

[![arXiv](https://img.shields.io/badge/arXiv-2208.12208-<COLOR>.svg)](https://arxiv.org/abs/2407.13840)

</div>

Official Pytorch implementation of the paper [Semi-Supervised Contrastive Learning of Musical Representations](), By [Julien Guinot](), [Elio Quinton]() and [Gyorgy Fazekas](). Accepted and to be published at [ISMIR 2024]() in San Francisco.

***

In this work, we introduce a novel framework for semi-supervised contrastive learning using labels and minimal amounts of labeled data. This flexible framework enables using any supervision signal (binary labels, multiclass labels, multilabel labels, regression targets, domain-knowledge similarity metric) to guide contrastive learning to learn musically-informed representations.

Briefly, the contributions of this work are the following:

-  We propose an architecturally simple extension of self-supervised and supervised contrastive learning to the semi-supervised case with the ability to make use of a variety of supervision signals. 
- We show the ability of our method to shape the representations according to the support supervision signal used for the learning task with minimal performance loss on other tasks.
-  We propose a representation learning framework with low-data regime potential and higher robustness to data corruption.

<div align="center">
  <img width="100%" alt="SemiSupCon" src="https://github.com/Pliploop/SemiSupCon/blob/main/media/SMSL_Horizontal.png?raw=true">
</div>
<div align="center">
  An illustration of SemiSupCon. Labels augment a target contrastive matrix which guides the learned representation towards a target similarity metric.
</div>

## Setup

We recommand setting up a new environment (using ```conda``` or ```venv```) before installing requirements for this project. This project using pytorch-lightning for training and probing experiments. We are currently working on a minimal requirements file and package only for loading models and running inference.

```
git clone https://github.com/Pliploop/SemiSupCon.git && cd SemiSupCon

pip install requirements.txt
```

<!-- TODO : add requirements.txt -->

We do not *yet* provide scripts for downloading datasets, but the following section provides a comprehensive guide on how to implement your own dataset or re-use the currently implemented datasets, namely:

- MagnaTagATune (all tags and top 50 tags)
- MTG Jamendo (top 50 tags, moodtheme split, genre split, instrumentation split)
- GTZAN
- Vocalset (singer identity or technique)
- Giantsteps key (as implemented in [MARBLE]())
- MedleyDB
- Nysnth (pitch class and instrument)
- EmoMusic

<!-- TODO : add links to each dataset -->

## Training your own SemiSupCon model

We provide a script ```pretrain.py``` to train your own SemiSupCon model in a supervised, semi-supervised or self-supervised fashion. training arguments are provided through pytorch lightning and LighningCLI, meaning config ```.yaml``` files are used to provide training arguments. all arguments can be overridden from the command line. The files 

- ```config/pretraining.pretrain_sl.yaml```
- ```config/pretraining.pretrain_smsl.yaml```
- ```config/pretraining.pretrain_ssl.yaml```

provide a boilerplate template for training. Disregarding details of dataset availability, a training run can be started with:

```
python pretrain.py --config config_file_path (--trainer.max_steps=100000 --data.data_dir=x --model.optimizer.lr=0.01)
```

### On dataloading and semi-supervised dataloading

when running an experiment, the config arguments `--data.ssl_task` and `--data.sl_task` are essential, and provide information on what data to use for self-supervised and supervised dataloading.

all `sl_tasks` must be named as their name will determine the dataloading logic used in the `dataloading/datamodule_splitter.py` file. naming an `ssl_task` is optional (e.g. a specific train_test_val split is required), but all named tasks must have a logic implemented into `SemiSupCon/dataloading/datamodule_splitter.py`.

If `None` is provided for `sl_task`, the training will be self-supervised. If `None` is provided for `ssl_task`, the datamodule will look to `--data.data_dir` for a data source folder and split it using `--data.val_split`. If none is provided, training will be fully-supervised. 

> If neither `sl_task`,`ssl_task`, nor `data_dir` are provided, the training will not be able to run.

Additional flags exist to toggle fully-supervised training even with an ssl directory or task provided. switch `--data.fully_supervised` to True to toggle this behaviour even when ssl data is provided:

```
python pretrain.py --config config_file_path --data.sl_task='mtat_top50' --data.fully_supervised=true
```


All logic for dealing with specific datasets is handled in the file `datamodule_splitter.py` file. If you wish to implement your own dataset for semi-supervised or fully-supervised training, a `get_{dataset}_annotations` method must be implemented which returns an annotations dataframe as well as a idx2class dictionary (optional). Please take inspiration from existing `get_{dataset}_annotations` methods to build your own. Once the method is implemented, kwargs for loading (e.g. file paths) can be passed through `--data.sl_kwargs` in the config file for flexibility. For existing datasets, these kwargs should be adapted from existing ones (defaults provided in the datamodule splitter).

### dealing with $p_s$ and $b_s$

[Our paper]() studies the influence of $p_s$ and $b_s$, respectively the proportion of the supervised dataset used for training and the proportion of labeled data in each batch. These parameters are easily controlled through `--data.supervised_data_p` and `--data.intrabatch_supervised_p`. note that $p_s = 0$ or $b_s = 0$ leads to fully-self-supervised training and $b_s = 1$ leads to fully-supervised training. By default, we suggest keeping $p_s$ as 1. $b_s$ is a hyperameter which we have found to have varying influence.

### logging

Logging was done using `wandb` for this project, and as such no other logger is implemented thus far. We invite different logger users to contribute by providing options to switch between loggers. Logging is turned off by default and can be activated with `--log=true`. The folder where model checkpoints are saved can also be modified. to resume a run, simply provide the path to the checkpoint to resume from:

```
python pretrain.py --config config_path --log=true --ckpt_path=myckpt --resume_from_ckpt=myckptpath
```

### augmentations and loading strategy

A plethora of augmentations are provided in `SemiSupCon/dataloading/custom_augmentations` and can be changed in the `--data.aug_list` argument. We recommend not modifying `--data.aug_severity` by default as it relates to one of the experiments in the paper.


### Custom encoders

By default, our approach uses [SampleCNN]() as the encoder. Simply implement your own encoder and change the `--model.encoder.class_path` and potential `--model.encoder.init_args` in the config file. Target length of loaded audio and sample rate for the encoder can be changed `--data.target_sr` and `--data.target_len

***

## Extracting representations / Finetuning

***

## Cite

***

## Contact

Please get in touch with any questions : [j.guinot@qmul.ac.uk](j.guinot@qmul.ac.uk)
