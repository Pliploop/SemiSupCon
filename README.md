# Semi-Supervised Contrastive Learning of Musical Representations

Official Pytorch implementation of the paper [Semi-Supervised Contrastive Learning of Musical Representations](), By [Julien Guinot](), [Elio Quinton]() and [Gyorgy Fazekas](). Accepted and to be published at [ISMIR 2024]() in San Francisco.




![SemiSupCon](https://github.com/spijkervet/clmr/actions/workflows/clmr.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2103.09410-b31b1b.svg)](https://arxiv.org/abs/2103.09410)


In this work, we introduce a novel framework for semi-supervised contrastive learning using labels and minimal amounts of labeled data. This flexible framework enables using any supervision signal (binary labels, multiclass labels, multilabel labels, regression targets, domain-knowledge similarity metric) to guide contrastive learning to learn musically-informed representations.

Briefly, the contributions of this work are the following:

-  We propose an architecturally simple extension of self-supervised and supervised contrastive learning to the semi-supervised case with the ability to make use of a variety of supervision signals. 
- We show the ability of our method to shape the representations according to the support supervision signal used for the learning task with minimal performance loss on other tasks.
-  We propose a representation learning framework with low-data regime potential and higher robustness to data corruption.

<div align="center">
  <img width="50%" alt="CLMR model" src="https://github.com/Pliploop/SemiSupCon/blob/master/media/SMSL_Horizontal.png?raw=true">
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

provide a boilerplate template for training