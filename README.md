# BottleGAN
This repo contains an implementation of the 2022 MICCAI paper *Federated Stain Normalization for Computational Pathology*.</br>
The federated simulation is intended to be flexible for a simplified application to a real-world federated system.</br>
Further, a non-federated simulation of BottleGAN can be conducted.

## Introduction
BottleGAN provides a Generative Adversarial Networks (GAN) which allows to align the individual staining styles of different clients in Federated Learning.</br>
This repository contains the code for BottleGAN and and different scripts for defining federated tasks, training and evaluation which will be explained in the following.

## Installation
- git clone https://github.com/nwWag/BottleGAN_prerelease.git
- conda create -n bottlegan python=3.9

Install the following dependencies that are all available with pip
- matplotlib
- torch
- torchvision
- scipy
- elasticdeform
- scikit-image
- einops
- staintools
- spams
- openslide-python
- zarr
- super_selfish
- tqdm
- colorama
- pyyaml
- piq
- sklearn

## Preparing PESO dataset
### 1. Downloading PESO dataset
The following data are required:
- Training WSIs (peso_training_wsi_[1-6].zip)
- Training masks (peso_training_masks.zip)
- Test WSIs (peso_testset_wsi_[1-4].zip)</br>

Create the folder ```../patho_data/peso```  and extract the aforementioned zips here. The folder should contain only the files and no subfolders.

### 2. Downsampling
In order to reduce the computational load it's recommended to apply downsampling to the WSIs and the corresponding labels by running 

```helper/restain_down.sh```

First replace the activated environment (ENVNAME) with yours and insert the FILEIDS of the WSIs that should be downsampled.
All scripts are assuming that the data are stored in ```../patho_data/peso```. If this is not the case for you then please adjust the paths in this script.

### 3. Staining
As we want to evaluate how well the network can handle different staining styles among the clients, we want to apply different staining styles to the WSIs.
With 

```helper/restain_mancenko_peso.sh```

you can apply staining styles to the WSIs.
As for ```helper/restain_down.sh``` please replace the activated environment (ENVNAME) and if necessary adapt the paths. Insert the FILEIDS and SCHEMES in the same order to select the staining styles for the WSIs. This script will save the stained WSIs and the corresponding downsampled labels.

### 4. Data Plan and Distribution
A dataplan defines which client can access which training and testing data.
Run 

```helper/create_federation.sh```

to automatically construct a new data, label, and staining scheme distribution. See Section 4.1 in the paper for details.
## Federated BottleGAN
### Training
Federated BottleGAN with a downstream segmentation task can be trained through

```experiments/federated/scripts/fed_bottle.sh *dataset* *id*```

the results are written to
experiments/federated/results/fed_sim/*dataset*.
Naive FedAvgM can be trained through

```experiments/federated/scripts/fed_avgm.sh *dataset* *id*```.

One can plot the IOU, NLL, and ECE over the course of simulation by invoking

```plot_results.py```

in the *utils* folder. The results can be found in *imgs*.

## Non-Federated BottleGAN
### Training
Non-Federated BottleGAN with 240 artificial staining styls can be trained through

```experiments/stain/scripts/bottle_gan.sh *id*```

the results are written to
experiments/stain/results/bottle_gan/*dataset*. Further a T-SNE overview of the normalization results can be found in the same folder.
The training progress can be monitored in imgs/stain_tf_bottle.jpg.

## Using other datasets
By default the dataloader is assuming PESO as dataset. For other datasets please change the ```dataset_name``` in ```datasets/wsi_data.py```.
The following dataset_names are supported:
- peso
- luad
- bcss
- bach
