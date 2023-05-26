# BottleGAN
This repo contains an implementation the pure BottleGAN of the 2022 MICCAI paper *Federated Stain Normalization for Computational Pathology*. \
This contains non-federated stain normalization of BottleGAN and should be easier to deploy in differen contexts.

## Installation
- git clone https://github.com/MECLabTUDA/BottleGAN.git
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
- scikit-learn

## Dataset
You can deploy the BottleGAN with any dataset providing a WSI patch as return of getitem().\
For the MultiStainMultiFile dataset a list of TIF files is required.

## Non-Federated BottleGAN
### Training
Non-Federated Stain Normalization BottleGAN with 240 artificial staining styls can be run through

```lean_bottle_gan.sh *id*```

the results are written to
experiments/stain/results/bottle_gan/*dataset*. \
Further a T-SNE overview of the normalization results can be found in the same folder.\
The training progress can be monitored in imgs/stain_tf_bottle.jpg.

### Parameters & Configurations
Examples of configurations files can be found in:

```lean_BottleGAN_config_test.yaml``` and ```lean_BottleGAN_config_train.yaml```