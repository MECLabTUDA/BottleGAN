# System
import argparse
import json
import random
import os
import yaml

# PyTorch
import torch
import torch.nn as nn

# Models
from models import BottleGAN, CombinedNet, MultiArgsIdentity, TemperatureScaledPredictor

# Supervisors
from supervisors import  BottleSupervisor, StainEvaluator

# Dataset
from torchvision import transforms
from torch.utils.data import Dataset
from dataset import PairedDatasetST

# Utils
from utils import bcolors, FocalLoss, accuracy, ECELoss, iou, nll, tace, class_balance

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dicts 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
metrics_map = {'acc': accuracy,
               'ece': ECELoss(),
               'iou': iou,
               'nll': nll,
               'tac': tace,
               'cbl': class_balance}


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Dummy Dataset
# How MultiStainMultiFile() is used in get_dataset_restain() function in single_configs.py
# This will give you an idea how to do WSI loading.
# However you can use any other Dataset you used for normal training.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class WSIDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        wsi_path = self.data_paths[idx]
        
        # Load the WSI data here (replace with your own loading logic)
        wsi_data = torch.zeros((3,114,114)) # load_wsi(wsi_path) 
        
        return wsi_data
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Run Training
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run(cfg, verbose=0):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    image_size = cfg['embedding_architecture']['image_size']
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    length = cfg['dataset']['length']
    img_files_from = cfg['dataset']['staining_schemes_from']
    img_files_to = cfg['dataset']['staining_schemes_to']


    if 'type' in cfg['dataset'].keys() and cfg['dataset']['type'] == 'test':
        dataset = WSIDataset(["List of paths to test dataset"])
    elif cfg['dataset']['type'] == 'train':
        dataset1 = WSIDataset(img_files_from)
        dataset2 = WSIDataset(img_files_to)
        dataset = PairedDatasetST(dataset1=dataset1, dataset2=dataset2)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Embedding and Predictor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    emb = BottleGAN(star_size=cfg['embedding_architecture']['star_size'], noisy=cfg['embedding_architecture']['noisy'])
    pred = nn.Identity()


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Supervisor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Supervisor for Test evaluation
    if cfg['supervisor']['name'] == 'stain_evaluation':
        store_name = 'store/' + str(cfg['task']['id']) + '_inc_meta.pt'
        stain_normalizer = torch.load(store_name)
        print(bcolors.OKBLUE + "Loaded", store_name+ "." + bcolors.ENDC)

        if 'classifier_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['classifier_path'])
            classifier = torch.load(store_name)
            print('Loaded', store_name, type(classifier.backbone))
            classifier = nn.DataParallel(classifier.to('cuda')) 

        if isinstance(stain_normalizer.backbone, BottleGAN):
            stain_normalizer_G = nn.DataParallel(stain_normalizer.backbone.G_S1_S2.to('cuda'))
            emb = stain_normalizer.backbone.label_emb.cpu()
            num_emb= emb.num_emb
            emb= [emb]

        stain_normalizer = {'type': 'bottle', 'G': stain_normalizer_G, 'G2':nn.DataParallel(stain_normalizer.backbone.G_S2_S1.to('cuda')), 'embeddings': emb, 'num_embeddings': num_emb, 'classifier': classifier}

        supervisor = StainEvaluator(stain_normalizer, result_file=cfg['result_file'])
    # Supervisor for Training
    elif cfg['supervisor']['name'] == 'stain_bottle_transfer':
        classifier = None
        if 'classifier_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['classifier_path'])
            classifier = torch.load(store_name)
            print('Loaded', store_name, type(classifier))
            classifier = nn.DataParallel(classifier.to('cuda'))  

        bb = CombinedNet(backbone=emb, predictor=pred)
        supervisor = BottleSupervisor(bb,
                                    dataset,
                                    {'cycle_loss': nn.L1Loss(reduction='mean'), 'disc_loss': torch.mean}, classifier=classifier).to('cuda')  
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Perform experiment
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    arch = cfg['task']['name']
    id = cfg['task']['id']
    store_name = 'store/' + str(id)
    
    if arch == 'train':
        lr = cfg['task']['lr']
        epochs = cfg['task']['epochs']
        batch_size = cfg['task']['batch_size']
        pretrained = cfg['task']['pretrained']
        pretrained_opt = cfg['task']['pretrained_opt']
        num_workers = cfg['task']['num_workers']

        try:
            supervisor.supervise(lr=lr, epochs=epochs,
                            batch_size=batch_size, name=store_name, pretrained=pretrained, pretrained_opt=pretrained_opt, num_workers=num_workers)
        finally:
            pass

    elif arch == 'test':
        supervisor.evaluate(dataset, cfg['task']['batch_size'], cfg['task']['num_workers'] )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')  
    parser.add_argument('--staining_files_from', default=None, type=json.loads, help='Staining scheme.') 
    parser.add_argument('--staining_files_to', default=None,  type=json.loads, help='Staining scheme.')    
    parser.add_argument('--result_file', type=str, default=None)   
    parser.add_argument('--deep', type=str, default=None, help='Cycle vs Bottle')
    parser.add_argument('--config', type=str, default='lean_BottleGAN_config_train.yaml', help='Path to config file.')
    parser.add_argument('--classifier_path', type=str, default=None, help='Path to stored classifier.') 
    parser.add_argument('--id', type=str, help='ID of Experiment.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='Set seed for repoducability.')

    args = parser.parse_args()

    if args.seed is not None:
        import torch
        import numpy as np
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.config is not None:
        cfg = yaml.safe_load(args.config)
        cfg['result_file'] = args.result_file
        cfg['task']['id'] = args.id
    if args.classifier_path is not None:
        cfg['supervisor']['classifier_path'] = args.classifier_path

    if args.deep == 'bottle' or args.deep == 'star':
        if 'dataset' not in cfg.keys():
            cfg['dataset'] = {}
        cfg['dataset']['length'] = 180000 if cfg['task']['name'] == 'train' else 10000
        cfg['dataset']['staining_schemes_from'] = args.staining_files_from
        cfg['dataset']['staining_schemes_to'] = args.staining_files_to
        cfg['embedding_architecture']['star_size'] = len(args.staining_files_from)
    run(cfg)
