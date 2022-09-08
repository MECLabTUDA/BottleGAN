# System
from cProfile import label
import json
import random
import yaml
import os

# Datasets
from datasets.wsi_data import MultiAugSingleImg, MultiStainMultiFile, PairedDatasetLT, PairedDatasetST,  MultiFileMultiRegion, read_region_list
from torchvision import transforms
from single.data import PathologyAugmentations
from functools import partial

# Models
from modules.embedding import StarGAN, UNet,  BottleGAN, StainGAN
from modules.predictors import MultiArgsIdentity, TemperatureScaledPredictor
from single.models import CombinedNet
import torch.nn as nn

# Supervisor
from single.supervisors import  BottleSupervisor, StarSupervisor, SimPseudoSupervisor
from evaluation.stain_evaluation import StainEvaluator
from evaluation.single_evaluation import Evaluator

# Utils
from single.utils import bcolors, FocalLoss
import torch
from evaluation.single_evaluation import accuracy, ECELoss, iou, nll, tace, class_balance

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
# General config
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_config(embedding_path, predictor_path, task_path, data_path=None, aug_path=None, verbose=0):
    # Load configuration from file itself
    with open(embedding_path, 'r') as f:
        cfg = yaml.safe_load(f)

    with open(predictor_path, 'r') as f:
        update_default(cfg, yaml.safe_load(f))

    with open(task_path, 'r') as f:
        update_default(cfg, yaml.safe_load(f))
    
    if data_path is not None:
       with open(data_path, 'r') as f:
        update_default(cfg, yaml.safe_load(f))     
    
    if aug_path is not None:
       with open(aug_path, 'r') as f:
        update_default(cfg, yaml.safe_load(f))     

    return cfg

    
def update_default(dict_special, dict_default):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary containing default entries
    '''
    for k, v in dict_default.items():
        if isinstance(v, dict):
            if k not in dict_special:
                dict_special[k] = dict()
            update_default(dict_special[k], v)
        else:
            dict_special[k] = v

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Getter
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_dataset(cfg, verbose=0):
    image_size = cfg['embedding_architecture']['image_size']
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    length = cfg['dataset']['length']
    region_list = cfg['dataset']['region_list']
    n_files = cfg['dataset']['n_files']
    n_regions_per_file = cfg['dataset']['n_regions_per_file']
    img_files, label_files, regions, all_img_files = read_region_list(region_list, 
                                                        n_files=n_files,
                                                        n_regions_per_file=n_regions_per_file,
                                                        staining = None if 'staining' not in cfg['dataset'].keys() else cfg['dataset']['staining'])
    dataset2 = MultiFileMultiRegion(img_files=img_files, label_files=label_files, regions=regions, length=length, transform=trans, h=image_size, w=image_size, verbose=verbose)

    if cfg['dataset']['type'] == 'train' or  cfg['dataset']['type'] == 'val':
        dataset = PairedDatasetLT(dataset1=None, dataset2=dataset2, 
                            aug2=partial(PathologyAugmentations, cfg=cfg['weak_augmentations']))
    
    elif cfg['dataset']['type'] == 'test':
        dataset = MultiAugSingleImg(dataset2, aug=partial(PathologyAugmentations, cfg=cfg['weak_augmentations']), n_augs=cfg['task']['n_augs'], return_wsi_name=True)
        
    elif cfg['dataset']['type'] == 'semi':
        img_files = all_img_files
        label_files = None
        regions = None
        dataset1 = MultiFileMultiRegion(img_files=img_files, label_files=label_files, regions=regions, length=length,transform=trans, h=image_size, w=image_size, verbose=verbose, context=random.Random(42))
        dataset = PairedDatasetLT(dataset1=dataset1, dataset2=dataset2, aug2=partial(PathologyAugmentations, cfg=cfg['weak_augmentations']), 
                                                    aug1=partial(PathologyAugmentations, cfg=cfg['strong_augmentations']))
                                                    
    return dataset

def get_dataset_restain(cfg, verbose=0):
    image_size = cfg['embedding_architecture']['image_size']
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    length = cfg['dataset']['length']
    img_files_from = cfg['dataset']['staining_schemes_from']
    img_files_to = cfg['dataset']['staining_schemes_to']

    n_examples = 1
    if 'n_examples' in cfg['dataset'].keys():
        n_examples = cfg['dataset']['n_examples']

    if 'type' in cfg['dataset'].keys() and cfg['dataset']['type'] == 'test':
        dataset = MultiFileMultiRegion(img_files=[file[0] for file in img_files_from], clean_img_files=None, label_files=None, 
            regions=None, length=length, transform=trans, h=image_size, w=image_size, verbose=verbose)
    else:
        dataset1 = MultiStainMultiFile(img_files=img_files_from, label_files=None, n_examples=n_examples, length=length, transform=trans, h=image_size, w=image_size, verbose=verbose, min_non_white=0.3,
        augs=partial(PathologyAugmentations, cfg=cfg['weak_augmentations']))

        dataset2 = MultiStainMultiFile(img_files=img_files_to, label_files=None, n_examples=n_examples, length=length, transform=trans, h=image_size, w=image_size, verbose=verbose, min_non_white=0.3,
        augs=partial(PathologyAugmentations, cfg=cfg['weak_augmentations']))
        dataset = PairedDatasetST(dataset1=dataset1, dataset2=dataset2)

    return dataset

def get_embedding(cfg):
    arch = cfg['embedding_architecture']['name']

    if arch == 'unet':
        model = UNet(3, cfg['embedding_architecture']['n_classes'])

    elif arch == 'bottlegan':
        model = BottleGAN(star_size=cfg['embedding_architecture']['star_size'], 
                 noisy=cfg['embedding_architecture']['noisy'])

    elif arch == 'stargan':
        many_to_one = cfg['embedding_architecture']['many_to_one'] if 'many_to_one' in cfg['embedding_architecture'].keys() else False
        model = StarGAN(star_size=cfg['embedding_architecture']['star_size'], many_to_one=many_to_one)

    return model

def get_predictor(cfg):
    arch = cfg['predictor_architecture']['name']

    if arch == 'identity':
        model = nn.Identity()
    elif arch == 'multi_args_identity':
        model = MultiArgsIdentity()
    elif arch == 'temperature':
        model = TemperatureScaledPredictor()
    return model

def get_supervisor(cfg, dataset, embedding, predictor):
    arch = cfg['supervisor']['name']
    if arch == 'stain_evaluation':
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
            embeddings = stain_normalizer.backbone.label_embeddings.cpu()
            num_embeddings = embeddings.num_embeddings
            embeddings = [embeddings]
        
        elif isinstance(stain_normalizer.backbone, StarGAN):
            stain_normalizer_G = nn.DataParallel(stain_normalizer.backbone.G_S1_S2.to('cuda'))
            embeddings = [stain_normalizer.backbone.label_embeddings.cpu()]
            num_embeddings = embeddings[0].num_embeddings            

        stain_normalizer = {'type': 'bottle', 'G': stain_normalizer_G, 'G2':nn.DataParallel(stain_normalizer.backbone.G_S2_S1.to('cuda')), 'embeddings': embeddings, 'num_embeddings': num_embeddings, 'classifier': classifier}

        supervisor = StainEvaluator(stain_normalizer, result_file=cfg['result_file'])

    elif arch == 'stain_bottle_transfer':
        classifier = None
        if 'classifier_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['classifier_path'])
            classifier = torch.load(store_name)
            print('Loaded', store_name, type(classifier))
            classifier = nn.DataParallel(classifier.to('cuda'))  

        bb = CombinedNet(backbone=embedding, predictor=predictor)
        supervisor = BottleSupervisor(bb,
                                    dataset,
                                    {'cycle_loss': nn.L1Loss(reduction='mean'), 'disc_loss': torch.mean}, classifier=classifier).to('cuda')        

    elif arch == 'stain_star_transfer':
        bb = CombinedNet(backbone=embedding, predictor=predictor)
        supervisor = StarSupervisor(bb,
                                    dataset,
                                    {'cycle_loss': nn.L1Loss(reduction='mean'), 'disc_loss': torch.mean}).to('cuda')  

    elif arch == 'sim_semi_segmentation':
        bb = CombinedNet(backbone=embedding, predictor=predictor)

        # Check for cutmix
        cutmix = None
        if 'strong_augmentations' in cfg.keys():
            if 'cutmix' in cfg['strong_augmentations']['augmentations'].keys():
                cutmix = cfg['strong_augmentations']['augmentations']['cutmix']

        localization = None
        if 'localization' in cfg['supervisor'].keys():
            localization = cfg['supervisor']['localization']

        stain_restainer = None
        if 'stain_restainer_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['stain_restainer_path'])
            stain_restainer = torch.load(store_name)
            print(bcolors.OKBLUE + "Loaded", store_name+ "." + bcolors.ENDC)

            if isinstance(stain_restainer.backbone, BottleGAN):
                stain_restainer_G = nn.DataParallel(stain_restainer.backbone.G_S2_S1.to('cuda'))
                embeddings =  stain_restainer.backbone.label_embeddings.cpu()
                num_embeddings = embeddings.num_embeddings

                embeddings = [embeddings]

                stain_restainer = {'type': 'bottle', 'G': stain_restainer_G, 'embeddings': embeddings, 'num_embeddings': num_embeddings}

        stain_normalizer = None
        if 'stain_normalizer_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['stain_normalizer_path'])
            stain_normalizer = torch.load(store_name)
            print(bcolors.OKBLUE + "Loaded", store_name+ "." + bcolors.ENDC)

            if isinstance(stain_normalizer.backbone, StainGAN):
                stain_normalizer_G = nn.DataParallel(stain_normalizer.backbone.G_S1_S2.to('cuda'))
                embeddings = stain_normalizer.backbone.label_embeddings.cpu()
                stain_normalizer = {'type': 'gan', 'G': stain_normalizer_G, 'embeddings': embeddings}

        style_noise = 0
        if 'style_noise' in cfg['supervisor'].keys():
            style_noise = cfg['supervisor']['style_noise']

        server_path = None
        if 'server_path' in cfg['supervisor'].keys():
            server_path = cfg['supervisor']['server_path'] if cfg['supervisor']['server_path']  != 'None' else None

        ssl_type = 'pseudo'
        if 'ssl_type' in cfg['supervisor'].keys():
            ssl_type = cfg['supervisor']['ssl_type']

        lambda_cl = 0.1
        if 'lambda_cl' in cfg['supervisor'].keys():
            lambda_cl = cfg['supervisor']['lambda_cl']


        slow_only = False
        if 'slow_only' in cfg['supervisor'].keys():
            slow_only = cfg['supervisor']['slow_only']
            
        fast_only = False
        if 'fast_only' in cfg['supervisor'].keys():
            fast_only = cfg['supervisor']['fast_only']

        supervisor = SimPseudoSupervisor(bb, 
                                dataset, 
                                FocalLoss(),
                                t=cfg['supervisor']['t'],
                                T=cfg['supervisor']['T'],
                                soft=cfg['supervisor']['soft'],
                                cutmix=cutmix,
                                stain_normalizer=stain_normalizer,
                                stain_restainer=stain_restainer,
                                localization=localization,
                                style_noise=style_noise,
                                server_path=server_path,
                                ssl_type=ssl_type,
                                lambda_cl=lambda_cl,
                                slow_only=slow_only,
                                fast_only=fast_only).to('cuda')

    elif arch == 'super_evaluation':

        store_name = 'store/' + str(cfg['task']['id'])
        model = torch.load(store_name + '_inc_meta.pt')
        print(bcolors.OKBLUE + "Loaded", store_name + "_inc_meta." + bcolors.ENDC)

        metrics = cfg['metrics']
        metrics = [(m, metrics_map[m]) for m in metrics]
        try:
            model.backbone.inference = True
            print('BE activated.')
        except:
            print('No particular inference mode.')

        stain_normalizer = None
        if 'stain_normalizer_path' in cfg['supervisor'].keys():
            store_name = str(cfg['supervisor']['stain_normalizer_path'])
            stain_normalizer = torch.load(store_name)
            print(bcolors.OKBLUE + "Loaded", store_name+ "." + bcolors.ENDC)

            if isinstance(stain_normalizer.backbone, BottleGAN):
                stain_normalizer_G = nn.DataParallel(stain_normalizer.backbone.G_S2_S1.to('cuda'))
                embeddings = stain_normalizer.backbone.label_embeddings_2_1.cpu() if stain_normalizer.backbone.split_embeddings else stain_normalizer.backbone.label_embeddings.cpu()
                num_embeddings = embeddings.num_embeddings

                embeddings = [embeddings, stain_normalizer.backbone.ms, stain_normalizer.backbone.c_ratio]

                stain_normalizer = {'type': 'bottle', 'G': stain_normalizer_G, 'embeddings': embeddings, 'num_embeddings': num_embeddings}

        supervisor = Evaluator(metrics, nn.DataParallel(model.to('cuda')), cfg['embedding_architecture']['n_classes'], result_file=cfg['result_file'], stain_normalizer=stain_normalizer)


    return supervisor


