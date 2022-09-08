import os
from single.data import augment_staining
from single.restain import restain
from configs.single_configs import load_config, update_default, get_dataset_restain, get_embedding, get_predictor, get_supervisor
import argparse
import json
import random
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run(cfg, verbose=0):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dataset = get_dataset_restain(cfg, verbose)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Embedding and Predictor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    emb = get_embedding(cfg)
    pred = get_predictor(cfg)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Supervisor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    supervisor = get_supervisor(cfg, dataset=dataset, embedding=emb, predictor=pred)


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--org_file', type=str, default=None, help='File to restain.')    
    parser.add_argument('--staining_file', type=str, help='Staining scheme.') 
    parser.add_argument('--staining_files_from', default=None, type=json.loads, help='Staining scheme.') 
    parser.add_argument('--staining_files_to', default=None,  type=json.loads, help='Staining scheme.')    
    parser.add_argument('--org_label_file', type=str, default=None, help='Org.labels..')  
    parser.add_argument('--org_label_files', type=json.loads, default=None, help='Org.labels..')    
    parser.add_argument('--result_file', type=str, default=None)   

    parser.add_argument('--staining_dicts', type=json.loads, default=None, help='Staining dicts.') 

    parser.add_argument('--deep', type=str, default=None, help='Cycle vs Bottle')
    parser.add_argument('--config_embedding', type=str, default='configs/embeddings/cycle_gan.yaml', help='Path to embedding config file.')
    parser.add_argument('--classifier_path', type=str, default=None, help='Path to stored classifier.') 
    parser.add_argument('--config_predictor', type=str, default='configs/predictors/identity.yaml', help='Path to predictor config file.')
    parser.add_argument('--config_task', type=str, default='configs/tasks/train/single/stain_transfer.yaml', help='Path to task config file.')
    parser.add_argument('--config_augmentations', type=str, default='configs/augmentations/off.yaml', help='Path to augmentation config file.')
    parser.add_argument('--down_factor', type=int, default=4)
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


    if args.deep is not None:
        cfg = load_config(args.config_embedding, args.config_predictor, args.config_task, aug_path=args.config_augmentations)
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

    else:
        restain(args.org_file, args.staining_file, args.org_label_file, deep=False, down_factor=args.down_factor, verbose=0, only_labels=True)