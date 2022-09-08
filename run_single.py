from single.utils import SchedulerSE
from configs.single_configs import load_config, get_dataset, get_embedding, get_predictor, get_supervisor
from utils.arg_reader import StoreDictKeyPair
import argparse
import yaml
from functools import partial
from torch import optim
import random
import torch

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run(cfg, verbose=0, onlyfast=False, multiheads=False, onlyslow=False):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dataset = get_dataset(cfg, verbose)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Embedding and Predictor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    emb = get_embedding(cfg)
    pred = get_predictor(cfg)
    
    if onlyfast:
        for name, param in emb.named_parameters():
            key_split = name.split('.')[-1]
            if not key_split == 'r' and not key_split == 's' and not key_split == 'r_s_' and not key_split == 's_s_' \
                and not (multiheads and name == 'backbone.outc.conv.weight') \
                and not (multiheads and name == 'backbone.outc.conv.weight'):
                param.requires_grad = False
    if onlyslow:
        for name, param in emb.named_parameters():
            key_split = name.split('.')[-1]
            if key_split == 'r' or key_split == 's' or key_split == 'r_s_' or key_split == 's_s_' \
                or (multiheads and name == 'backbone.outc.conv.weight') \
                or (multiheads and name == 'backbone.outc.conv.weight'):
                param.requires_grad = False
    """
    else:
        for name, param in emb.named_parameters():
            key_split = name.split('.')[-1]
            if key_split == 'r' or key_split == 's':
                param.requires_grad = False
    """
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

        if 'lr_scheduler' in cfg['task'].keys() and cfg['task']['lr_scheduler'] == 'calr':
            lr_scheduler = lambda optimizer : SchedulerSE(optimizer, members=cfg['task']['members'], max_iter=len(dataset) // batch_size)
        else:
            lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

        try:
            supervisor.supervise(lr=lr, epochs=epochs,
                            batch_size=batch_size, name=store_name, pretrained=pretrained, pretrained_opt=pretrained_opt, num_workers=num_workers, lr_scheduler=lr_scheduler)
        finally:
            pass

    elif arch == 'test':
        supervisor.evaluate(dataset, cfg['task']['batch_size'], cfg['task']['num_workers'] )
    



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    # Read
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--client_type', type=str, default=None, help='A default client, anything else is overwritten.')
    parser.add_argument('--config_embedding', type=str, default='configs/embeddings/unet.yaml', help='Path to embedding config file.')
    parser.add_argument('--stain_restainer_path', type=str, default=None, help='Path to stored stain normalizer.')
    parser.add_argument('--config_predictor', type=str, default='configs/predictors/identity.yaml', help='Path to predictor config file.')
    parser.add_argument('--config_task', type=str, default='configs/tasks/seg_train_super.yaml', help='Path to task config file.')
    parser.add_argument('--config_data', type=str, default=None, help='Path to data config file.')
    parser.add_argument('--region_list', type=str, default=None, help='Path to file enumerating WSI regions.')
    parser.add_argument('--config_augmentations', type=str, default='configs/augmentations/base.yaml', help='Path to augmentation config file.')
    parser.add_argument('--id', type=str, help='ID of Experiment.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='Set seed for repoducability.')
    parser.add_argument('--n_files', type=int, default=0)
    parser.add_argument('--n_regions_per_file', type=int, default=0)   
    parser.add_argument('--result_file', type=str, default=None)   
    parser.add_argument("--other_args", default={}, dest="other_dict", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL")
    args = parser.parse_args()

    if args.client_type is not None:
        with open(args.client_type, 'r') as f:
            client = yaml.safe_load(f)
        args.config_embedding = client['embedding']
        args.config_predictor= client['predictor']
        args.config_task = client['task']
        args.config_data = client['data']
        args.config_augmentations = client['augmentations']
    
    if args.seed is not None:
        import torch
        import numpy as np
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        np.random.seed(args.seed)
        
    cfg = load_config(args.config_embedding, args.config_predictor, args.config_task, args.config_data, args.config_augmentations)

    # Update parameters that are unique per experiement
    cfg['task']['id'] = args.id
    cfg['dataset']['n_files'] = args.n_files
    cfg['dataset']['n_regions_per_file'] = args.n_regions_per_file
    cfg['result_file'] = args.result_file
    cfg['dataset']['region_list'] = args.region_list

    if args.stain_restainer_path is not None:
        cfg['supervisor']['stain_restainer_path'] = args.stain_restainer_path

    cfg.update(args.other_dict)

    run(cfg)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
