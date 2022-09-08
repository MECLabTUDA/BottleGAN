# System
import yaml
import os
from copy import deepcopy

# Models
from federated.supervisor import  FedAvg, FedAvgM
from evaluation.federated_evaluation import FederatedEvaluator

# Single config
from configs.single_configs import update_default, load_config

# Utils
from super_selfish.utils import bcolors

def load_federated_config(federation_path, data_path, hyperparams_path, verbose=0):

    # Load Files
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    with open(federation_path, 'r') as f:
        federation = yaml.safe_load(f)

    # Hyperparams
    with open(hyperparams_path, 'r') as f:
        params_cfg = yaml.safe_load(f)
    
    # Data
    with open(data_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    if verbose > 0:
        print("Dataplan", data_cfg)

    # Hyper Params for Federated Supervisor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train = params_cfg['task']['name'] == 'train'

    print(bcolors.OKBLUE + "\nLoaded " + str(params_cfg['task']['name']) + "  hyperparameters" + bcolors.ENDC)
    if verbose > 0:
        print(params_cfg)

    # Wether to load pretrained
    if train:
        pretrained = params_cfg['task']['pretrained']
        pretrained_opt = params_cfg['task']['pretrained_opt']

    # Clients
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    clients = federation['clients']
    def load_client(client):
        with open(client[1], 'r') as f:
            client_type = yaml.safe_load(f)
        if not train:
            client_type['task'] = client_type['test_task']

        cfg = load_config(client_type['embedding'], client_type['predictor'],  client_type['task'], 
                    client_type['data'], client_type['augmentations'], verbose=0)
        cfg['task']['id'] = client[0]

        if 'prior_noise' in federation.keys():
            cfg['embedding_architecture']['prior_noise'] = federation['prior_noise']

        if 'init_noise' in federation.keys():
            cfg['embedding_architecture']['init_noise'] = federation['init_noise']

        if 'init_offset' in federation.keys():
            cfg['embedding_architecture']['init_offset'] = federation['init_offset']

        if 'add_noise' in federation.keys():
            cfg['embedding_architecture']['add_noise'] = federation['add_noise']

        if 'samples' in federation.keys():
            cfg['embedding_architecture']['samples'] = federation['samples']

        if train:

            cfg_u = load_config(client_type['embedding'], client_type['predictor'],  client_type['unlabeled_task'], 
                        client_type['data'], client_type['unlabeled_augmentations'], verbose=0)
            cfg_u['task']['id'] = client[0]
            if 'stain_restainer_path' in client_type.keys():
                cfg['supervisor']['stain_restainer_path'] = client_type['stain_restainer_path']
                cfg_u['supervisor']['stain_restainer_path'] = client_type['unlabeled_stain_restainer_path']
                cfg['supervisor']['stain_normalizer_path'] = 'store/gans/' + client[0].split('/')[1] + '_stain_inc_meta.pt'
                cfg_u['supervisor']['stain_normalizer_path'] = 'store/gans/' + client[0].split('/')[1] + '_stain_inc_meta.pt'
            cfg['unlabeled'] = cfg_u

            if 'style_noise' in federation.keys():
                cfg['supervisor']['style_noise'] = federation['style_noise']
                cfg_u['supervisor']['style_noise'] = federation['style_noise']

            if 'inter_emb' in federation.keys():
                cfg['supervisor']['inter_emb'] = federation['inter_emb']
                cfg_u['supervisor']['inter_emb'] = federation['inter_emb']

            if 'server_path' in federation.keys():
                cfg['supervisor']['server_path' ] = federation['server_path']
                cfg_u['supervisor']['server_path' ] = federation['server_path']

            if 'ssl_type' in federation.keys():
                cfg['supervisor']['ssl_type' ] = federation['ssl_type']
                cfg_u['supervisor']['ssl_type' ] = federation['ssl_type']   

            if 'prior_noise' in federation.keys():
                cfg_u['embedding_architecture']['prior_noise'] = federation['prior_noise']

            if 'init_noise' in federation.keys():
                cfg_u['embedding_architecture']['init_noise'] = federation['init_noise']

            if 'init_offset' in federation.keys():
                cfg_u['embedding_architecture']['init_offset'] = federation['init_offset']

            if 'add_noise' in federation.keys():
                cfg_u['embedding_architecture']['add_noise'] = federation['add_noise']

            if 'samples' in federation.keys():
                cfg_u['embedding_architecture']['samples'] = federation['samples']



        return cfg

    clients_cfgs = []

    for client_id_nr in range(federation['clients']['n_clients']):
        client_id = 'id' + str(client_id_nr)
        labeled = data_cfg[client_id]['labeled']
        try:
            n_files = data_cfg[client_id]['n_files']
        except: 
            n_files = data_cfg['n_files']

        try:
            n_regions = data_cfg[client_id]['n_regions']
        except: 
            n_regions = data_cfg['n_regions']


        client = federation['clients']
        if client['base'] is None:
            continue

        cfg = load_client(('clients/' + client_id, client['base']))

        if 'dataset' not in cfg.keys():
            cfg['dataset'] = {}
        cfg['dataset']['n_files'] = n_files if train else -1
        cfg['dataset']['n_regions_per_file'] = n_regions if train else -1
        cfg['dataset']['length'] = data_cfg['length'] if train else 2000
        cfg['dataset']['type'] = ('semi' if (cfg['supervisor']['name'] == 'semi_segmentation' or cfg['supervisor']['name'] != 'sim_semi_segementation') else 'train') if train else 'test'
        cfg['dataset']['region_list'] = data_cfg['fold_folder'] + 'clients/' + str(client_id_nr) + "_train.txt"
        if train:
            cfg['unlabeled']['dataset'] = deepcopy(cfg['dataset'])
            cfg['unlabeled']['dataset']['n_regions_per_file'] = -1
            cfg['unlabeled']['dataset']['type'] = 'semi'
            cfg['unlabeled']['dataset']['length'] = 1000
            cfg['labeled'] = labeled


        if 'update' in client.keys():
            update_default(cfg, client['update'])
        
        

        if train:
            cfg['task']['pretrained'] = pretrained
            cfg['task']['pretrained_opt']= pretrained_opt

        print(bcolors.OKBLUE + "\nLoaded " + str(params_cfg['task']['name']) + " config for client " + str(client_id)  + bcolors.ENDC)

        if verbose > 0:
            print(cfg)
        clients_cfgs.append(cfg)

    # Server
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    n_regions, labeled = data_cfg['server']['n_regions'],  data_cfg['server']['labeled']

    server = federation['server']
    if server['base'] is not None:
        server_cfg = load_client(('server/server', server['base']))
        if 'dataset' not in server_cfg.keys():
            server_cfg['dataset'] = {}

        server_cfg['dataset']['n_files'] = data_cfg['server']['n_files'] if train else -1
        server_cfg['dataset']['n_regions_per_file'] = n_regions if train else -1
        server_cfg['dataset']['length'] = data_cfg['server']['length'] if train else 2000
        server_cfg['dataset']['type'] = 'train' if train else 'test'

        if train:
            server_cfg['dataset']['region_list'] = data_cfg['fold_folder'] + "server/train.txt"
        else:
            server_cfg['dataset']['region_list'] = data_cfg['fold_folder'] + "clients/all_test.txt"

        if train:
            server_cfg['unlabeled']['dataset'] = deepcopy(cfg['dataset'])
            server_cfg['unlabeled']['dataset']['n_regions_per_file'] = -1
            server_cfg['unlabeled']['dataset']['type'] = 'semi'
            server_cfg['unlabeled']['dataset']['length'] = 1000
            server_cfg['labeled'] = labeled


        if 'update' in server.keys():
            update_default(server_cfg, federation['server']['update'])

        if train:
            server_cfg['task']['pretrained'] = pretrained
            server_cfg['task']['pretrained_opt']= pretrained_opt
    else:
        server_cfg = None

    print(bcolors.OKBLUE + "\nLoaded " + str(params_cfg['task']['name']) + "  config for server " + bcolors.ENDC)

    if verbose > 0:
        print(server_cfg)


    if 'gauss_weights' in federation.keys():
        params_cfg['supervisor']['gauss_weights'] = federation['gauss_weights']

    return clients_cfgs, server_cfg, params_cfg

def get_federated_supervisor(cfg, verbose):
    clients, server, cfg = cfg
    arch = cfg['supervisor']['name']

    if arch == 'base_evaluator':
        supervisor = FederatedEvaluator(clients=clients, server=server, cfg=cfg, verbose=verbose)
    elif arch == 'fed_avg':
        supervisor = FedAvg(clients=clients, server=server, cfg=cfg, verbose=verbose)
    elif arch == 'fed_avgm':
        supervisor = FedAvgM(clients=clients, server=server, cfg=cfg, momentum=cfg['task']['server_momentum'], 
                             only_labeled=cfg['task']['only_labeled'], with_std=cfg['supervisor']['gauss_weights'],
                             verbose=verbose)
    return supervisor
