from configs.federated_configs import load_federated_config, get_federated_supervisor
import argparse
from federated.supervisor import FedStain
from single.utils import bcolors
import random

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def run(cfg, cfg_t, verbose=0):
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Supervisor
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    supervisor = get_federated_supervisor(cfg, verbose)
    evaluator = get_federated_supervisor(cfg_t, verbose)
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Perform experiment
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    arch = cfg[2]['task']['name']
    if arch == 'train':
        supervisor.supervise(cfg[2]['task']['n_rounds'], cfg[2]['task']['n_clients_per_round'], evaluator=evaluator)
    elif arch == 'test':
        supervisor.evaluate()

def run_stain_sampling(server, clients):
    stainer = FedStain(server=server, clients=clients, target_schemes=[["../patho_data/peso/pds_39_HE_small.tif", "../patho_data/peso/pds_32_HE_small.tif"]])
    stainer.supervise()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
    # Read
    print(bcolors.WARNING  + ("+" * 80) + "\n" + "Parse Args and Load Configs" + ("\n") + ("+" * 80) + bcolors.ENDC)    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--federation_type', type=str, default=None, help='Federation to train.')
    parser.add_argument('--dataplan', type=str, default=None, help='Data plan.')
    parser.add_argument('--config_task', type=str, default='configs/tasks/train/federated/seg_fed_base.yaml', help='Path to hyperparams config file.')
    parser.add_argument('--stain_sampling', type=bool, default=False, help='Wether do perform stain normalization and sampling.')
    parser.add_argument('--config_test_task', type=str, default='configs/tasks/train/federated/seg_fed_base.yaml', help='Path to hyperparams config file.')
    parser.add_argument('--config_unlabeled_task', type=str, default='configs/tasks/train/federated/seg_fed_base.yaml', help='Path to hyperparams config file.')
    parser.add_argument('--verbose', type=int, default=0, help='How much to log.')
    parser.add_argument('--id', type=str, help='ID of Experiment.')
    parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='Set seed for repoducability.')
    parser.add_argument('--result_file', type=str, default=None) 

    args = parser.parse_args()

    if args.seed is not None:
        import torch
        import numpy as np
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

        np.random.seed(args.seed)
    

    clients, server, cfg = load_federated_config(args.federation_type, args.dataplan, args.config_task, verbose=args.verbose)
    clients_t, server_t, cfg_t = load_federated_config(args.federation_type, args.dataplan, args.config_test_task, verbose=args.verbose)

    if args.stain_sampling:
        run_stain_sampling(server, clients)

    server['result_file'] = args.result_file
    server_t['result_file'] = args.result_file
    run(cfg=(clients, server, cfg), cfg_t=(clients_t, server_t, cfg_t), verbose=args.verbose)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++