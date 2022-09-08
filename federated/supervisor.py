from functools import partial
import os
from torchvision import transforms
import yaml
from datasets.wsi_data import MultiStainMultiFile, MultiStainTensor, PairedDatasetST, read_region_list
from modules.embedding import BottleGAN, StainGAN
from run_single import run
import numpy as np
import torch
from torch import nn
from single.data import PathologyAugmentations
from single.supervisors import BottleSupervisor,  StainSupervisor
from single.utils import bcolors
from single.models import CombinedNet
from configs.single_configs import get_embedding, get_predictor

class FedStain():
    def __init__(self, server, clients, target_schemes, n_examples=1, length=6000, image_size=114):
        self.clients = clients
        self.server = server
        self.target_schemes = target_schemes
        self.n_examples = n_examples
        self.length = length
        self.trans = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()])
        self.image_size = image_size

    def _get_supervisor(self, client, with_augs=False):
        region_list = client['dataset']['region_list'] 
        n_files = -1
        n_regions_per_file = -1
        img_files_from, label_files, regions, all_img_files = read_region_list(region_list, 
                                                            n_files=n_files,
                                                            n_regions_per_file=n_regions_per_file)
        img_files_from = [img_files_from]

        if not with_augs:
            dataset1 = MultiStainMultiFile(img_files=img_files_from,      n_examples=self.n_examples, length=self.length, transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3)
            dataset2 = MultiStainMultiFile(img_files=self.target_schemes, n_examples=self.n_examples, length=self.length, transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3)
        else:
            with open('configs/augmentations/all_staining.yaml', 'r') as f:
                augs = yaml.safe_load(f)['weak_augmentations']
            dataset1 = MultiStainMultiFile(img_files=img_files_from,      n_examples=self.n_examples, length=self.length * 5, transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3,
                                            augs=partial(PathologyAugmentations, cfg=augs))
            dataset2 = MultiStainMultiFile(img_files=self.target_schemes, n_examples=self.n_examples, length=self.length * 5, transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3,
                                            augs=partial(PathologyAugmentations, cfg=augs))
        dataset = PairedDatasetST(dataset1=dataset1, dataset2=dataset2)
        embedding = StainGAN()

        bb = CombinedNet(backbone=embedding, predictor=nn.Identity())
        supervisor = StainSupervisor(bb,
                                    dataset,
                                    {'cycle_loss': nn.L1Loss(reduction='mean'), 'disc_loss': torch.mean}, classifier=None).to('cuda')     
        
        return supervisor

    def _get_all_supervisor(self, with_augs=False):
        img_files_from_all = []
        for client in self.clients:
            region_list = client['dataset']['region_list'] 
            n_files = -1
            n_regions_per_file = -1
            img_files_from, label_files, regions, all_img_files = read_region_list(region_list, 
                                                                n_files=n_files,
                                                                n_regions_per_file=n_regions_per_file)
            img_files_from_all += [img_files_from]
        img_files_from_all += self.target_schemes

        dataset_name = img_files_from_all[0][0].split('/')[2]

        if not with_augs:
            dataset1 = MultiStainTensor(folder_path='../patho_data/' + dataset_name + '/stainings/')
            dataset2 = MultiStainMultiFile(img_files=self.target_schemes, n_examples=self.n_examples, length=self.length * 5, 
                                            transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3)
        else:
            with open('configs/augmentations/all_staining.yaml', 'r') as f:
                augs = yaml.safe_load(f)['weak_augmentations']
            dataset1 = MultiStainTensor(folder_path='../patho_data/' + dataset_name + '/stainings/', augs=partial(PathologyAugmentations, cfg=augs))
            dataset2 = MultiStainMultiFile(img_files=self.target_schemes, n_examples=self.n_examples, length=self.length * 5, 
                                            transform=self.trans, h=self.image_size, w=self.image_size, verbose=False, min_non_white=0.3,
                                            augs=partial(PathologyAugmentations, cfg=augs))       

        dataset = PairedDatasetST(dataset1=dataset1, dataset2=dataset2)
        embedding = BottleGAN(star_size=len(self.clients)+1)

        bb = CombinedNet(backbone=embedding, predictor=nn.Identity())
        supervisor = BottleSupervisor(bb,
                                    dataset,
                                    {'cycle_loss': nn.L1Loss(reduction='mean'), 'disc_loss': torch.mean}, classifier=None).to('cuda')     
        return supervisor

    def supervise(self, with_augs=False):
        for client_id, client in enumerate(self.clients):
            supervisor = self._get_supervisor(client, with_augs=with_augs)
            store_name = 'store/gans/' + client['task']['id'].split('/')[1] + '_stain' + ('' if not with_augs else '_augs')
            if os.path.isfile(store_name +'.pt') :            
                client_model = torch.load(store_name +'_inc_meta.pt').to('cuda')
                emb = client_model.backbone.label_embeddings
                bb = nn.DataParallel(client_model.backbone.G_S2_S1)
                dataset_name = supervisor.dataset.dataset1.img_files[0][0].split('/')[2]
                store_folder = '../patho_data/' + dataset_name + '/stainings/' + str(client_id)
                store_name_b = store_folder + '/' + str(0) + '.pt'
                if os.path.isfile(store_name_b):
                    continue    

                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)

                dataset = supervisor.dataset.dataset2
                dataset.len = (dataset.len * 4) // len(self.clients)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=36,
                                                   shuffle=False, num_workers=12,  pin_memory=False, prefetch_factor=2)

                for batch_id, batch in enumerate(train_loader):
                    batch, L =  batch[0]
                    store_name_b = store_folder + '/' + str(batch_id) + '.pt'

                    T1 = emb(torch.ones(batch.shape[0], dtype=int, device='cuda'))

                    with torch.no_grad():
                        out = bb(batch.to('cuda'), T1.to('cuda'), None, None)
                        torch.save(out, store_name_b)
                continue

            supervisor.supervise(lr=1e-3, epochs=1,
                            batch_size=36, name=store_name, pretrained=False, pretrained_opt=False, num_workers=12)
        client = self.server
        store_name = 'store/gans/' + client['task']['id'].split('/')[1] + '_stain'
        if not os.path.isfile(store_name +'.pt') :
            supervisor = self._get_supervisor(client)
            supervisor.supervise(lr=1e-3, epochs=1,
                        batch_size=36, name=store_name, pretrained=False, pretrained_opt=False, num_workers=12)

        store_name = 'store/gans/' + 'all_stain'
        if not os.path.isfile(store_name +'.pt'):
            supervisor = self._get_all_supervisor()
            supervisor.supervise(lr=1e-3, epochs=1,
                            batch_size=36, name=store_name, pretrained=False, pretrained_opt=False, num_workers=12)

        store_name = 'store/gans/' + 'all_stain_augs'
        if not os.path.isfile(store_name +'.pt'):
            supervisor = self._get_all_supervisor(with_augs=True)
            supervisor.supervise(lr=1e-3, epochs=1,
                            batch_size=36, name=store_name, pretrained=False, pretrained_opt=False, num_workers=12)

class FedAvg():
    def __init__(self, clients, server, cfg, only_labeled=True, verbose=0):
        self.clients = clients
        self.server = server
        self.cfg = cfg
        self.only_labeled = only_labeled
        self.verbose = verbose

        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)

    def supervise(self, n_rounds, n_clients_per_round, evaluator):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Initialize Server
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Get Network 
        embedding = get_embedding(self.server)
        predictor = get_predictor(self.server)
        model = CombinedNet(backbone=embedding, predictor=predictor)
        store_name = 'store/' + str(self.server['task']['id'])


        # Store both as clean init
        torch.save(model.state_dict(), store_name + ".pt")   
        torch.save(model, store_name + "_inc_meta.pt")  

        #Get Optimizer
        embedding = get_embedding(self.clients[0])
        predictor = get_predictor(self.clients[0])
        model = CombinedNet(backbone=embedding, predictor=predictor)
        store_name = 'store/' + str(self.server['task']['id'])
        optimizer = torch.optim.Adam(model.parameters())

        torch.save(optimizer.state_dict(), store_name + "_opt.pt")   
        print(bcolors.OKBLUE + "Saved at", store_name + "." + bcolors.ENDC)


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Round per round logic
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        print(bcolors.WARNING + '\n\n' + ("+" * 80) + "\n" + "Started Federated Learning" + ("\n") + ("+" * 80) + bcolors.ENDC)    
        for round in range(n_rounds):
            self.round = round

            # Set initial loading true, otherwise might not be clean
            if round == 0:
                for client in self.clients:
                    client['task']['pretrained'] = True
                    client['task']['pretrained_opt'] = True
                    client['unlabeled']['task']['pretrained'] = True
                    client['unlabeled']['task']['pretrained_opt'] = True

            print(bcolors.WARNING + "\n" + "Round  " + str(round + 1) + "/" + str(n_rounds) + ("\n") + ("+" * 40) + bcolors.ENDC)   

            # Train clients (internally also distributes server model if needed)
            self._train_clients(n_clients_per_round)

            # Collect clients back (needs server training for some supervisors)
            _ = self._collect_client_models()

            # Evaluate eprformance on val set
            evaluator.evaluate(round)
            torch.cuda.empty_cache()
    
    def _train_clients(self, n_clients_per_round):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Select clients to participate
        clients_selected = np.random.choice(np.arange(0, len(self.clients)), size=n_clients_per_round, replace=False)
        self.to_train = []

        for client_id in clients_selected:
            if self.clients[client_id]['labeled'] or not self.only_labeled:
                self.to_train.append(client_id)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Distribute server model
        self._distribute_server_model()

        print(bcolors.FAIL + "\n**** Run..." + bcolors.FAIL)  
        self.trained_ids = []
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Train clients
        for client_id in clients_selected:
            if not self.clients[client_id]['labeled'] and self.only_labeled:
                print('Skipped as not labeled ' + str(self.clients[client_id]['task']['id']) +  '.')
                continue 
            print(bcolors.FAIL + "**  ...Client " + str(self.clients[client_id]['task']['id']) + bcolors.ENDC)    
            run(self.clients[client_id] if self.clients[client_id]['labeled'] else self.clients[client_id]['unlabeled'], verbose=self.verbose)
            self.trained_ids.append(client_id)


    def _collect_client_models(self):
        client_models = []
        opts = []

        print(bcolors.FAIL + "\n**** Collect Clients" + bcolors.FAIL)   
        # Load clients
        sum_regions = 0


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Collect client models and optimizer     
        for client in self.clients:
            try:
                # Weight equally as done in recent publications
                n_regions = 1 #max(client['dataset']['n_regions_per_file'], 1)
                sum_regions += n_regions
                embedding = get_embedding(client)
                predictor = get_predictor(client)

                model = CombinedNet(backbone=embedding, predictor=predictor)
                store_name = 'store/' + str(client['task']['id'])

                # Load client model
                pretrained_dict = torch.load(store_name + ".pt")
                print(bcolors.OKBLUE + "Loaded", store_name + "." + bcolors.ENDC)
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                client_models.append((n_regions, model.state_dict()))

                # Load client optimizer
                optimizer = torch.optim.Adam(model.parameters())
                pretrained_dict = torch.load(store_name + "_opt.pt")
                print(bcolors.OKBLUE + "Loaded", store_name + "_opt." + bcolors.ENDC)
                model_dict = optimizer.state_dict()
                model_dict.update(pretrained_dict)
                optimizer.load_state_dict(model_dict)
                opts.append(optimizer.state_dict())
            except:
                print(bcolors.OKBLUE + "Not trained yet", store_name + "." + bcolors.ENDC)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Average
        default_dict = client_models[0][1]
        for key in default_dict:
            default_dict[key] = (default_dict[key]  * (client_models[0][0] / sum_regions))

        for key in default_dict:
            for c_i in range(1, len(client_models)):
                default_dict[key] = (default_dict[key] + (client_models[c_i][1][key] * (client_models[c_i][0] / sum_regions)))

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Store to server
        embedding = get_embedding(self.server)
        predictor = get_predictor(self.server)
        model = CombinedNet(backbone=embedding, predictor=predictor)
        store_name = 'store/' + str(self.server['task']['id'])

        model_dict = model.state_dict()
        model_dict.update(default_dict)
        model.load_state_dict(model_dict)
        torch.save(model.state_dict(), store_name + ".pt")   
        torch.save(model, store_name + '_inc_meta.pt')
        print(bcolors.OKBLUE + "Saved at", store_name + "." + bcolors.ENDC)


        default_dict = opts[0]
        torch.save(default_dict, store_name + "_opt.pt")   
        print(bcolors.OKBLUE + "Saved at", store_name + "_opt." + bcolors.ENDC)

        return default_dict
    
    def _distribute_server_model(self):
        print(bcolors.FAIL + "\n**** Distribute Server" + bcolors.FAIL)  
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Do the loading
        embedding = get_embedding(self.server)
        predictor = get_predictor(self.server)
        model = CombinedNet(backbone=embedding, predictor=predictor)
        store_name = 'store/' + str(self.server['task']['id'])

        pretrained_dict = torch.load(store_name + ".pt")
        print(bcolors.OKBLUE + "Loaded", store_name + "." + bcolors.ENDC)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        torch.save(model, store_name + '_inc_meta.pt')


        optimizer = torch.optim.Adam(model.parameters())
        pretrained_dict = torch.load(store_name + "_opt.pt")
        print(bcolors.OKBLUE + "Loaded", store_name + "_opt." + bcolors.ENDC)
        opt_dict = optimizer.state_dict()
        opt_dict.update(pretrained_dict)
        optimizer.load_state_dict(opt_dict)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Store to clients that participate in the round
        for client_id in self.to_train:
            client = self.clients[client_id]
            store_name = 'store/' +  str(client['task']['id'])
            torch.save(model.state_dict(), store_name + ".pt")   
            torch.save(optimizer.state_dict(), store_name + "_opt.pt")   
            print(bcolors.OKBLUE + "Saved at", store_name + "." + bcolors.ENDC)
            print(bcolors.OKBLUE + "Saved at", store_name + "_opt." + bcolors.ENDC)

class FedAvgM(FedAvg):
    def __init__(self, clients, server, cfg, momentum, only_labeled=True, with_std = False, verbose=0):#
        super().__init__(clients=clients, server=server, cfg=cfg, only_labeled=only_labeled, verbose=verbose)
        self.momentum = momentum
        self.with_std = with_std

    def _collect_client_models(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Same as for FedAVG but weighted update
        client_models = []
        opts = []
        sum_regions = 0
        print(bcolors.FAIL + "\n**** Collect Clients" + bcolors.FAIL)   

        # Load clients
        for client in self.clients:
            try:
                n_regions = 1 #max(client['dataset']['n_regions_per_file'], 1)
                embedding = get_embedding(client)
                predictor = get_predictor(client)

                model = CombinedNet(backbone=embedding, predictor=predictor)
                store_name = 'store/' + str(client['task']['id'])

                pretrained_dict = torch.load(store_name + ".pt")
                print(bcolors.OKBLUE + "Loaded", store_name + "." + bcolors.ENDC)
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                sum_regions += n_regions
                client_models.append((n_regions, model.state_dict()))

                optimizer = torch.optim.Adam(model.parameters())
                pretrained_dict = torch.load(store_name + "_opt.pt")
                print(bcolors.OKBLUE + "Loaded", store_name + "_opt." + bcolors.ENDC)
                model_dict = optimizer.state_dict()
                model_dict.update(pretrained_dict)
                optimizer.load_state_dict(model_dict)
                opts.append(optimizer.state_dict())
            except:
                print(bcolors.OKBLUE + "Not trained yet", store_name + "." + bcolors.ENDC)

        # Average
        default_dict = client_models[0][1]
        for key in default_dict:
            default_dict[key] = (default_dict[key]  * (client_models[0][0] / sum_regions)).cpu()

        for key in default_dict:
            for c_i in range(1, len(client_models)):
                default_dict[key] = (default_dict[key] + (client_models[c_i][1][key] * (client_models[c_i][0] / sum_regions))).cpu()

        if self.with_std:
            std_dict = {}
            for key in default_dict:
               if key.split('.')[-1] == 'std':
                   std_dict[key] = []

            for key in std_dict:
                w_key = key[:-4]
                w_key += '.weight'
                for c_i in range(1, len(client_models)):    
                    std_dict[key].append(client_models[c_i][1][w_key])   

            for key in std_dict:
                std_dict[key] = torch.std(torch.stack(std_dict[key], dim=0), dim=0)

            for key in std_dict:
                default_dict[key] = std_dict[key]


        # Store to server
        embedding = get_embedding(self.server)
        predictor = get_predictor(self.server)
        model = CombinedNet(backbone=embedding, predictor=predictor)
        store_name = 'store/' + str(self.server['task']['id'])

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Momentum Update
        if self.round > 0:
            print(bcolors.OKBLUE + "Loaded for Momentum", store_name + "." + bcolors.ENDC)
            pretrained_dict = torch.load(store_name + ".pt")
            for key in default_dict:
                default_dict[key] = (default_dict[key]  * (1 - self.momentum)) + (pretrained_dict[key].cpu()  * (self.momentum))

        model_dict = model.state_dict()
        model_dict.update(default_dict)
        model.load_state_dict(model_dict)
        torch.save(model.state_dict(), store_name + ".pt")   
        torch.save(model, store_name + "_inc_meta.pt")   
        print(bcolors.OKBLUE + "Saved at", store_name + "." + bcolors.ENDC)

        default_dict = opts[0]
        torch.save(default_dict, store_name + "_opt.pt")   
        print(bcolors.OKBLUE + "Saved at", store_name + "_opt." + bcolors.ENDC)

        return default_dict


