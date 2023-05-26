import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models import CombinedNet#, CombinedUNet not reqiure here but possible extension
from tqdm import tqdm
from colorama import Fore
from utils import FocalLoss, bcolors
import json
import collections
import piq
from copy import deepcopy
import numpy as np
import random
import PIL.Image as im_
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fold Classes...
# Bottle GAN is at 392
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Stain Evaluator
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class StainEvaluator():
    def __init__(self, model, result_file=None):
        self.stain_normalizer = model['G']
        self.stain_restainer = model['G2']
        if model['classifier'] is not None:
            self.classifier = model['classifier']
            self.features = deepcopy(self.classifier.module.backbone)
            self.features.return_features=True
            self.features = nn.DataParallel(self.features)
        self.embeddings = model['embeddings']
        self.result_file = result_file

    def evaluate(self, dataset, batch_size, num_workers):
        dataset.fix_index = True
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        metric_names = ['psnr', 'mse', 'fid']
        tkb = [tqdm(total=int(len(test_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), position=i, desc=str(metric_names[i]), leave=False) for i in range(len(metric_names))]
    

        metric_values = torch.zeros((len(test_loader), len(metric_names)))

        
        # Evaluation +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for batch_id, batch in enumerate(test_loader):
            S1, S2 = batch
            if S1.shape[0] != test_loader.batch_size:
                batch_id -= 1
                continue
            
            S1, S2 = S1.to('cuda'), S2.to('cuda')
            L1 = torch.arange((batch_id * test_loader.batch_size), ((batch_id + 1) * test_loader.batch_size)) % 240

            
            with torch.no_grad():
                T1 = self.embeddings[0](L1).to('cuda')
                S1_f = self.stain_normalizer(S1, T1, None, None)  
                S1_ff = self.stain_restainer(S1_f, T1, None, None)

                loss = PSNR(S1_ff, S1)
                metric_values[batch_id, 0] = loss
                mean = metric_values[:batch_id+1, 0].mean(0)
                tkb[0].set_postfix({metric_names[0]: mean.item()})
                tkb[0].update(1)

                loss = F.mse_loss(S1_ff, S1)
                metric_values[batch_id, 1] = loss
                mean = metric_values[:batch_id+1, 1].mean(0)
                tkb[1].set_postfix({metric_names[1]: mean.item()})
                tkb[1].update(1)

                loss = FID(self.features(S1_ff).reshape(S1_ff.shape[0], 256, -1).mean(2), self.features(S1).reshape(S1.shape[0], 256, -1).mean(2))
                metric_values[batch_id, 2] = loss
                mean = metric_values[:batch_id+1, 2].mean(0)
                tkb[2].set_postfix({metric_names[2]: mean.item()})
                tkb[2].update(1)

        
        # Hacky, but working....
        print('\n' * len(metric_names))
        # Write to file
        if self.result_file is not None:
            out_dicts = []
            for m_id, n in enumerate(metric_names):
                mean = metric_values[:batch_id+1, m_id].mean(0)
                out_dicts.append((n, mean.item()))

            metrics_json = json.dumps(collections.OrderedDict(out_dicts))
            with open(self.result_file, 'w') as f:
                f.write(metrics_json)


        # T-SNE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        data_t = torch.zeros(test_loader.batch_size * 2, 114, 114, 3)
        for batch_id, batch in enumerate(test_loader):
            S1, S2 = batch
            if S1.shape[0] != test_loader.batch_size:
                batch_id -= 1
                continue
            
            S1, S2 = S1.to('cuda'), S2.to('cuda')
            L1 = torch.arange((batch_id * test_loader.batch_size), ((batch_id + 1) * test_loader.batch_size)) % 240

            
            with torch.no_grad():
                T1 = self.embeddings[0](L1).to('cuda')
                S1_ff = self.stain_normalizer(S1, T1, None, None)    

                data_t[batch_id * test_loader.batch_size :  (batch_id+1) * test_loader.batch_size] =  S1_ff.permute(0,2,3,1).cpu()
                if batch_id > 0:
                    break   

        data = (data_t * 255).numpy().astype(np.uint8)
        pca = PCA(50)

        converted_data = pca.fit_transform(data.reshape(data.shape[0], -1))
        tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(converted_data)
        tx, ty = tsne[:,0], tsne[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 2000
        height = 1500
        max_dim = 100

        full_image = im_.new('RGBA', (width, height))
        for img, x, y in zip(data, tx, ty):
            tile = im_.fromarray(img)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), im_.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        print(self.result_file[:-5] + '.png')
        full_image.save(self.result_file[:-5] + '.png')

# Metrics +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def PSNR(img1, img2):
    return piq.psnr(img1, img2)


def FID(img1, img2):
    loss = piq.FID()
    return loss(img1, img2)
 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Base Supervisor
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class Supervisor():

    def __init__(self, model, dataset=None, loss=nn.CrossEntropyLoss(reduction='mean'), collate_fn=None):
        """Constitutes a self-supervision algorithm. All implemented algorithms are childs. Handles training, storing, and
        loading of the trained model/backbone.

        Args:
            model (torch.nn.Module): The module to self supervise.
            dataset (torch.utils.data.Dataset): The dataset to train on.
            loss (torch.nn.Module, optional): The critierion to train on. Defaults to nn.CrossEntropyLoss(reduction='mean').
            collate_fn (function, optional): The collate function. Defaults to None.
        """
        if not isinstance(model, CombinedNet): # and not isinstance(model, CombinedUNet):
            raise("You must pass a CombinedNet to model.")
        self.model = nn.DataParallel(model)
        self.dataset = dataset
        self.loss = loss
        self.collate_fn = collate_fn
        self.memory=None

    def supervise(self, lr=1e-3, optimizer=torch.optim.Adam, epochs=10, batch_size=32, shuffle=False,
                  num_workers=0, name="store/base", pretrained=False, pretrained_opt=False, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)):
        """Starts the training procedure of a self-supervision algorithm.

        Args:
            lr (float, optional): Optimizer learning rate. Defaults to 1e-3.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler. Defaults to lambdaoptimizer:torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0).
            optimizer (torch.optim.Optimizer, optional): Optimizer to use. Defaults to torch.optim.Adam.
            epochs (int, optional): Number of epochs to train. Defaults to 10.
            batch_size (int, optional): Size of bachtes to process. Defaults to 32.
            shuffle (bool, optional): Wether to shuffle the dataset. Defaults to True.
            num_workers (int, optional): Number of workers to use. Defaults to 0.
            name (str, optional): Path to store and load models. Defaults to "store/base".
            pretrained (bool, optional): Wether to load pretrained model. Defaults to False.
        """
        if not isinstance(self.dataset, torch.utils.data.Dataset):
            raise("No dataset has been specified.")
        print(bcolors.OKGREEN + "Train with " +
              type(self).__name__ + bcolors.ENDC)
        self.name = name
        self._load_pretrained(name, pretrained)
        try:
            train_loader, optimizer, lr_scheduler = self._init_data_optimizer(
                optimizer=optimizer, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn, lr=lr, lr_scheduler=lr_scheduler)
            if pretrained_opt:
                try:
                    optimizer = self.load_optim(optimizer, name)
                except:
                    print("No pretrained optimizer loaded")

            self._epochs(epochs=epochs, train_loader=train_loader,
                         optimizer=optimizer, lr_scheduler=lr_scheduler)
        finally:
            self.save(name, optimizer)
            print()

    def _load_pretrained(self, name, pretrained):
        """Private method to load a pretrained model

        Args:
            name (str): Path to model.
            pretrained (bool): Wether to load pretrained model.

        Raises:
            IOError: [description]
        """
        try:
            if pretrained:
                self.load(name)

        except Exception as e:
            print("No pretrained model loaded")
            print(e)

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        """Creates all objects that are neccessary for the self-supervision training and are dependend on self.supervise(...).

        Args:
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
            batch_size (int, optional): Size of bachtes to process.
            shuffle (bool, optional): Wether to shuffle the dataset.
            num_workers (int, optional): Number of workers to use.
            collate_fn (function, optional): The collate function.
            lr (float, optional): Optimizer learning rate.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.

        Returns:
            Tuple: All created objects
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2, worker_init_fn=seed_worker, generator=g)

        optimizer = optimizer(self.model.parameters(), lr=lr)
        lr_scheduler = lr_scheduler(optimizer)

        return train_loader, optimizer, lr_scheduler

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        """Implements the training loop (epochs, batches, visualization) excluding the actual training step.

        Args:
            epochs (int, optional): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): Iterator over the dataset.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), position=0)
        for epoch_id in range(epochs):
            loss_sum = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size and batch_id == (len(train_loader) - 1):
                    continue
                optimizer.zero_grad()
                loss = self._forward(data)
                loss_sum += loss.item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer, 
                             lr_scheduler=lr_scheduler)
            tkb.reset()
        del tkb

    def _forward(self, data):
        """Forward part of training step. Conducts all forward calculations.

        Args:
            data (Tuple(torch.FloatTensor,torch.FloatTensor)): Batch of instances with corresponding labels.

        Returns:
            torch.FloatTensor: Loss of batch.
        """
        inputs, labels = data
        outputs = self.model(inputs.to('cuda'))
        loss = self.loss(outputs, labels.to('cuda'))
        return loss

    def _update(self, loss, optimizer, lr_scheduler):
        """Backward part of training step. Calculates gradients and conducts optimization step.
        Also handles other updates like lr scheduler.

        Args:
            loss (torch.nn.Module, optional): The critierion to train on.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    def to(self, name):
        """Wraps device handling.

        Args:
            name (str): Name of device, see pytorch.

        Returns:
            Supervisor: Returns itself.
        """
        self.model = self.model.to(name)
        return self

    def get_backbone(self):
        """Extracts the backbone network that creates features

        Returns:
            torch.nn.Module: The backbone network.
        """
        try:
            return self.model.module.backbone
        except:
            return self.model.backbone

    def get_predictor(self):
        """Extracts the predictor network

        Returns:
            torch.nn.Module: The backbone network.
        """
        try:
            return self.model.module.predictor
        except:
            return self.model.predictor

    def save(self, name="store/base", optimizer=None):
        """Saves model parameters to disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        torch.save(self.model.module.state_dict(), name + ".pt")
        torch.save(self.model.module, name + "_inc_meta.pt")

        if self.memory is not None:
            self.memory.save(name + "_memory" + ".pt")
        print(bcolors.OKBLUE + "Saved at " + name + "." + bcolors.ENDC)
        if optimizer is not None:
            try:
                torch.save(optimizer.state_dict(), name + "_opt.pt")
            except:
                print('Optimizer not written.')
        return self

    def load(self, name="store/base"):
        """Loads model parameters from disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        pretrained_dict = torch.load(name + ".pt")
        print(bcolors.OKBLUE + "Loaded", name + "." + bcolors.ENDC)
        model_dict = self.model.module.state_dict()
        model_dict.update(pretrained_dict)
        self.model.module.load_state_dict(model_dict)
        if self.memory is not None:
            self.memory.load(name + "_memory" + ".pt")
        return self

    def load_optim(self, optimizer, name="store/base"):
        """Loads model parameters from disk.

        Args:
            name (str, optional): Path to storage. Defaults to "store/base".
        """
        pretrained_dict = torch.load(name + "_opt.pt")
        print(bcolors.OKBLUE + "Loaded", name + "_opt." + bcolors.ENDC)
        model_dict = optimizer.state_dict()
        model_dict.update(pretrained_dict)
        optimizer.load_state_dict(model_dict)

        return optimizer

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Cycle Supervisor
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
class CycleSupervisor(Supervisor):
    def __init__(self, model, dataset, loss, alpha=1.0):
        super().__init__(model, dataset, loss)
        self.alpha = alpha
    
    def _construct_loss(self,S1, S2, S1_i, S2_i, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f):

        # Cycle
        cycle_loss = self.loss['cycle_loss'](S1, S1_ff) + self.loss['cycle_loss'](S2, S2_ff)
        idt_loss = self.loss['cycle_loss'](S2, S2_i) + self.loss['cycle_loss'](S1, S1_i)  
        
        disc_loss = self.loss['disc_loss'](D1_f) + self.loss['disc_loss'](D2_f) \
                    - self.loss['disc_loss'](D1_r) - self.loss['disc_loss'](D2_r)

        gen_loss = -self.loss['disc_loss'](D1_f) - self.loss['disc_loss'](D2_f) \
            + cycle_loss * self.alpha + idt_loss * self.alpha

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'mse_loss': cycle_loss}

    def _forward(self, data, batch_id):
        S1,_, S2,_ = data
        S1, S2 = S1.to('cuda'), S2.to('cuda')
        S1_i, S2_i, S1_f, S2_f, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f = self.model(S1, S2)

        if batch_id % 50 == 0:
            self._draw(S1, S2, S1_f, S2_f, S1_ff, S2_ff)

        return self._construct_loss(S1, S2, S1_i, S2_i, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f)

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        """Implements the training loop (epochs, batches, visualization) excluding the actual training step.

        Args:
            epochs (int, optional): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): Iterator over the dataset.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), position=0)
        for epoch_id in range(epochs):
            loss_sum_G = 0
            loss_sum_D = 0
            loss_sum_mse = 0
            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0].shape[0] != train_loader.batch_size and batch_id == (len(train_loader) - 1):
                    continue
                for opt in optimizer.values():
                    opt.zero_grad()
                loss = self._forward(data, batch_id)
                loss_sum_G += loss['gen_loss'].item()
                loss_sum_D += loss['disc_loss'].item()
                loss_sum_mse += loss['mse_loss'].item()
                tkb.set_postfix(loss_G='{:3f}'.format(
                    loss_sum_G / (batch_id+1)), loss_D='{:3f}'.format(
                    loss_sum_D / (batch_id+1)), MSE='{:3f}'.format(
                    loss_sum_mse / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer, 
                             lr_scheduler=lr_scheduler, batch_id=batch_id)
            tkb.reset()
        del tkb

    def _update(self, loss, optimizer, lr_scheduler, batch_id):
        """Backward part of training step. Calculates gradients and conducts optimization step.
        Also handles other updates like lr scheduler.

        Args:
            loss (torch.nn.Module, optional): The critierion to train on.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        if batch_id % 2 == 0:
            loss['disc_loss'].backward()
            optimizer['D_opt'].step()
        else:
            loss['gen_loss'].backward()
            optimizer['G_opt'].step()
        lr_scheduler.step()

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2, generator=g, worker_init_fn=seed_worker)
        G_optimizer = optimizer(list(self.model.module.backbone.G_S1_S2.parameters()) + list(self.model.module.backbone.G_S2_S1.parameters()), lr=lr)
        D_optimizer = optimizer(list(self.model.module.backbone.D2.parameters()) + list(self.model.module.backbone.D1.parameters()), lr=lr)
        lr_scheduler = lr_scheduler(G_optimizer)

        return train_loader, {'G_opt': G_optimizer, 'D_opt': D_optimizer}, lr_scheduler

    def _draw(self, S1, S2, S1_f, S2_f, S1_ff, S2_ff, store_file='imgs/stain_tf.jpg', idx=0):
        images = [im_.fromarray((S1[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8)),
                  im_.fromarray((S2[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8)),
                  im_.fromarray((S1_f[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8)),
                  im_.fromarray((S2_f[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8)),
                  im_.fromarray((S1_ff[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8)),
                  im_.fromarray((S2_ff[idx].detach().permute(1,2,0).cpu().numpy().clip(min=0, max=1) * 255).astype(np.uint8))]
        widths, heights = zip(*(i.size for i in images[:2]))
        total_width = sum(widths)
        max_height = heights[0] * 3

        new_im = im_.new('RGB', (total_width, max_height))

        x_offset = 0
        y_offset = 0
        for i, im in enumerate(images):
            new_im.paste(im, (x_offset,y_offset))
            x_offset += im.size[0]
            if i % 2 == 1   :
                y_offset += im.size[1]
                x_offset = 0

        new_im.save(store_file)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Bottle Supervisor
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class BottleSupervisor(CycleSupervisor):
    def __init__(self, model, dataset, loss, alpha_1=1e+3, classifier=None):
        super().__init__(model, dataset, loss, alpha_1)
        self.classifier = classifier
        if self.classifier is not None:
            self.soft_loss = FocalLoss()

    def _construct_loss(self, batch_id, S1_i, S2_i, S1, S2, L1, L2, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f):
    
        # Cycle
        cycle_loss = self.loss['cycle_loss'](S1, S1_ff) +  self.loss['cycle_loss'](S2, S2_ff)
        idt_loss = self.loss['cycle_loss'](S2, S2_i) + self.loss['cycle_loss'](S1, S1_i)

        if self.classifier is None:
            classifier_loss = torch.zeros(1).to('cuda')
        else:
            with torch.no_grad():
                targets = torch.argmax(self.classifier(S2), dim=1)
            preds = self.classifier(S2_ff)
            classifier_loss = self.soft_loss(preds, targets) * 1e-0

        
        disc_loss = + self.loss['disc_loss'](D1_f) + self.loss['disc_loss'](D2_f) \
                    - self.loss['disc_loss'](D1_r) - self.loss['disc_loss'](D2_r) 

        gen_loss = - self.loss['disc_loss'](D1_f) - self.loss['disc_loss'](D2_f) \
                   + cycle_loss * self.alpha + idt_loss * self.alpha + classifier_loss

        return {'gen_loss': gen_loss, 'disc_loss': disc_loss, 'mse_loss': cycle_loss}

    def _epochs(self, epochs, train_loader, optimizer, lr_scheduler):
        """Implements the training loop (epochs, batches, visualization) excluding the actual training step.
        Args:
            epochs (int, optional): Number of epochs to train.
            train_loader (torch.utils.data.DataLoader): Iterator over the dataset.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        tkb = tqdm(total=int(len(train_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), position=0)
        for epoch_id in range(epochs):
            loss_sum_G = 0
            loss_sum_D = 0
            loss_sum_mse = 0

            tkb.set_description(desc="Batch Process Epoch " + str(epoch_id))
            for batch_id, data in enumerate(train_loader):
                if data[0][0][0].shape[0] != train_loader.batch_size and batch_id == (len(train_loader) - 1):
                    continue
                for opt in optimizer.values():
                    opt.zero_grad()
                loss = self._forward(data, batch_id)
                loss_sum_G += loss['gen_loss'].item()
                loss_sum_D += loss['disc_loss'].item()
                loss_sum_mse += loss['mse_loss'].item()

                tkb.set_postfix(loss_G='{:3f}'.format(
                    loss['gen_loss'].item()), loss_D='{:3f}'.format(
                    loss['disc_loss'].item()), MSE='{:3f}'.format(
                    loss['mse_loss'].item()))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer, 
                             lr_scheduler=lr_scheduler, batch_id=batch_id)
            tkb.reset()
        del tkb

    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        g = torch.Generator()
        g.manual_seed(0)

        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2, generator=g, worker_init_fn=seed_worker)
        G_optimizer = optimizer(list(self.model.module.backbone.G_S1_S2.parameters()) + list(self.model.module.backbone.G_S2_S1.parameters()), lr=lr)
        D_optimizer = optimizer(list(self.model.module.backbone.D2.parameters()) + list(self.model.module.backbone.D1.parameters()), lr=lr)
        lr_scheduler = lr_scheduler(G_optimizer)

        return train_loader, {'G_opt': G_optimizer, 'D_opt': D_optimizer}, lr_scheduler

    def _forward(self, data, batch_id):
        S, L = zip(*data)
        S1, L1, S2, L2 = S[:(len(S) // 2)], L[:(len(L) // 2)], S[(len(S) // 2):], L[(len(L) // 2):]
        S1, L1, S2, L2 = torch.cat(S1, dim=0), torch.cat(L1, dim=0), torch.cat(S2, dim=0), torch.cat(L2, dim=0)

        S1, S2 = S1.to('cuda'), S2.to('cuda')
        L1, L2 = L1.to('cuda'), L2.to('cuda')

        S1_i, S2_i, S1_f, S2_f, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f = self.model(S1, S2, L1, L2)  

        if batch_id % 50 == 0:
            self._draw(S1, S2, S1_f, S2_f, S1_ff, S2_ff, store_file='imgs/stain_tf_bottle.jpg')


        return self._construct_loss(batch_id,S1_i, S2_i, S1, S2, L1, L2, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f)