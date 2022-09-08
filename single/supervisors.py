import torch
from torch import nn
from torch.nn import functional as F
from .models import CombinedNet, CombinedUNet
from tqdm import tqdm
from colorama import Fore
from .utils import FocalLoss, bcolors
import numpy as np
import random
import PIL.Image as im_

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
        if not isinstance(model, CombinedNet) and not isinstance(model, CombinedUNet):
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

class StarSupervisor(CycleSupervisor):
    def __init__(self, model, dataset, loss, alpha_1=1e+3):
        super().__init__(model, dataset, loss, alpha_1)

    def _construct_loss(self, batch_id, S1_i, S2_i, S1, S2, L1, L2, S1_ff, S2_ff, D1_r, D1_f, D2_r, D2_f):
        size_f = 1.0 / float(self.model.module.backbone.star_size)

        # Cycle
        cycle_loss = self.loss['cycle_loss'](S1, S1_ff) + self.loss['cycle_loss'](S2, S2_ff)
        idt_loss = self.loss['cycle_loss'](S1, S1_i) + self.loss['cycle_loss'](S2, S2_i)

        disc_loss = self.loss['disc_loss'](D1_f) + self.loss['disc_loss'](D2_f)  * size_f\
                    - self.loss['disc_loss'](D1_r) - self.loss['disc_loss'](D2_r) * size_f 

        gen_loss = -self.loss['disc_loss'](D1_f) - self.loss['disc_loss'](D2_f) * size_f \
                   + cycle_loss * self.alpha + idt_loss * self.alpha 

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


    def _init_data_optimizer(self, optimizer, batch_size, shuffle, num_workers, collate_fn, lr, lr_scheduler):
        # TODO pin memory again with pytorch 1.10 otherwise stupid warnings
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2)
        G_optimizer = optimizer(list(self.model.module.backbone.G_S1_S2.parameters()), lr=lr)
        D_optimizer = optimizer(list(self.model.module.backbone.D1.parameters()), lr=lr)
        lr_scheduler = lr_scheduler(G_optimizer)

        return train_loader, {'G_opt': G_optimizer, 'D_opt': D_optimizer}, lr_scheduler

class SimPseudoSupervisor(Supervisor):
    def __init__(self, model, dataset=None, loss=nn.CrossEntropyLoss(reduction='mean'), t=0.0, T=1.0, soft=False, cutmix=None, localization=None, stain_normalizer=None, 
                        server_path=None, stain_restainer=None, style_noise=.0, ssl_type='pseudo',
                        lambda_cl=0.1,
                        fast_only=False,
                        slow_only=False):
        super().__init__(model,dataset,loss)
        self.t = t
        self.T = T
        self.soft = soft
        self.cutmix = cutmix
        self.localization = localization
        self.style_noise = style_noise
        self.ssl_type = ssl_type
        self.lambda_cl = lambda_cl

        self.fast_only = fast_only
        self.slow_only = slow_only

        print( "Style Noise", self.style_noise,"-", "SSL", self.lambda_cl > 0.0, "-" "Cutmix", self.cutmix)
        if self.lambda_cl > 0.0:
            print("SSL Thres.", self.t, "-", "Temp.", self.T, "-",  "SSL Type", self.ssl_type)
        self.stain_restainer = stain_restainer
        if stain_restainer is not None:
            self.stain_restainer_type = stain_restainer['type']
            if self.stain_restainer_type == 'bottle':
                self.stain_restainer_G = stain_restainer['G']
                self.embeddings = stain_restainer['embeddings']
                self.num_embeddings = stain_restainer['num_embeddings']

            self.stain_restainer_G.eval()
        
        self.stain_normalizer = stain_normalizer
        if self.stain_normalizer is not None:
            self.stain_normalizer_type = stain_normalizer['type']
            if self.stain_normalizer_type == 'gan':
                self.embeddings_norm = stain_normalizer['embeddings']
                self.stain_normalizer_G = stain_restainer['G']
            self.stain_normalizer_G.eval()

        if self.soft:
            self.cl = FocalLoss(soft=True)
        else:
            self.cl = FocalLoss()

        if server_path is not None:
            self.server_model = nn.DataParallel(torch.load(server_path).to('cuda'))
            print(bcolors.OKBLUE + "Loaded Server Predictor", server_path + "." + bcolors.ENDC)
        else:
            self.server_model = None

    
    def _cutmix(self, weak_aug, strong_aug, max_h_ratio, max_w_ratio, min_h_ratio, min_w_ratio, p):
        if  np.random.uniform() < p:
            # get boundaries
            max_h = weak_aug.shape[2]
            max_w = weak_aug.shape[3]

            h_ratio = min_h_ratio + np.random.uniform(0,1,1) * (max_h_ratio - min_h_ratio)
            w_ratio = min_w_ratio + np.random.uniform(0,1,1) * (max_w_ratio - min_w_ratio)

            h = int(max_h * h_ratio)
            w = int(max_w * w_ratio)

            max_h -= h
            max_w -= w

            # get x and y
            x = random.randint(0, max_h)
            y = random.randint(0, max_w)

            # permute tensors
            perm = torch.randperm(weak_aug.shape[0])

            weak_aug_p = weak_aug[perm, :,:,:]
            strong_aug_p = strong_aug[perm, :,:,:]

            # Mix
            weak_aug[:,:,x:x+h,y:y+w] =  weak_aug_p[:,:,x:x+h,y:y+w]
            strong_aug[:,:,x:x+h,y:y+w] =  strong_aug_p[:,:,x:x+h,y:y+w]

        return weak_aug, strong_aug


    def _forward(self, data):
        weak_aug, strong_aug, img, labels = data
        labels = labels.to('cuda')
        img = img[:img.shape[0] // 2]
        labels = labels[:labels.shape[0] // 2]
        
        weak_aug = weak_aug[:weak_aug.shape[0] // 2]
        strong_aug = strong_aug[:strong_aug.shape[0] // 2]  
        
        if self.cutmix is not None:
            weak_aug, strong_aug = self._cutmix(weak_aug, strong_aug, self.cutmix['max_h_ratio'], self.cutmix['max_w_ratio'], self.cutmix['min_h_ratio'], self.cutmix['min_w_ratio'], self.cutmix['p'])

        if self.ssl_type == 'augs_af_c':
            diff = weak_aug.to('cuda') - strong_aug.to('cuda')
        elif self.ssl_type == 'staining':
            strong_aug = weak_aug
            

        if self.stain_normalizer is not None:
            with torch.no_grad():
                img = torch.cat((img.cpu(), weak_aug, strong_aug), dim=0)
                T1 = self.embeddings_norm(torch.zeros(img.shape[0], dtype=int))
                noise = np.random.randn(T1.shape[0], T1.shape[1]) * float(self.style_noise)
                img = self.stain_normalizer_G(img, T1 + torch.from_numpy(noise).float(), None, None)
                img, weak_aug, strong_aug = img[:img.shape[0] // 3], img[img.shape[0] // 3: img.shape[0] // 3 * 2], img[img.shape[0] // 3 * 2:]


        if self.stain_restainer is not None:
            with torch.no_grad():
                if self.stain_restainer_type == 'bottle':
                    
                    L1 = torch.randint(0, self.num_embeddings, (img.shape[0],)).repeat(3)
                    T1 = self.embeddings[0](L1)
                    img = torch.cat((img, weak_aug, strong_aug), dim=0)     

                    noise = np.random.randn(T1.shape[0], T1.shape[1]) * float(self.style_noise)
                    img = self.stain_restainer_G(img, T1 + torch.from_numpy(noise).float(), None, None)
                    img, weak_aug, strong_aug = img[:img.shape[0] // 3], img[img.shape[0] // 3: img.shape[0] // 3 * 2], img[img.shape[0] // 3 * 2:]

            if self.ssl_type == 'augs_af_c':
                strong_aug = diff.to('cuda') - weak_aug.to('cuda')
            elif self.ssl_type == 'pseudo':
                strong_aug = weak_aug
            elif self.ssl_type == 'staining':
                pass
            
        if self.lambda_cl > 0:
            with torch.no_grad():
                #self.model.module.backbone.inference = True
                if self.server_model is not None:
                    if isinstance(self.server_model.module.backbone, UNetBE):
                        selected_members = np.random.choice(self.server_model.module.backbone.members, 1)
                        server_model = copy.deepcopy(self.server_model)
                        server_model.module.backbone.samples = 1
                        for module in server_model.module.backbone.modules():
                            try:
                                module.r.data = module.r.data[selected_members:selected_members+1] #.data.mean(0, keepdim=True)
                                module.s.data = module.s.data[selected_members:selected_members+1]#data[selected_members]#data.mean(0, keepdim=True)

                                module.members = 1
                            except Exception as e:
                                pass
                        pseudo_labels = server_model(weak_aug) / self.T
                    else:
                        pseudo_labels = self.server_model(weak_aug) / self.T
                else:
                    pseudo_labels = self.model(weak_aug) / self.T

                mask = (torch.max(torch.softmax(pseudo_labels, dim=1), dim=1)[0] >= self.t).reshape(-1)
                pseudo_labels = pseudo_labels.permute(0,2,3,1).reshape(-1, pseudo_labels.shape[1])[mask]
                #self.model.module.backbone.inference = False

            # LOCALIZATION
            if self.localization is not None:
                
                class Aggregation(nn.Module):
                    def __init__(self, model, reduction='mean'):
                        super().__init__()
                        self.model = model
                        self.reduction = reduction
                    
                    def forward(self, x):
                        x = self.model(x)
                        if self.reduction == 'mean':
                            return x.reshape(x.shape[0], x.shape[1], -1).mean(-1)
                        elif self.reduction == 'sum':
                            return x.reshape(x.shape[0], x.shape[1], -1).sum(-1)
                
                model_c = copy.deepcopy(self.model.module)
                self.cam = GradCAM(model=Aggregation(model_c, self.localization['reduction']), target_layer=model_c.backbone.up3, use_cuda=True)
                
                cam_labels = torch.zeros(weak_aug.shape[0], 2, *weak_aug.shape[2:], device='cuda')
                for c in range(2):
                    cam = torch.from_numpy(self.cam(input_tensor=weak_aug, target_category=int(c)))
                    cam_labels[:,c] = cam

                cam_labels = cam_labels.permute(0,2,3,1).reshape(-1, cam_labels.shape[1])[mask]
                #print(cam_labels)
                with torch.no_grad():
                    l_ = 0.5
                    pseudo_labels = (l_ * torch.softmax(pseudo_labels, dim=1) + (1-l_) * torch.softmax(cam_labels / self.T, dim=1))
            
            elif self.localization is None and self.soft:
                pass
            
            elif not self.soft:
                pseudo_labels = torch.argmax(pseudo_labels, dim=1)
                
            pred_u = self.model(strong_aug)
            pred_u = pred_u.permute(0,2,3,1).reshape(-1, pred_u.shape[1])[mask]
            
            loss = self.cl(pred_u, pseudo_labels) * self.lambda_cl

            pred_l = self.model(img)
            pred_l = pred_l.permute(0,2,3,1).reshape(-1, pred_l.shape[1])
            labels = labels.reshape(-1)

            mask = labels != -1

            pred_l = pred_l[mask]
            labels = labels[mask]

            loss = {'super':self.loss(pred_l, labels), 'semi': loss}

            return loss
        else:

            pred_l = self.model(img)
            pred_l = pred_l.permute(0,2,3,1).reshape(-1, pred_l.shape[1])
            labels = labels.reshape(-1)

            mask = labels != -1

            pred_l = pred_l[mask]
            labels = labels[mask]

            loss = self.loss(pred_l, labels)

            return {'super':self.loss(pred_l, labels), 'semi': torch.zeros(1, device=labels.device)}

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
                loss_sum += loss['super'].item() + loss['semi'].item()
                tkb.set_postfix(loss='{:3f}'.format(
                    loss_sum / (batch_id+1)))
                tkb.update(1)

                self._update(loss=loss, optimizer=optimizer, 
                             lr_scheduler=lr_scheduler)
            tkb.reset()
        del tkb

    def _update(self, loss, optimizer, lr_scheduler):
        """Backward part of training step. Calculates gradients and conducts optimization step.
        Also handles other updates like lr scheduler.

        Args:
            loss (torch.nn.Module, optional): The critierion to train on.
            lr_scheduler (torch.optim._LRScheduler, optional): Optional learning rate scheduler.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use.
        """
        if not self.fast_only and not self.slow_only:
            (loss['super'] + loss['semi']).backward()
        
        elif self.fast_only:
            loss['super'].backward(retain_graph=True)
            for name, param in self.model.module.backbone.named_parameters():
                key_split = name.split('.')[-1]
                if not key_split == 'r' and not key_split == 's':
                    param.requires_grad = False
                    
            loss['semi'].backward()#

            for name, param in self.model.module.backbone.named_parameters():
                key_split = name.split('.')[-1]
                if not key_split == 'r' and not key_split == 's':
                    param.requires_grad = True
        
        elif self.slow_only:
            loss['super'].backward(retain_graph=True)
            for name, param in self.model.module.backbone.named_parameters():
                key_split = name.split('.')[-1]
                if key_split == 'r' or key_split == 's':
                    param.requires_grad = False
                    
            loss['semi'].backward()#

            for name, param in self.model.module.backbone.named_parameters():
                key_split = name.split('.')[-1]
                if key_split == 'r' or key_split == 's':
                    param.requires_grad = True
        optimizer.step()
        lr_scheduler.step()

class StainSupervisor(CycleSupervisor):
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
