from tqdm import tqdm
from colorama import Fore
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def classification_loss(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum()
    return correct / total


def test(model, test_dataset, loss_f=classification_loss, batch_size=48, shuffle=False, num_workers=0, collate_fn=None):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    tkb = tqdm(total=int(len(test_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
        Fore.GREEN, Fore.RESET), desc="Test Accuracy ")

    loss_sum = 0
    for batch_id, data in enumerate(test_loader):
        if data[0].shape[0] != test_loader.batch_size:
            continue

        inputs, labels = data
        with torch.no_grad():
            outputs = model(inputs.to('cuda'))
            loss = loss_f(outputs, labels.to('cuda'))
            loss_sum += loss.item()
        tkb.set_postfix(Accuracy='{:3f}'.format(
            loss_sum / (batch_id+1)))
        tkb.update(1)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean', soft=False):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        if soft:
            self.loss_f = SoftCrossEntropy()
        else:
            self.loss_f = F.cross_entropy

    def forward(self, input, target):
        ce_loss = self.loss_f(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, reduction, weight):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        softtargets = torch.softmax(target, dim=1)

        vals = -(softtargets * logprobs).sum(1)

        if reduction == 'mean':
            return vals.mean()
        elif reduction == 'sum':
            return vals.sum()

class MaskedFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super().__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        mask = target != 0
        ce_loss = (F.cross_entropy(input, target,reduction='none', weight=self.weight) * mask).mean()
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def entropy(logits, dim=1):
    soft = torch.softmax(logits, dim=dim)
    log_soft = torch.log(soft)
    entropy = -torch.sum(soft * log_soft, dim=dim, keepdim=True)
    return entropy


class SchedulerSE():
    def __init__(self, optimizer, members, max_iter) -> None:
        self.max_iter_per_member = max_iter // members
        self.iter = 0
        self.optimizer = optimizer
        self.initial_lr = optimizer.state_dict()["param_groups"][0]["lr"]
        self.resets = 0
        self.members = members

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.initial_lr / 2 * (math.cos(math.pi * self.iter / self.max_iter_per_member) + 1)
            self.last_lr = g['lr']
            self.iter = (self.iter + 1) % self.max_iter_per_member

            if self.iter == 0:
                reset_lr = True
                self.resets += 1

                if self.resets == self.members:
                    last_reset = True
                else:
                    last_reset = False
            
            else:
                reset_lr = False
                last_reset = False
        
        return reset_lr, last_reset


    def get_last_lr(self):
        return self.last_lr
