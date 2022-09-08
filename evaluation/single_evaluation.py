from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from colorama import Fore
import collections.abc
import numpy as np
import json
import matplotlib.pyplot as plt

class Evaluator():
    def __init__(self, metrics, model, n_classes, result_file=None, stain_normalizer=None, samples=1):
        self.metrics = metrics
        self.model = model
        self.n_classes = n_classes
        self.result_file = result_file
        self.stain_normalizer = stain_normalizer
        self.model.module.backbone.samples = samples


    def evaluate(self, dataset, batch_size, num_workers, model=None):
        if model is not None:
            self.model = model
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        tkb = [tqdm(total=int(len(test_loader)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN, Fore.RESET), position=i, desc=str(self.metrics[i][0]), leave=False) for i in range(len(self.metrics))]

        metric_values = torch.zeros((len(test_loader), len(self.metrics) * (self.n_classes + 1)))

        for batch_id, batch in enumerate(test_loader):
            batch, wsi_name = batch
            if batch[0][0].shape[0] != test_loader.batch_size:
                batch_id -= 1
                continue
            labels = batch[0][1].to('cuda')

            with torch.no_grad():

                for aug_batch_id, aug_batch in enumerate(batch):
                    
                    if self.stain_normalizer is not None:                     
                        aug_batch[0] = self.stain_normalizer(aug_batch[0])

                    if not isinstance(self.model, list):
                        if not aug_batch_id == 0:
                            prediction += self.model(aug_batch[0].to('cuda'))
                        else:
                            prediction = self.model(aug_batch[0].to('cuda'))
                    else:
                        for mb_id, mb in enumerate(self.model):
                            if not aug_batch_id == 0 and not mb_id == 0:
                                prediction += mb(aug_batch[0].to('cuda'))
                            else:
                                prediction = mb(aug_batch[0].to('cuda'))
                
                if isinstance(self.model, list):
                    prediction /= len(self.model)
                prediction /= len(batch)

            prediction = prediction.permute(0,2,3,1).reshape(-1, prediction.shape[1])
            labels = labels.reshape(-1)
            mask = labels != -1

            prediction = prediction[mask]
            labels = labels[mask]
            for m_id, (n,m) in enumerate(self.metrics):
                loss = m(prediction, labels)
                for c_id in range(0, self.n_classes + 1):
                    metric_values[batch_id, m_id * (self.n_classes + 1) + c_id] = loss[c_id]

                mean = metric_values[:batch_id+1, m_id * (self.n_classes + 1):m_id * (self.n_classes + 1) + self.n_classes + 1].mean(0)
                metric_values_batch = collections.OrderedDict([(str(c), '{:1f}'.format(mean[c].item())) for c in range(self.n_classes + 1)])
                tkb[m_id].set_postfix(metric_values_batch)
                tkb[m_id].update(1)
        
        # Hacky, but working....
        print('\n' * len(self.metrics))
        # Write to file
        if self.result_file is not None:
            out_dicts = []
            for m_id, (n,m) in enumerate(self.metrics):
                mean = metric_values[:batch_id+1, m_id * (self.n_classes + 1):m_id * (self.n_classes + 1) + self.n_classes + 1].mean(0)
                metric_values_batch = collections.OrderedDict([(str(c), '{:1f}'.format(mean[c].item())) for c in range(self.n_classes + 1)])
                out_dicts.append((n, metric_values_batch))

            metrics_json = json.dumps(collections.OrderedDict(out_dicts))
            with open(self.result_file, 'w') as f:
                f.write(metrics_json)
                
# Performance +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def accuracy(prediction, labels):
    n_classes = prediction.shape[1]
    prediction = torch.argmax(prediction, 1)
    acc = prediction.eq(labels).float()
    acc_per_class = []
    for c in range(n_classes):
        acc_per_class.append(acc[labels == c].mean())
    acc_per_class.append(acc.mean())
    return acc_per_class

def class_balance(prediction, labels):
    n_classes = prediction.shape[1]
    percentage = []
    labels_flat =  labels.reshape(-1)
    total = labels_flat.shape[0]
    for c in range(n_classes):
        percentage.append((torch.sum(labels_flat == c) / total).item())
    percentage.append(np.sum(np.array(percentage)))
    return percentage

SMOOTH = 1e-6
def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist

def jaccard_index(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + SMOOTH)

    return jaccard

def iou(prediction, labels):
    n_classes = prediction.shape[1]
    hist = _fast_hist(labels, torch.argmax(prediction, 1), n_classes)
    iou_all = jaccard_index(hist)
    
    iou_list = []
    for val in iou_all:
        iou_list.append(val.item())
    iou_list.append(torch.mean(iou_all[1:]).item())

    return iou_list
    



# Probs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ECELoss(nn.Module):
    def __init__(self, n_bins=40):

        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        n_classes = logits.shape[1]
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        ece_per_class = [0 for _ in range(n_classes)]
        ece_per_class.append(ece.item())
        return ece_per_class

def nll(prediction, labels):
    n_classes = prediction.shape[1]
    acc = F.cross_entropy(prediction, labels, reduction='none')
    acc_per_class = []
    for c in range(n_classes):
        acc_per_class.append(acc[labels == c].mean())
    acc_per_class.append(acc.mean())
    return acc_per_class


def tace(prediction, labels, n_bins=20, threshold=1e-3, **args):
    prediction = F.softmax(prediction, dim=1)
    n_classes = prediction.shape[1]
    prediction = prediction.reshape(-1, n_classes)
    labels = labels.reshape(-1)
    n_objects = prediction.shape[0]
    tace_per_class = []

    
    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = prediction[:, cur_class]
        
        targets_sorted = labels[cur_class_conf.argsort()]
        cur_class_conf_sorted = torch.sort(cur_class_conf)[0]
        
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        
        bin_size = len(cur_class_conf_sorted) // n_bins
                
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class).float()
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            avg_confidence_in_bin = torch.mean(bin_conf)
            avg_accuracy_in_bin = torch.mean(bin_acc)
            delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            # print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)

    tace_per_class = [0 for _ in range(n_classes)]
    tace_per_class.append(res)        
    return tace_per_class