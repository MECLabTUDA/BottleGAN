from modules.embedding import BottleGAN, SBNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from colorama import Fore
import collections.abc
import numpy as np
import json
import piq
from copy import deepcopy
import PIL.Image as im_
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

