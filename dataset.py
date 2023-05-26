import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import argparse 
import glob
import tifffile as tiff
import zarr
import random
import matplotlib.pyplot as plt
from super_selfish.utils import bcolors
from torchvision.datasets.folder import find_classes

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Single Region Dataset used to load and sample from a SINGLE REGION of a TIF
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class PairedDatasetST(Dataset):
    def __init__(self, dataset1, dataset2, aug1=None, aug2=None) -> None:
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.aug1 = aug1
        self.aug2 = aug2
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2))
    
    def __getitem__(self, index):
        il2 = self.dataset2[index]
        if self.dataset1.has_labels and self.dataset1.return_labels:
            illl1= self.dataset1[index]
            return illl1 + il2
        else:
            il1 = self.dataset1[index]
        return il1 + il2

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Single Region Dataset used to load and sample from a SINGLE REGION of a TIF
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class MultiStainMultiFile(Dataset):
    def __init__(self, img_files, label_files=None, transform=None, length=1, h=225, w=225, max_tries=30, verbose=0, min_non_white=0.0, n_examples=1, augs=None, **kwargs):   
        img_files = sorted(img_files)
        self.img_files = img_files
        self.n_examples = n_examples
        if label_files is None:
            self.files = [MultiFileMultiRegion(img_files=files, length=length, transform=transform, h=h, w=w, verbose=verbose, min_non_white=min_non_white) for files in img_files]
            self.has_labels = False
        else:
            self.files = [MultiFileMultiRegion(img_files=files_i, label_files=files_l, length=length, transform=transform, h=h, w=w, verbose=verbose, min_non_white=min_non_white) for files_i, files_l in zip(img_files, label_files)]
            self.has_labels = True

        self.len = length
        self.act_len = len(self.files)
        self.return_labels = False
        self.n_examples = n_examples
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.augs = augs

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        label_id = random.randint(0, self.act_len - 1)
        if self.has_labels and self.return_labels:
            data, labels = self.files[label_id][index]
            return [(data, labels, label_id)]
        if self.augs is None:
            return [(self.files[label_id][index][0], label_id) for _ in range(self.n_examples)]
        else:
            return [(self.to_tensor(self.augs(self.to_pil(self.files[label_id][index][0]))), label_id) for _ in range(self.n_examples)]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Consider this a Standard Dataset:
# Multi Region Dataset used to load and sample from MUTLIPLE REGIONs of MULTIPLE FILES
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
class MultiFileMultiRegion(Dataset):
    def __init__(self, img_files, label_files=None, regions=None, transform=None, length=1, h=225, w=225, max_tries=30, verbose=0, return_wsi_name=False, dataset_name='peso', min_non_white=0.0, clean_img_files=None,
                    context=None, **kwargs):
        # Source WSIs
        self.img_files = img_files
        self.label_files = label_files
        self.clean_img_files = clean_img_files
        self.dataset_name = dataset_name
        self.fix_index = False
        self.h = h
        self.w = w
        self.context = context

        # Regions
        self.regions = []
        for i, img_file in enumerate(self.img_files):
            dataset_name = img_file.split('/')[2]
            if regions is not None:
                for region in regions[i]:
                
                    start_x = region[0]
                    start_y = region[1]
                    max_x_range = region[2]
                    max_y_range = region[3]
                    self.regions.append(SingleRegion([img_file], [self.label_files[i]] if self.label_files is not None else None,  clean_img_files=[self.clean_img_files[i]] if self.clean_img_files is not None else None, 
                                                    start_x=start_x, start_y=start_y, max_x_range=max_x_range, max_y_range=max_y_range,
                                                    h=h, w=w, max_tries=max_tries, verbose=verbose, transform=transform, len_mult=1, dataset_name=dataset_name, min_non_white=min_non_white, context=context, **kwargs))
            else:
                start_x = 0
                start_y = 0
                max_x_range = 'None'
                max_y_range = 'None'
                self.regions.append(SingleRegion([img_file], [self.label_files[i]] if self.label_files is not None else None,   clean_img_files=[self.clean_img_files[i]] if self.clean_img_files is not None else None, 
                                                    start_x=start_x, start_y=start_y, max_x_range=max_x_range, max_y_range=max_y_range,
                                                    h=h, w=w, max_tries=max_tries, verbose=verbose, transform=transform, len_mult=1, dataset_name=dataset_name, min_non_white=min_non_white, context=context, **kwargs))

        # Len
        self.act_len = len(self.regions)
        self.len = length

        # Return Name
        self.return_wsi_name = return_wsi_name

        #print(img_files)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.context is not None:
            region_id = self.context.randint(0, self.act_len - 1)
        else:
            region_id = random.randint(0, self.act_len - 1)
        if self.fix_index:
            region_id = index %  240
        if self.return_wsi_name:
            return self.regions[region_id][index] + (self.regions[region_id].img_files[0],)
        return self.regions[region_id][index]
    
    def print_regions(self, store_folder):
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)

        for region in self.regions:
            if region.max_x_range != 'None':
                visualize(file=region.img_files[0], label_file=region.label_files[0] if region.label_files is not None else None,
                store_file=os.path.join(store_folder, region.img_files[0].split('/')[-1][:-4] + '_x' + str(region.start_x) + '_y' + str(region.start_y) + '.png'), 
                from_x=region.start_x,to_x=region.start_x+region.max_x_range, from_y=region.start_y, to_y=region.start_y + region.max_y_range,
                level=0)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Single Region Dataset used to load and sample from a SINGLE REGION of a TIF
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
class SingleRegion(Dataset):
    def __init__(self, img_files, label_files=None, transform=None, len_mult=1, h=225,w=225, max_tries=15,
                    start_x=0, start_y=0, max_x_range='None', max_y_range='None', verbose=0, dataset_name='peso', min_non_white=0.0, clean_img_files=None, context=None,  **kwargs):
        # ['pds_6_HE.tif', 'pds_35_HE.tif', 'pds_38_HE.tif', 'pds_39_HE.tif', 'pds_40_HE.tif']
        # Sources
        self.img_files = img_files
        self.label_files = label_files
        self.clean_img_files = clean_img_files
        self.dataset_name = dataset_name

        # Len
        self.act_len = len(self.img_files)
        self.len = self.act_len * len_mult

        # Image size
        self.h = h
        self.w = w

        # Crop WSIs
        self.start_x = start_x
        self.start_y = start_y
        self.max_x_range = max_x_range
        self.max_y_range = max_y_range

        # Transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.to_gray = torch.tensor([[[.2126, .7152, .0722]]], device='cpu')
        self.transform = transform
        self.max_tries = max_tries
        self.min_non_white = min_non_white

        self.context =  context

        # Print files loaded
        if verbose > 0:
            print(bcolors.OKBLUE + "Loaded dataset files: " + bcolors.ENDC + str(self.img_files))

        # Return type
        self.return_xys = False

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.context is not None:
            slide_id = self.context.randint(0, self.act_len - 1)
        else:
            slide_id = random.randint(0, self.act_len - 1)

        shapes = read_tif_shapes(self.img_files[slide_id], level=0)

        # Get non empty crop
        tries = 0
        while tries < self.max_tries:
            range_x = min(self.max_x_range + self.start_x - self.h - 1, shapes[0] - self.h - 1) if self.max_x_range != 'None' else (shapes[0] - self.h - 1)
            range_y = min(self.max_y_range + self.start_y - self.w - 1, shapes[1] - self.w - 1) if self.max_y_range != 'None' else (shapes[1] - self.w - 1)

            rand_x_start = random.randint(self.start_x, range_x) if self.context is None else self.context.randint(self.start_x, range_x)
            rand_y_start = random.randint(self.start_y, range_y) if self.context is None else self.context.randint(self.start_y, range_y)
            x, y = self.start_x if self.start_x >= range_x else rand_x_start,  self.start_y if self.start_y >= range_y else rand_y_start

            img_t = torch.from_numpy(read_tif_region(self.img_files[slide_id], from_x=x , to_x=x + self.h, from_y=y, to_y=y + self.w, level=0)/ 255).permute(2,0,1)
            img = self.to_pil(img_t)

            if torch.unique(img_t).shape[0] == 1:
                continue

            if self.clean_img_files is not None:
                img_t = torch.from_numpy(read_tif_region(self.clean_img_files[slide_id], from_x=x , to_x=x + self.h, from_y=y, to_y=y + self.w, level=0)/ 255).permute(2,0,1)
                img_c = self.to_pil(img_t)    
            else:
                img_c = None

            img_g = torch.sum(img_t.permute(1,2,0) * self.to_gray, dim=-1)

            # Ignore white
            if torch.any(img_g < 0.8) and (torch.sum(img_g < 0.9) >= (img_g.view(-1).shape[0] * self.min_non_white)):
                break
            
            tries +=1

        # Get labels
        if self.label_files is not None:
            labels = read_tif_region(self.label_files[slide_id], from_x=x , to_x=x + self.h, from_y=y, to_y=y + self.w, level=0)
            labels = torch.from_numpy(labels.copy()).long().squeeze(-1)
        else:
            labels = torch.zeros(1)            

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            if img_c is not None:
                img_c = self.transform(img_c)

        # Clean up useless class for peso
        if self.dataset_name == 'peso':
            labels[labels==0] = 1 
            labels -= 1

        # Form binary task for luaad
        if self.dataset_name == 'luad':
            labels *= -1
            labels += 3
            labels[labels==0] = 1
            labels -= 1
            labels[labels==2] = 1

        # Form binary task for bcss
        if self.dataset_name == 'bcss':
            labels[((labels == 0) | ((labels == 7) | (labels == 15)))] = -1
            labels[labels > 1] = 0

        if self.dataset_name == 'bach':
            labels[labels > 0] = 1

        # Return
        if not self.return_xys and img_c is None:     
            return img, labels.long()
        elif not self.return_xys and img_c is not None:
            return img, img_c, labels.long()
        else:
            return img, labels.long(), torch.tensor([x,y, slide_id])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Helpers/Actual Loaders
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
def read_tif_region(file, from_x=None, to_x=None, from_y=None, to_y=None, level=0):
    store = tiff.imread(file, aszarr=True)
    out_labels_slide = zarr.open(store, mode='r')

    if isinstance(out_labels_slide, zarr.hierarchy.Group):
        out_labels_slide = out_labels_slide[level]

    if from_x is None or to_x is None or from_y is None or to_y is None:
        return out_labels_slide
    else:
        return out_labels_slide[from_x:to_x, from_y:to_y]

def read_tif_shapes(file, level=0):
    store = tiff.imread(file, aszarr=True)
    out_labels_slide = zarr.open(store, mode='r')

    if isinstance(out_labels_slide, zarr.hierarchy.Group):
        out_labels_slide = out_labels_slide[level]
    
    return out_labels_slide.shape

def visualize(file, label_file, store_file, from_x, to_x, from_y, to_y, level):
    img = read_tif_region(file, from_x, to_x, from_y, to_y, level)
    if label_file is not None:
        labels = read_tif_region(label_file, from_x, to_x, from_y, to_y, level)

    plt.imshow(img)
    if label_file is not None:
        plt.imshow(labels, alpha=0.4)
    plt.savefig(store_file, dpi=600)

