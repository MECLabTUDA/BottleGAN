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

class MultiStainTensor(Dataset):
    def __init__(self, folder_path, imgs_per_class=1200, img_size=114, batch_size=36, augs=None,):   
        self.classes = find_classes(folder_path)[0]
        self.imgs_per_class = imgs_per_class
        self.data = torch.zeros(len(self.classes), imgs_per_class, 3, img_size, img_size)
        self.has_labels = False
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.augs = augs

        for c_id, c in enumerate(self.classes):
            for b_id in range(imgs_per_class // batch_size):
                load_path = folder_path + str(c) + '/' + str(b_id) + '.pt'
                self.data[c_id, (b_id) * batch_size : (b_id + 1) * batch_size ] = torch.load(load_path)
        
    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]
    
    def __getitem__(self, index):
        label_id = random.randint(0, len(self.classes) - 1)
        img_id = random.randint(0, self.imgs_per_class - 1)
        if self.augs is None:
            return [(self.data[label_id][img_id], label_id)]
        return [(self.to_tensor(self.augs(self.to_pil(self.data[label_id][img_id]))), label_id)]

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

class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2, aug1=None, aug2=None) -> None:
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.aug1 = aug1
        self.aug2 = aug2
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return min(len(self.dataset1) if self.dataset1 is not None else len(self.dataset2), len(self.dataset2))
    
    def __getitem__(self, index):
        if self.dataset1 is not None:
            i1, _ = self.dataset1[index]
        else:
            i1 = None
        i2, l2 = self.dataset2[index]

        if i1 is None:
            return 0,0,i2, l2
        return self.to_tensor(self.aug1(self.to_pil(i1))) if self.aug1 is not None and i1 is not None else i1, self.to_tensor(self.aug2(self.to_pil(i1))), \
                     i2, l2

class TensorDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        self.to_pil = transforms.ToPILImage()
        
    def __getitem__(self, index):
        return (self.tensors[0][index]), self.tensors[1][index] 

    def __len__(self):
        return self.tensors[0].size(0)

class PairedDatasetLT(Dataset):
    def __init__(self, dataset1, dataset2, aug1=None, aug2=None) -> None:
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.aug1 = aug1
        self.aug2 = aug2
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return min(len(self.dataset1) if self.dataset1 is not None else len(self.dataset2), len(self.dataset2))
    
    def __getitem__(self, index):
        if self.dataset1 is not None:
            i1, _ = self.dataset1[index]
        else:   
            i1 = None

        i2, l2 = self.dataset2[index]
        i2, l2 = self.aug2(self.to_pil(i2), labels=l2)

        if i1 is None:
            return 0,0, self.to_tensor(i2), l2.long()
        return i1, self.to_tensor(self.aug1(self.to_pil(i1))), \
            self.to_tensor(i2), l2.long()

class MultiAugSingleImg(Dataset):
    def __init__(self, dataset, aug, n_augs=1, return_wsi_name=False):
        self.dataset = dataset
        self.return_wsi_name = return_wsi_name
        self.dataset.return_wsi_name = return_wsi_name
        self.aug = aug
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.n_augs = n_augs
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if self.return_wsi_name:
            i_org, l_org, wsi_name = self.dataset[index]
        else:
            i_org, l_org = self.dataset[index]

        i_l = [self.aug(self.to_pil(i_org), labels=l_org) for _ in range(self.n_augs)]
        i_org, l_org = self.aug(self.to_pil(i_org), labels=l_org)
        i_l = [(self.to_tensor(i_org), l_org.long())] + [(self.to_tensor(i), 0) for (i,_) in i_l]

        if self.return_wsi_name:
            return i_l, wsi_name
        return i_l

def store_valid_region_list(img_files, label_files, store_file, w=None, h=None, regions_per_file=5, n_labels=3, min_share=0.00, max_share_background=0.1, append=False, luad=False, bcss=False, bach=False):
    stats = np.zeros((regions_per_file, n_labels))
    with open(store_file, 'w' if not append else 'a') as f:
        for img_file, label_file in zip(img_files,label_files):
            f.write('+++ ' + img_file + ' ' + label_file + '\n')
            shape = read_tif_shapes(label_file, level=0)    

            for region_id in range(regions_per_file):
                found = False
                while not found:
                    range_x = shape[0] - h - 1
                    range_y = shape[1] - w - 1
                    x, y = random.randint(0, range_x), random.randint(0, range_y)

                    labels = torch.from_numpy(read_tif_region(label_file, from_x=x , to_x=x + h, from_y=y, to_y=y + w, level=0))

                    unique_labels, count_labels = torch.unique(labels, return_counts=True)
                    unique_labels = unique_labels.cpu().numpy().astype(np.uint8)
                    count_labels = count_labels / torch.sum(count_labels)
                    count_bool = torch.all(count_labels >= min_share)
                    if count_bool and (count_labels[0] < max_share_background):
                        f.write(str(x) + ' ' + str(y) + ' ' + str(h) + ' ' + str(w) + '\n')
                        found = True        
                        for l_id, l in enumerate(unique_labels):
                            stats[region_id, l] += count_labels[l_id]


        stats = np.round_(stats / len(img_files), decimals=3)
        np.savetxt(store_file[:-4] + '_stats.txt', stats)

def read_region_list(store_file, n_files, n_regions_per_file, staining=None):

    with open(store_file,'r') as fin:
        lines = fin.read().splitlines() 

    img_files = []
    label_files = []
    regions = []

    def list_loop(s=None):
        file_counter = 0
        for line in lines:
            split_line = line.split(' ')
            if line[:3] == '+++':
                if n_files > 0 and file_counter == n_files:
                    pass
                else:
                    img_files.append(str(split_line[1]).replace('small', s) if s is not None else str(split_line[1]))
                    label_files.append(str(split_line[2]))
                    regions.append([])
                    region_counter = 0
                    file_counter += 1
            else:
                if (n_regions_per_file > 0 and region_counter == n_regions_per_file) or n_regions_per_file == -1:
                    pass
                else:
                    regions[-1].append([int(elm) for elm in  split_line])          
                    region_counter += 1

    # None the less read also all image files for SSL
    all_img_files = []
    if staining is None:
        list_loop()
        for line in lines:
            split_line = line.split(' ')
            if line[:3] == '+++':
                all_img_files.append(str(split_line[1]))
            else:
                pass
    else:
        for s in staining:
          list_loop(s)
          for line in lines:
            split_line = line.split(' ')
            if line[:3] == '+++':
                all_img_files.append(str(split_line[1]).replace('small', s) if s is not None else str(split_line[1]))
            else:
                pass

    if len(regions[-1]) == 0:
        regions = None

    return img_files, label_files, regions, all_img_files

def print_labels_pred(labels, pred, org):
    labels_np = labels.cpu().numpy()
    labels_img = np.zeros((labels_np.shape[0], labels_np.shape[1], 3))
    labels_img[labels_np == 0] = np.array([1,0,0]) 
    labels_img[labels_np == 1] = np.array([0,1,0]) 
    labels_img[labels_np == 2] = np.array([0,0,1]) 

    org_np = org.permute(1,2,0).cpu().numpy()
    pred_np = torch.argmax(pred, dim=0).cpu().numpy()

    pred_img = np.zeros((pred_np.shape[0], pred_np.shape[1], 3))
    pred_img[pred_np == 0] = np.array([1,0,0]) 
    pred_img[pred_np == 1] = np.array([0,1,0]) 
    pred_img[pred_np == 2] = np.array([0,0,1]) 

    plt.imshow(org_np)
    plt.imshow(np.asarray(labels_img), alpha=0.4)
    plt.savefig('imgs/labels_slide.png')

    plt.imshow(org_np)
    plt.imshow(np.asarray(pred_img), alpha=0.4)
    plt.savefig('imgs/pred_slide.png')

def visualize(file, label_file, store_file, from_x, to_x, from_y, to_y, level):
    img = read_tif_region(file, from_x, to_x, from_y, to_y, level)
    if label_file is not None:
        labels = read_tif_region(label_file, from_x, to_x, from_y, to_y, level)

    plt.imshow(img)
    if label_file is not None:
        plt.imshow(labels, alpha=0.4)
    plt.savefig(store_file, dpi=600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.dataset == 'peso':
        img_files = glob.glob('../patho_data/peso/*_small.tif', recursive=True)
        label_files =  [file.replace('_small', '_labels')  for file in img_files if file.replace('_small', '_labels') in glob.glob('../patho_data/peso/*_labels.tif', recursive=True)]      

        img_files = sorted(img_files)
        label_files = sorted(label_files)

        all_stained_files = glob.glob('../patho_data/peso/*_S*.tif', recursive=True)
        schemes = sorted(list(set([file[-7:] for file in all_stained_files])))
        schemes = [scheme for scheme in schemes if scheme not in ['S01.tif', 'S04.tif', 'S05.tif', 'S08.tif']]
        exclude = [[40,12,11,36], [40,35,10,14], [36,38,11,14]]

        # ++++++++++++++++++++++++++++++++++++++
        # Clients
        for j, scheme in enumerate(schemes):
            all_with_scheme = glob.glob('../patho_data/peso/*_' + scheme, recursive=True)
            for i, i_n in enumerate(['one', 'two', 'three']):
                filtered = [file for file in all_with_scheme if not any('pds_' + str(s) in file for s in exclude[i])]
                random.shuffle(filtered)
                store_valid_region_list(img_files=filtered[:1], label_files=[file.replace(scheme, 'labels.tif') for file in filtered[:1]], 
                                        store_file='configs/dataset/peso/' + i_n +'/clients/' + str(j) + '_train.txt',
                                        regions_per_file=15, n_labels=3,
                                        h=300, w=300)

                store_valid_region_list(img_files=filtered[2:3], label_files=[file.replace(scheme, 'labels.tif') for file in filtered[2:3]], 
                                        store_file='configs/dataset/peso/' + i_n +'/clients/' + str(j) + '_test.txt',
                                        regions_per_file=0, n_labels=3,
                                        h=300, w=300)
            
                store_valid_region_list(img_files=filtered[1:2], label_files=[file.replace(scheme, 'labels.tif') for file in filtered[1:2]], 
                                        store_file='configs/dataset/peso/' + i_n +'/clients/' + str(j) + '_val.txt',
                                        regions_per_file=0, n_labels=3,
                                        h=300, w=300)

        # ++++++++++++++++++++++++++++++++++++++
        # Server
        all_stained_files = glob.glob('../patho_data/peso/*_small.tif', recursive=True)
        scheme = 'small.tif'
        for i, i_n in enumerate(['one', 'two', 'three']):
            filtered = [file for file in all_stained_files if not any('pds_' + str(s) in file for s in exclude[i])]
            random.shuffle(filtered)
            store_valid_region_list(img_files=filtered[:5], label_files=[file.replace(scheme, 'labels.tif') for file in filtered[:5]], 
                                    store_file='configs/dataset/peso/' + i_n +'/server/' +  'train.txt',
                                    regions_per_file=20, n_labels=3,
                                    h=300, w=300)

            store_valid_region_list(img_files=filtered[5:6], label_files=[file.replace(scheme, 'labels.tif') for file in filtered[5:6]], 
                                    store_file='configs/dataset/peso/' + i_n +'/server/' + 'test.txt',
                                    regions_per_file=0, n_labels=3,
                                        h=300, w=300)

        for i, i_n in enumerate(['one', 'two', 'three']):            
            with open('configs/dataset/peso/' + i_n +'/clients/' + 'all' + '_test.txt', 'w') as f:
                for j, scheme in enumerate(schemes):
                    store_file='configs/dataset/peso/' + i_n +'/clients/' + str(j) + '_test.txt'
                    with open(store_file,'r') as fin:
                        lines = fin.read()
                        f.write(lines)

        for i, i_n in enumerate(['one', 'two', 'three']):            
            with open('configs/dataset/peso/' + i_n +'/clients/' + 'all' + '_val.txt', 'w') as f:
                for j, scheme in enumerate(schemes):
                    store_file='configs/dataset/peso/' + i_n +'/clients/' + str(j) + '_val.txt'
                    with open(store_file,'r') as fin:
                        lines = fin.read()
                        f.write(lines)
