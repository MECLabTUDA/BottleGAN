from os import replace
from os import listdir
from os.path import isfile, join    
from matplotlib.pyplot import fill
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import torchvision.transforms.functional as tF
import numpy as np
from scipy.ndimage import gaussian_filter, interpolation
import elasticdeform
import math
import torch
import random
from skimage.color import lab2rgb, rgb2lab
from PIL import Image as im_
from PIL import ImageOps as imo
from PIL import ImageEnhance as ime
from PIL import ImageFilter as imf
import io
from einops import rearrange
import staintools
from torchvision.transforms.transforms import ToTensor

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Library Datasets
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class AugmentationDataset(Dataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Standard dataset for contrastive algorithms. Augments each image with given transformations.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image) identifier): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image) identifier, optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image.
        """
        self.dataset = dataset
        self.trans = transformations
        self.trans2 = transformations2
        self.clean1 = clean1
        self.clean2 = clean2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = transforms.functional.to_pil_image(self.dataset[idx][0])
        
        if self.clean1:
            img1 = img
        else:
            img1 = self.trans(img)

        if self.clean2:
            img2 = img
        else:
            if self.trans2 is None:
                img2 = self.trans(img)
            else:
                img2 = self.trans2(img)

        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)

        return img1, img2

class AugmentationDatasetAndClean(Dataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Standard dataset for contrastive algorithms. Augments each image with given transformations.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image) identifier): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image) identifier, optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image.
        """
        self.dataset = dataset
        self.trans = transformations
        self.trans2 = transformations2
        self.clean1 = clean1
        self.clean2 = clean2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = transforms.functional.to_pil_image(self.dataset[idx][0])

        if self.clean1:
            img1 = img
        else:
            img1 = self.trans(img)

        if self.clean2:
            img2 = img
        else:
            if self.trans2 is None:
                img2 = self.trans(img)
            else:
                img2 = self.trans2(img)

        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)

        return img1, img2, self.dataset[idx][0]
        
class LDataset(Dataset):
    def __init__(self, dataset):
        """Extracts the L channel of an RGB image and keeps the label the same.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
        """    
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label = self.dataset[idx]
        img1 = rgb2lab(img1.permute(1, 2, 0).cpu().numpy())

        img1_l = torch.from_numpy(img1).permute(
            2, 0, 1)[0:1, :, :]

        return img1_l, label

class AugmentationIndexedDataset(AugmentationDataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Extends AugmentationDataset to return instance index.

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image)): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image), optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image and the image index.
        """
        super().__init__(dataset, transformations, transformations2, clean1, clean2)

    def __getitem__(self, idx):
        img1, img2 = super().__getitem__(idx)
        return img1, img2, idx

class AugmentationLabIndexedDataset(AugmentationIndexedDataset):
    def __init__(self, dataset, transformations, transformations2=None, clean1=False, clean2=False):
        """Extends AugmentationIndexedDataset to split images into l and ab channels

        Args:
            dataset (torch.utils.data.Dataset)): The backbone dataset.
            transformations (function(PIL.Image)): Transformations for either img1 or img1 and img2. Must return a PIL.Image.
            transformations2 (function(PIL.Image), optional): Transformations for img2. Defaults to None. Must return a PIL.Image if specified.
            clean1 (bool, optional): Wether to not augment img1. Defaults to False.
            clean2 (bool, optional): Wether to not augment img2. Defaults to False.

        Returns:
            Tuple via __getitem__: Two augmentations of an image as l and ab channels as well as the image index.
        """
        super().__init__(dataset, transformations, transformations2, clean1, clean2)

    def __getitem__(self, idx):
        img1, img2, idx = super().__getitem__(idx)
        img1, img2 = rgb2lab(img1.permute(1, 2, 0).cpu().numpy()), rgb2lab(
            img2.permute(1, 2, 0).cpu().numpy())

        img1_l, img1_ab = torch.from_numpy(img1).permute(
            2, 0, 1)[0:1, :, :], torch.from_numpy(img1).permute(2, 0, 1)[1:, :, :]

        img2_l, img2_ab = torch.from_numpy(img2).permute(
            2, 0, 1)[0:1, :, :], torch.from_numpy(img2).permute(2, 0, 1)[1:, :, :]

        return img1_l, img1_ab, img2_l, img2_ab, idx


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data utils
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def batched_collate(data):
    transposed_data = list(zip(*data))
    img = torch.cat(transposed_data[0], 0)
    labels = torch.cat(transposed_data[1], 0)
    return img, labels


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Transformations
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def jigsaw(img, perm, s, trans=lambda x: x, normed=True, crops=3):
    """Jigsaws image into crops and shuffles.

    Args:
        img (PIL.Image): Image.
        perm ([int]): Permutation to apply.
        s (int): Output size of each crop as well as size of area to crop from
        trans (function(PIL.Image) identifier, optional): Augmentations on crop. Defaults to lambdax:x.
        normed (bool, optional): Wether to norm crop. Defaults to True.

    Returns:
        PIL.Image: Output image
    """
    img_out = img.copy()
    for n in range(crops * crops):
        i = (n // crops) * s
        j = (n % crops) * s

        patch = transforms.functional.to_tensor(
            trans(img.crop(box=(i, j, i + s, j + s))))

        if normed:
            patch_mean = torch.mean(patch)
            patch_std = torch.std(patch)
            patch_std = 1 if patch_std == 0 else patch_std
            normed_patch = transforms.functional.normalize(
                patch, patch_mean, patch_std)
        else:
            normed_patch = patch

        normed_patch = transforms.functional.to_pil_image(normed_patch)

        i_out = (perm[n] // crops) * s
        j_out = (perm[n] % crops) * s

        img_out.paste(normed_patch, box=(
            i_out, j_out, i_out + s, j_out + s))
    return img_out

def elastic_transform(img, sigma, points):
    """Elastic 3x3 transform like for U-Nets.

    Args:
        img (PIL.Image): Image.
        sigma (float): Std. Dev.

    Returns:
        PIL.Image: Output image
    """
    def t1(image, sigma): return elasticdeform.deform_random_grid(
        image, axis=(0, 1), sigma=sigma, mode='reflect', points=points)

    img = transforms.functional.to_tensor(img)
    img = torch.from_numpy(t1(img.permute(1, 2, 0).cpu(
    ).numpy(), sigma=10.0)).permute(2, 0, 1)
    img = transforms.functional.to_pil_image(img)
    return img

def compress_jpeg(img, quality):
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=quality)
    output.seek(0)
    return im_.open(output)

def augment_staining(img, scheme_path='datasets/staining_schemes'):
    try:
        if scheme_path is None:
            augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.2, sigma2=0.2)
            augmentor.fit(img)
            return im_.fromarray(augmentor.pop().astype(np.uint8))
        else:

            files = [f for f in listdir(scheme_path) if isfile(join(scheme_path, f))]
            stain_file = random.choice(files)
            st = staintools.read_image(join(scheme_path, stain_file))
            stain_norm = staintools.StainNormalizer(method='macenko')
            st = staintools.LuminosityStandardizer.standardize(st)
            stain_norm.fit(st)
            img = staintools.LuminosityStandardizer.standardize(img)
            img = stain_norm.transform(img)

            return img
    except:
        return img

def transparent_overlay(back, fore, fore_alpha=0.5, 
                        pos=(0,0), rand_pos_offset=None, 
                        rescale_to_back=True, rescale_down_fore=1.0, rescale_down_fore_max=0.5):
    if rescale_to_back:
        fore = fore.resize(back.size, im_.ANTIALIAS)
    if rescale_down_fore < 1.0:
        fore = fore.resize(tuple([int(dim * random.uniform(rescale_down_fore_max, rescale_down_fore)) for dim in fore.size]), im_.ANTIALIAS)
    if rand_pos_offset is not None:
        pos = (random.randint(pos[0], pos[0] + rand_pos_offset[0] - 1), random.randint(pos[1], pos[1] + rand_pos_offset[1] - 1))


    fore_np = np.asarray(fore).copy() 
    fore_np[:,:,-1] = (fore_np[:,:,-1] * fore_alpha).astype(np.uint8)
    fore = im_.fromarray(fore_np, mode='RGBA')
    back.paste(fore, pos, fore)
    return back

def dark_spots(img, n_spots, alpha, spots_path='datasets/dark_spots'):
    files = [f for f in listdir(spots_path) if isfile(join(spots_path, f))]
    for _ in range(n_spots):
        spot_file = random.choice(files)
        s = im_.open(join(spots_path, spot_file))
        img = transparent_overlay(img, s, fore_alpha=alpha, rescale_down_fore=0.99, rand_pos_offset=(img.size[0], img.size[1]))

    return img

def threads(img, n_threads, alpha, threads_path='datasets/threads'):
    files = [f for f in listdir(threads_path) if isfile(join(threads_path, f))]
    for _ in range(n_threads):
        spot_file = random.choice(files)
        s = im_.open(join(threads_path, spot_file))
        img = transparent_overlay(img, s, fore_alpha=alpha, rescale_down_fore=0.99, rescale_down_fore_max=0.8,
                         rand_pos_offset=(img.size[0], img.size[1]))

    return img

def squamous(img, n_threads, alpha, squamous_path='datasets/squamous'):
    files = [f for f in listdir(squamous_path) if isfile(join(squamous_path, f))]
    for _ in range(n_threads):
        spot_file = random.choice(files)
        s = im_.open(join(squamous_path, spot_file))
        img = transparent_overlay(img, s, fore_alpha=alpha, rescale_to_back=False, rescale_down_fore=0.99, rescale_down_fore_max=0.8,
                         rand_pos_offset=(img.size[0], img.size[1]))

    return img

def fat(img, n_threads, alpha, fat_path='datasets/fat'):
    files = [f for f in listdir(fat_path) if isfile(join(fat_path, f))]

    for _ in range(n_threads):
        spot_file = random.choice(files)
        s = im_.open(join(fat_path, spot_file))
        img = transparent_overlay(img, s, fore_alpha=alpha, rescale_to_back=False, rescale_down_fore=0.7, rescale_down_fore_max=0.3,
                         rand_pos_offset=(img.size[0], img.size[1]))

    return img

def rotate_w_labels(img, degrees):
    img, labels = img[:3], img[3:]    
    angle = int(np.random.uniform() * degrees)
    img = tF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=1)
    labels = tF.rotate(labels, angle, fill=-1)
    return torch.cat((img, labels), dim=0)

def flip_w_labels(img):
    img, labels = img[:3], img[3:]    
    img = tF.hflip(img)
    labels = tF.hflip(labels)
    return torch.cat((img, labels), dim=0)

def elastic_w_labels(img, sigma=25, points=3):
    img, labels = img[:3], img[3:]    

    img, labels = elasticdeform.deform_random_grid(
        [img.permute(1,2,0).cpu().numpy(), labels.permute(1,2,0).cpu().numpy()], axis=[(0, 1), (0,1)], sigma=sigma, mode='reflect', points=points, order=[0,0])

    img, labels = torch.from_numpy(img).permute(2,0,1), torch.from_numpy(labels).permute(2,0,1)

    return torch.cat((img, labels), dim=0)

def cutout_w_labels(img, max_h_ratio, max_w_ratio, min_h_ratio, min_w_ratio,):
    img, labels = img[:3], img[3:]    

    # get boundaries
    max_h = img.shape[1]
    max_w = img.shape[2]

    h_ratio = min_h_ratio + np.random.uniform(0,1,1) * (max_h_ratio - min_h_ratio)
    w_ratio = min_w_ratio + np.random.uniform(0,1,1) * (max_w_ratio - min_w_ratio)

    h = int(max_h * h_ratio)
    w = int(max_w * w_ratio)

    max_h -= h
    max_w -= w

    # get x and y
    x = random.randint(0, max_h)
    y = random.randint(0, max_w)

    # Cut
    img[:,x:x+h,y:y+w] =  0.0
    labels[x:x+h,y:y+w] =  -1

    return torch.cat((img, labels), dim=0)

def PathologyAugmentations(img, cfg, labels=None):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    pool = []
    necessary_ids = []
    with_labels = []

    t_per_img = cfg['transforms_per_image']
    rand_app_p = cfg['random_apply']
    cfg = cfg['augmentations']

    # Appearence
    if 'gauss' in cfg.keys():
        if np.random.uniform() < cfg['gauss']['p']: 
            pool.append(transforms.Lambda(lambda x: x.filter(imf.GaussianBlur(
                radius=cfg['gauss']['radius']))))
            with_labels.append(False)

            if 1.0 < cfg['gauss']['p']:
                necessary_ids.append(len(pool) - 1)
    
    if 'jpeg' in cfg.keys():
        if np.random.uniform() < cfg['jpeg']['p']: 
            pool.append(transforms.Lambda(lambda x: compress_jpeg(x, cfg['jpeg']['ratio'])))
            with_labels.append(False)

            if 1.0 < cfg['jpeg']['p']:
                necessary_ids.append(len(pool) - 1)
    
    if 'brightness_contrast' in cfg.keys():
        if np.random.uniform() < cfg['brightness_contrast']['p']:
            pool.append(transforms.Lambda(lambda x: ime.Brightness(x).enhance((np.random.uniform() * 2 - 1)  * cfg['brightness_contrast']['range'] + 1))) # Brightness
            pool.append(transforms.Lambda(lambda x: ime.Contrast(x).enhance((np.random.uniform() * 2 - 1)  * cfg['brightness_contrast']['range'] + 1)))  # Contras)
            with_labels.append(False)
            with_labels.append(False)

            if 1.0 < cfg['brightness_contrast']['p']:
                necessary_ids.append(len(pool) - 1)
    
    if 'staining' in cfg.keys():
        if np.random.uniform() < cfg['staining']['p']: 
            pool.append(transforms.Lambda(lambda x: augment_staining(np.asarray(x).copy())))
            with_labels.append(False)

            if 1.0 < cfg['staining']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'dark_spots' in cfg.keys():
        if np.random.uniform() < cfg['dark_spots']['p']: 
            pool.append(transforms.Lambda(lambda x: dark_spots(x, n_spots=int(np.random.uniform(0.5, 1) * cfg['dark_spots']['n_spots']), 
                                                                    alpha=cfg['dark_spots']['alpha'])))
            with_labels.append(False)

            if 1.0 < cfg['dark_spots']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'threads' in cfg.keys():
        if np.random.uniform() < cfg['threads']['p']: 
            pool.append(transforms.Lambda(lambda x: threads(x, n_threads=int(np.random.uniform(0.5, 1) * cfg['threads']['n_threads']), 
                                                                    alpha=cfg['threads']['alpha'])))
            with_labels.append(False)

            if 1.0 < cfg['threads']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'squamous' in cfg.keys():
        if np.random.uniform() < cfg['squamous']['p']: 
            pool.append(transforms.Lambda(lambda x: squamous(x, n_threads=int(np.random.uniform(0.5, 1) * cfg['squamous']['n_threads']), 
                                                                    alpha=cfg['squamous']['alpha'])))
            with_labels.append(False)

            if 1.0 < cfg['squamous']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'fat' in cfg.keys():
        if np.random.uniform() < cfg['fat']['p']: 
            pool.append(transforms.Lambda(lambda x: fat(x, n_threads=int(np.random.uniform(0.5, 1) * cfg['fat']['n_threads']), 
                                                                    alpha=cfg['fat']['alpha'])))
            with_labels.append(False)

            if 1.0 < cfg['fat']['p']:
                necessary_ids.append(len(pool) - 1)
    
    # Geometry
    if 'rotation' in cfg.keys():
        if np.random.uniform() < cfg['rotation']['p']: 
            pool.append(transforms.Lambda(lambda x: rotate_w_labels(x, 
                cfg['rotation']['degrees'])))
            with_labels.append(True)

            if 1.0 < cfg['rotation']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'flip' in cfg.keys():
        if np.random.uniform() < cfg['flip']['p']: 
            pool.append(transforms.Lambda(lambda x: flip_w_labels(x)))
            with_labels.append(True)

            if 1.0 < cfg['flip']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'elastic' in cfg.keys():
        if np.random.uniform() < cfg['elastic']['p']:
            pool.append(transforms.Lambda(lambda x: elastic_w_labels(x, sigma=np.random.uniform(0.5, 1) * cfg['elastic']['sigma'], points=int(cfg['elastic']['points']))))
            with_labels.append(True)

            if 1.0 < cfg['elastic']['p']:
                necessary_ids.append(len(pool) - 1)
  
    # RandAug
    if 'autocontrast' in cfg.keys():
        if np.random.uniform() < cfg['autocontrast']['p']:
            pool.append(transforms.Lambda(lambda x: imo.autocontrast(x)))
            with_labels.append(False)

            if 1.0 < cfg['autocontrast']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'invert' in cfg.keys():
        if np.random.uniform() < cfg['invert']['p']:
            pool.append(transforms.Lambda(lambda x: imo.invert(x)))
            with_labels.append(False)

            if 1.0 < cfg['invert']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'equalize' in cfg.keys():
        if np.random.uniform() < cfg['equalize']['p']:
            pool.append(transforms.Lambda(lambda x: imo.equalize(x)))
            with_labels.append(False)

            if 1.0 < cfg['equalize']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'solarize' in cfg.keys():
        if np.random.uniform() < cfg['solarize']['p']:
            pool.append(transforms.Lambda(lambda x: imo.solarize(x)))
            with_labels.append(False)

            if 1.0 < cfg['solarize']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'posterize' in cfg.keys():
        if np.random.uniform() < cfg['posterize']['p']:
            pool.append(transforms.Lambda(lambda x: imo.posterize(
            x, bits=int(np.random.randint(1, 4) + 1))))
            with_labels.append(False)

            if 1.0 < cfg['posterize']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'color' in cfg.keys():
        if np.random.uniform() < cfg['color']['p']:
            pool.append(transforms.Lambda(lambda x: ime.Color(
            x).enhance((np.random.uniform()) * cfg['color']['range'] + 1)))
            with_labels.append(False)

            if 1.0 < cfg['color']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'sharpness' in cfg.keys():
        if np.random.uniform() < cfg['sharpness']['p']:
            pool.append(transforms.Lambda(lambda x: ime.Sharpness(
            x).enhance((np.random.uniform() * 2 - 1)  * cfg['sharpness']['range'] + 1)))
            with_labels.append(False)

            if 1.0 < cfg['sharpness']['p']:
                necessary_ids.append(len(pool) - 1)
    
    if 'cutout' in cfg.keys():
        if np.random.uniform() < cfg['cutout']['p']: 
            pool.append(transforms.Lambda(lambda x: cutout_w_labels(x, 
                cfg['cutout']['max_h_ratio'], cfg['cutout']['max_w_ratio'], cfg['cutout']['min_h_ratio'], cfg['cutout']['min_w_ratio'])))
            with_labels.append(True)

            if 1.0 < cfg['cutout']['p']:
                necessary_ids.append(len(pool) - 1)
    
    # Torchvision
    if 'grayscale' in cfg.keys():
        if np.random.uniform() < cfg['grayscale']['p']: 
            pool.append(transforms.RandomGrayscale(p=0.9))
            with_labels.append(False)

            if 1.0 < cfg['grayscale']['p']:
                necessary_ids.append(len(pool) - 1)

    if 'colorjitter' in cfg.keys():
        if np.random.uniform() < cfg['colorjitter']['p']: 
            pool.append(transforms.ColorJitter(brightness=np.random.uniform() * cfg['colorjitter']['brightness'], contrast=np.random.uniform() * cfg['colorjitter']['contrast'], 
                                                saturation=np.random.uniform() * cfg['colorjitter']['saturation'], hue=np.random.uniform() * cfg['colorjitter']['hue']))
            with_labels.append(False)

            if 1.0 < cfg['colorjitter']['p']:
                necessary_ids.append(len(pool) - 1)
    

    def apply_t(t, img, labels):
        if labels is not None and with_labels[t_idx]:
            if isinstance(img, im_.Image):
                img = to_tensor(img)

            img_b = torch.cat((img, labels.unsqueeze(0)), dim=0)
            img_b = t(img_b)
            img, labels = img_b[:3], img_b[3]
        else:
            img = t(img)

        if not isinstance(img, im_.Image):
            img = to_pil(img)    
        
        return img, labels


    if len(pool) > 0 and np.random.uniform() < rand_app_p:
        trans_idx = np.random.choice(len(pool), t_per_img, replace=True if len(pool) < t_per_img else False)
        for t_idx in trans_idx:
            t = pool[t_idx]
            img, labels = apply_t(t, img, labels)

        for t_idx in necessary_ids:
            t = pool[t_idx]
            img, labels = apply_t(t, img, labels)

    if 'restainer' in cfg.keys():
        img = to_tensor(img).unsqueeze(0)
        with torch.no_grad():
            L1 = torch.randint(0, cfg['restainer'].backbone.label_embeddings.num_embeddings, (img.shape[0],))
            T1 = cfg['restainer'].backbone.label_embeddings(L1)
            img = cfg['restainer'].backbone.G_S1_S2(img, T1.float(), None, None).squeeze(0).cpu()

        img = to_pil(img)

    if labels is not None:
        return img, labels
    return img


class RandomYoda(torch.nn.Module):

    def __init__(self, min_wh=25, max_wh=150, yoda_path='datasets/baby_yoda.png'):
        super().__init__()
        self.min_wh = min_wh
        self.max_wh = max_wh
        img = im_.open(yoda_path).convert('RGBA')
        self.yoda = torch.from_numpy(np.asarray(img).copy()).permute(2,0,1)


    def get_params(self,img):
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]

        h = torch.randint(self.min_wh, self.max_wh, size=(1, )).item()
        w = torch.randint(self.min_wh, self.max_wh, size=(1, )).item()

        i = torch.randint(0, img_h - h + 1, size=(1, )).item()
        j = torch.randint(0, img_w - w + 1, size=(1, )).item()

        return i,j,h,w

    def forward(self, img):
        i,j,h,w = self.get_params(img)
        rescale = transforms.Resize((h,w))
        scaled_yoda = rescale(self.yoda)

        tmp_patch = img[:3, i:i+h,j:j+w]
        alpha_mask = scaled_yoda[3] > 0.0
        tmp_patch[:3, alpha_mask] = scaled_yoda[:3, alpha_mask] / 255.0
        img[:3, i:i+h,j:j+w] = tmp_patch
        return img