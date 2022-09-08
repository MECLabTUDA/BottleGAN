import os
import openslide as os_reader
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import staintools
import argparse

from datasets.wsi_data import read_tif_region
from single.models import CombinedNet
from single.utils import bcolors
from modules.predictors import MultiArgsIdentity
import torch
import torch.nn as nn
from datasets.wsi_data import read_tif_region, read_tif_shapes
from skimage.measure import block_reduce
import time
import PIL.Image as im_
from copy import deepcopy
import json

def restain(wsi_file, stain_file, label_file=None, deep=False, crop_size=[26322, 11871], down_factor=2, verbose=0, only_print=False, only_labels=False):
    start_time = time.time()
    wsi_file_split = os.path.split(wsi_file)
    if stain_file is not None:
        stain_file_split = os.path.split(stain_file)
        out_file_stain = stain_file_split[-1].split('.')[0]
    else: 
        out_file_stain = 'small'

    out_file = '/'.join(wsi_file_split[:-1]) + '/' + \
                    wsi_file_split[-1].split('.')[0] + '_' + out_file_stain + '.' + wsi_file_split[-1].split('.')[1]
    out_labels_file = '/'.join(wsi_file_split[:-1]) + '/' + \
                    wsi_file_split[-1].split('.')[0]  + '_labels.' + wsi_file_split[-1].split('.')[1]



    dimensions = read_tif_shapes(file=wsi_file, level=0)[:2]
    out_slide = np.ones((dimensions[0] // down_factor, dimensions[1] // down_factor, 3), dtype=np.uint8)

    crop_size[0] = min(crop_size[0], dimensions[0])
    crop_size[1] = min(crop_size[1], dimensions[1])

    h_steps = dimensions[0] // crop_size[0] // down_factor
    w_steps = dimensions[1] // crop_size[1] // down_factor

    if not only_print and not only_labels:
        if not deep and stain_file is not None:
            print('Read Staining Scheme')
            st = staintools.read_image(stain_file)
            stain_norm = staintools.StainNormalizer(method='macenko')
            st = staintools.LuminosityStandardizer.standardize(st)
            stain_norm.fit(st) #If you get an AssertionError see https://github.com/Peter554/StainTools/issues/33 for possible bug fix
        elif not deep:
            print('Downsample.')
        else:
            print('Read Neural Style Transfer Network')
            model = CombinedNet(backbone=StainGAN(), predictor=MultiArgsIdentity())
            store_name = 'store/'
            store_name += 'cycle_' + stain_file.split('/')[-1].replace('.', '_') + '_' + wsi_file.split('/')[-1].replace('.', '_')
            pretrained_dict = torch.load(store_name + ".pt")
            print(bcolors.OKBLUE + "Loaded", store_name + "." + bcolors.ENDC)
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            model = model.backbone.to('cuda')

        print('Steps', h_steps, w_steps, 'Dimensions', dimensions)
        print('Restain Crops')
        for h in range(h_steps):
            print('Line', h, end='\r')
            for w in range(w_steps):
                # Image
                crop = read_tif_region(wsi_file, from_x = h * crop_size[0] * down_factor, to_x = (h+1) * crop_size[0] * down_factor,
                                                 from_y = w * crop_size[1] * down_factor, to_y = (w+1) * crop_size[1] * down_factor,
                                                 level=0)[:,:,:3]
                crop = block_reduce(crop, (down_factor, down_factor, 1), np.mean).astype(np.uint8)

                if not deep and stain_file is not None:
                    crop = staintools.LuminosityStandardizer.standardize(crop)
                    crop = stain_norm.transform(crop)
                elif not deep:
                    crop = crop
                else:
                    with torch.no_grad():
                        crop_torch = torch.from_numpy(crop).reshape(1, *crop.shape).permute(0,3,1,2).to('cuda')
                        crop_torch = crop_torch.float() / 255.
                        crop_torch = model.G_S1_S2(crop_torch)
                        crop_torch = (crop_torch.squeeze(0).permute(1,2,0) * 255).byte()
                        crop = crop_torch.cpu().numpy()

                out_slide[ h * crop_size[0] : (h + 1) * crop_size[0], w * crop_size[1] : (w + 1) * crop_size[1]] = crop
        
        if label_file is not None:
            # Labels
            #print("label file:",label_file)
            #print("out label file:",out_labels_file)
            #print("before block reduce:", np.max(read_tif_region(label_file, 0, dimensions[0], 0, dimensions[1], 5)))
            out_labels_slide = block_reduce(read_tif_region(label_file, 0, dimensions[0], 0, dimensions[1], 0), (down_factor, down_factor), np.max)
            #print(out_labels_slide)
            #print(out_labels_file)
            tiff.imwrite(out_labels_file, out_labels_slide, tile=(512,512), photometric='minisblack')

        print('Write image with size', out_slide.shape)
        tiff.imwrite(out_file, out_slide, tile=(512,512), photometric='rgb')
    
    if only_labels:
        if label_file is not None:
            # Labels
            out_labels_slide = block_reduce(read_tif_region(label_file, 0, dimensions[0], 0, dimensions[1], 0), (down_factor, down_factor), np.max)
            tiff.imwrite(out_labels_file, out_labels_slide, tile=(512,512), photometric='minisblack')
            print('Labels written to ', out_labels_file)

    if verbose > 0:
        print('Write TN')
        tn_bf = read_tif_region(wsi_file, level=0)
        #tn_bf.save('tn_bf.png', "PNG")
        plt.imshow(np.asarray(tn_bf))
        plt.savefig('tn_bf.png', dpi=600)

        out_labels_slide = read_tif_region(out_labels_file)
        tn_af = read_tif_region(out_file)

        plt.imshow(tn_af)
        #plt.imshow(out_labels_slide, alpha=0.5)
        plt.savefig('tn_af.png', dpi=600)

    print('Written to', out_file)
    print("--- %s Minutes ---" % ((time.time() - start_time) / 60))


def create_staining(schemes, n_out_schemes, out_schemes_folder, dataset_to_stain=None, verbose=0):

        if verbose > 0:
            total_schemes = len(schemes) + n_out_schemes
            n_rows = int(np.ceil(total_schemes / 5))
            view_shape = (300,300)
            view_img = np.zeros((view_shape[0] * n_rows , view_shape[1] * 5, 3), dtype=np.uint8)

        start = 26
        schemes_fitted = []
        print('Load actual schemes.')
        for scheme in schemes:
            st = staintools.read_image(scheme)
            stain_norm = staintools.StainNormalizer(method='macenko')
            st = staintools.LuminosityStandardizer.standardize(st)
            stain_norm.fit(st)

            schemes_fitted.append(stain_norm)

        print('Generate novel schemes.')
        for i in range(n_out_schemes):
            print('Stain ',i + start)
            try:
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Create target stain matrix
                target_id = np.random.choice(len(schemes_fitted), 1)
                source_id = np.random.choice(len(schemes_fitted), 1)

                target = deepcopy(schemes_fitted[int(target_id)])
                source = deepcopy(schemes_fitted[int(source_id)])

                p = np.random.uniform(1)
                target.stain_matrix_target = p * target.stain_matrix_target + (1 - p) * source.stain_matrix_target
                target.maxC_target = p * target.maxC_target + (1 - p) * source.maxC_target
                target.stain_matrix_target += np.random.normal(loc=0, scale=0.3, size=target.stain_matrix_target.shape)
                target.stain_matrix_target[target.stain_matrix_target < 0] = 0

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Read image to restain
                if dataset_to_stain is not None:
                    out_img, labels = dataset_to_stain[0]
                    tiff.imwrite(out_schemes_folder + 'S' + str(start + i) + '_labels.tif', labels.cpu().numpy(), tile=(512,512), photometric='minisblack')
                    out_img = np.asarray(out_img).copy()
                else:
                    out_img = staintools.read_image(schemes[int(source_id)])

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # restain
                out_img_c = out_img.copy()
                out_img = target.transform(out_img)

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # pixel wise restain
                augmentor = staintools.StainAugmentor(method='macenko', sigma1=0.1, sigma2=0.2)
                augmentor.fit(out_img)
                out_img_p = augmentor.pop().astype(np.uint8)

                out_path = out_schemes_folder + 'S' + str(start + i) + '_.tif'
                tiff.imwrite(out_path, out_img_p, tile=(512,512), photometric='rgb')


                out_img = deepcopy(schemes_fitted[0]).transform(out_img_c)

                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # get new stain matrix
                popped_c, mt, ms, c_ratio, r1, r2 = deepcopy(schemes_fitted[0]).transform(out_img_p.copy(), with_w=True)
                
                plt.imshow(r1)
                plt.savefig('imgs/r1.jpg')
                plt.imshow(r2)
                plt.savefig('imgs/r2.jpg')

                plt.imshow(popped_c)
                plt.savefig('imgs/out_popped_c.jpg')

                plt.imshow(out_img_p)
                plt.savefig('imgs/out_popped_jpg')

                out_path = out_schemes_folder + 'S' + str(start + i) + '.txt'
                json.dump({'mt': mt.tolist(), 'ms': ms.tolist(), 'c_ratio':c_ratio.tolist()}, open(out_path, 'w'))


                out_path = out_schemes_folder + 'S' + str(start + i) + '_01.tif'
                tiff.imwrite(out_path, out_img, tile=(512,512), photometric='rgb')

            except Exception as e:
                print(e)
                i -= 1
