import glob
from datasets.wsi_data import MultiFileMultiRegion
from single.restain import create_staining

if __name__ == '__main__':
    schemes =  ['datasets/staining_schemes/S01.tif',
                'datasets/staining_schemes/S02.tif',
                'datasets/staining_schemes/S03.tif',
                'datasets/staining_schemes/S04.tif',
                'datasets/staining_schemes/S05.tif',
                'datasets/staining_schemes/S06.tif',
                'datasets/staining_schemes/S07.tif',
                'datasets/staining_schemes/S08.tif',
                'datasets/staining_schemes/S09.tif']

    img_files = sorted(glob.glob('../patho_data/peso/*_small.tif', recursive=True))
    label_files = sorted(glob.glob('../patho_data/peso/*_labels.tif', recursive=True))
    for p in zip(img_files, label_files):
        print(p)
    dataset = MultiFileMultiRegion(img_files=img_files, label_files=label_files, h=500,w=500, max_tries=100, min_non_white=0.7)
    create_staining(schemes, n_out_schemes=484, out_schemes_folder='datasets/staining_schemes/', dataset_to_stain=dataset, verbose=0)