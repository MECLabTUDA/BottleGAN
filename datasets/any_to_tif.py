import tifffile as tiff
from PIL import Image
import numpy as np
import argparse
import os
import glob

def convert(files):
    for file in files:
        file_split = file.split('.')
        file_split[-1] = '.tif'
        for i in range(len(file_split)):
            e = file_split[i]
            if e == '':
                file_split[i] = '.'
        out_file = ('').join(file_split)
        out_data = np.asarray(Image.open(file))
        tiff.imwrite(out_file, out_data, tile=(512,512), photometric='rgb' if out_data.shape[-1] == 3 else 'minisblack')    
        print('Written to', out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dir")
    args = parser.parse_args()
    print(glob.glob(args.dir, recursive=True))
    files = []
    for file in glob.glob(args.dir, recursive=True):
        if file.endswith(".jpg") or file.endswith(".png"):
            files.append(file)

    convert(files)