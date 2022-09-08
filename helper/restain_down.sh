#!/bin/bash
eval "$(conda shell.bash hook)"
#export CUDA_VISIBLE_DEVICES=0
echo "GPUs in use: $CUDA_VISIBLE_DEVICES"
export PYTHONPATH="/opt/ASAP/bin"
conda activate ENVNAME

FILEIDS="14 38 32 37 35 6 4 27 33 24 27 33 24 44 31 8 44 43 33 40 41 44 100 102 55 54 101 59 60 64 65 31 99 71 11 96 36 37 38 32 35 39 38 34 35 101 54 41 41 8 42 44 32 33 72 29 42 42 31"

for fileid in $FILEIDS; do
    echo $fileid
    python run_restaining.py --org_file=../patho_data/peso/pds_${fileid}_HE.tif --org_label_file=../patho_data/peso/pds_${fileid}_HE_training_mask.tif --down_factor=4
done
conda deactivate