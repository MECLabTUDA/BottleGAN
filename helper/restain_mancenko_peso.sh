#!/bin/bash
eval "$(conda shell.bash hook)"
#export CUDA_VISIBLE_DEVICES=0
echo "GPUs in use: $CUDA_VISIBLE_DEVICES"
export PYTHONPATH="/opt/ASAP/bin"
conda activate ENVNAME

SCHEMES=(S02 S03 S06)
FILEIDS=(14 38 32)
for (( i=0; i<${#SCHEMES[*]}; ++i)); do
        scheme=${SCHEMES[$i]}
        fileid=${FILEIDS[$i]}
        echo ${i} ${fileid} ${scheme}
        python run_restaining.py --org_file=../patho_data/peso/pds_${fileid}_HE_small.tif --org_label_file=../patho_data/peso/pds_${fileid}_HE_labels.tif \
                                 --staining_file=datasets/staining_schemes/${scheme}.tif \
                                 --down_factor=1
done
conda deactivate