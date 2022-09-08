#!/bin/bash
eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0,1,2
echo "GPUs in use: $CUDA_VISIBLE_DEVICES"
export PYTHONPATH="/opt/ASAP/bin"
conda activate learning

python datasets/wsi_data.py --dataset peso

EXP_GROUPS="one two three"
IDS="0 1 2"

for group in $EXP_GROUPS; do
    for id in $IDS; do
        python datasets/dataplan_generator.py --fold_folder=configs/dataset/peso/${group}/ \
                                                        --id=${id} \
                                                        --scenario=lac \
                                                        --n_clients=20 \
                                                        --n_labeled_windows=60 \
                                                        --n_min_windows=1 \
                                                        --labeled_share=0.5
    done
done
rm -f store/clients/*
rm -f store/server/*  
rm -f store/gans/* 
conda deactivate