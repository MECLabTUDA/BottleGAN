#!/bin/bash
eval "$(conda shell.bash hook)"
#export CUDA_VISIBLE_DEVICES=0
echo " GPUs in use: $CUDA_VISIBLE_DEVICES"

conda activate bg39
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'
start=`date +%s`

FILE_NAME=${0##*/}
EXP=${FILE_NAME::-3}
RUN=2
OUT_FOLDER=experiments/federated/results/${EXP}/${1}
mkdir -p ${OUT_FOLDER}
EXP_GROUPS="one"

for group in $EXP_GROUPS; do
    
    OUT_FILE=${OUT_FOLDER}/${RUN}_${group}_
    python run_federated.py --federation_type=configs/federations/combinations/base.yaml \
                            --dataplan=configs/dataset/peso/${1}${group}/dataplan_${RUN}_lac.yaml \
                            --config_task=configs/tasks/train/federated/fed_avgm.yaml  \
                            --config_test_task=configs/tasks/test/federated/base.yaml \
                            --id=${RUN} \
                            --stain_sampling=False \
                            --seed=${RUN} \
                            --result_file=${OUT_FILE} \

    rm -f store/clients/*
    rm -f store/server/*    

    
done
conda deactivate
end=`date +%s`
runtime=$((end-start))
echo -e "${RED} ${runtime}s \n +++++++++++++++++++++++++++++ \n${NC}"