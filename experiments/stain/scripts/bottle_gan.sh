#!/bin/bash
eval "$(conda shell.bash hook)"
echo " GPUs in use: $CUDA_VISIBLE_DEVICES"
conda activate bottlegan
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'
start=`date +%s`

FILE_NAME=${0##*/}
EXP=${FILE_NAME::-3}
RUN=${1}
OUT_FOLDER=experiments/stain/results/${EXP}/
mkdir -p ${OUT_FOLDER}

STAININGS=(datasets/staining_schemes/*_.tif)
echo -e "${RED} ${EXP} \n Stored at ${OUT_FOLDER} \n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n${NC}"



n_stainings=250
ID=${n_stainings}
FROM_FILES=(${STAININGS[*]::${n_stainings}})

# Create input strings
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FROM_FILES_PTH="[[\"../patho_data/peso/pds_39_HE_S01.tif\",\"../patho_data/peso/pds_32_HE_S01.tif\"],"
for f in ${FROM_FILES[*]}; do
    FROM_FILES_PTH="${FROM_FILES_PTH[*]}[\"${f}\"],"
done
FROM_FILES_PTH="${FROM_FILES_PTH[*]::-1}]"

TO_FILES_PTH="[[\"../patho_data/peso/pds_39_HE_S01.tif\",\"../patho_data/peso/pds_32_HE_S01.tif\"]]"

# Train BottleGAN
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

python run_restaining.py --staining_files_from=${FROM_FILES_PTH} \
                        --deep=bottle \
                        --staining_files_to=${TO_FILES_PTH} \
                        --config_embedding=configs/embeddings/bottlegan.yaml \
                        --config_predictor=configs/predictors/identity.yaml \
                        --config_task=configs/tasks/train/stain/stain_bottle_transfer.yaml \
                        --id=${ID} \
                        --seed=${RUN} \
                        --result_file=${OUT_FILE} \
                        #> /dev/null 2>&1

# Eval BottleGAN
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

OUT_FILE=${OUT_FOLDER}/${RUN}_${ID}_rgb_.json
python run_restaining.py --staining_files_from=${FROM_FILES_PTH} \
                        --deep=bottle \
                        --staining_files_to=${TO_FILES_PTH} \
                        --classifier_path=store/staining_classifier_inc_meta.pt \
                        --config_embedding=configs/embeddings/bottlegan.yaml \
                        --config_predictor=configs/predictors/identity.yaml \
                        --config_task=configs/tasks/test/stain/stain_transfer.yaml \
                        --id=${ID} \
                        --seed=${RUN} \
                        --result_file=${OUT_FILE} \
                        #> /dev/null 2>&1


conda deactivate
end=`date +%s`
runtime=$((end-start))
echo -e "${RED} ${runtime}s \n +++++++++++++++++++++++++++++ \n${NC}"