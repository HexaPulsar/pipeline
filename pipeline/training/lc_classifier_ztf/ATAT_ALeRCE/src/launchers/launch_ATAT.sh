#!/bin/bash

cd  ../..
export CUDA_VISIBLE_DEVICES=0,3

# Define variables

<<<<<<< Updated upstream
EXPERIMENT_TYPE="lc_md"
<<<<<<< Updated upstream
EXPERIMENT_NAME=mm_scaleshiftfine_${seed}
=======
EXPERIMENT_NAME=MM_V1_finalmnorm${seed}
=======
EXPERIMENT_TYPE="lc_md_feat"
EXPERIMENT_NAME=MM_BASELINE_md_ft${seed}
>>>>>>> Stashed changes
>>>>>>> Stashed changes
DATASET_NAME="ztf_ff"
DATA_ROOT="data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12"

# Run the Python script with variables
python training.py \
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
  --patience_general 15 \
  --num_harmonics 4 \
  --use_sampler_general 1 \
  --lr_general 1e-04