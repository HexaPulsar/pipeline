#!/bin/bash

cd ../../
<<<<<<< Updated upstream
export CUDA_VISIBLE_DEVICES=0#1,2 
=======
<<<<<<< Updated upstream
export CUDA_VISIBLE_DEVICES=0 #1,2 
=======
export CUDA_VISIBLE_DEVICES=0,3 #1,2 
>>>>>>> Stashed changes
>>>>>>> Stashed changes

# Define variables

for seed in {0..0}; do
<<<<<<< Updated upstream
  EXPERIMENT_TYPE="md"
  EXPERIMENT_NAME=baseline_1e5_${seed}
=======
  EXPERIMENT_TYPE="md_feat"
<<<<<<< Updated upstream
  EXPERIMENT_NAME=class_BASELINE_feat_md${seed}
=======
  EXPERIMENT_NAME=class_gaus001mask32${seed}
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
    --lr_general 1e-04 \
    --use_sampler_general 1
done 
