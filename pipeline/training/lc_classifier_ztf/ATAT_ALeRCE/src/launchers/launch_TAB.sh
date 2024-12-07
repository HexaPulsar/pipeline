#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=1,2 

# Define variables

for seed in {0..0}; do
  EXPERIMENT_TYPE="md"
  EXPERIMENT_NAME=debugmd_${seed}
  DATASET_NAME="ztf_ff"
  DATA_ROOT="data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12"

  # Run the Python script with variables
  python training.py \
    --experiment_type_general "$EXPERIMENT_TYPE" \
    --experiment_name_general "$EXPERIMENT_NAME" \
    --name_dataset_general "$DATASET_NAME" \
    --data_root_general "$DATA_ROOT" \
    --patience_general 10 \
    --lr_general 2e-04
done 
