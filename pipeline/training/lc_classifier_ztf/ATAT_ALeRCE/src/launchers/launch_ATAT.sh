#!/bin/bash

cd  ../..
export CUDA_VISIBLE_DEVICES=2 

# Define variables

EXPERIMENT_TYPE="lc"
EXPERIMENT_NAME=BASELINE_${seed}
DATASET_NAME="ztf_ff"
pwd
DATA_ROOT="src/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12"

# Run the Python script with variables
python training.py \ 
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
   