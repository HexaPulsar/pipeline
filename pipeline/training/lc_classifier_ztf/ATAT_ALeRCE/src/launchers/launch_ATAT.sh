#!/bin/bash

cd  ../../
export CUDA_VISIBLE_DEVICES=0,3

for seed in {0..0}; do

# Define variables
  EXPERIMENT_TYPE="lc_md"
  EXPERIMENT_NAME=CLASS_pretrain_lcmd_v1_${seed}
  DATASET_NAME="ztf_ff"
  DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/"

  python training.py \
    --experiment_type_general "$EXPERIMENT_TYPE" \
    --experiment_name_general "$EXPERIMENT_NAME" \
    --name_dataset_general "$DATASET_NAME" \
    --data_root_general "$DATA_ROOT" \
    --patience_general 15 \
    --lr_general 1e-3 \
    --batch_size_general 512 \
    --use_sampler_general 1 \
    --num_encoders 3 \
    --num_encoders_tab 3 \
    --embedding_size 128 \
    --embedding_size_sub 512 \
    --num_heads 4 \
    --num_epochs_general 35 \
    --pe_type 'tm'

done 
