#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0 #,3

# Define variables

for seed in {0..0}; do
  EXPERIMENT_TYPE="md"
  EXPERIMENT_NAME=MD_TEST_02_${seed}
  DATASET_NAME="ztf_ff"
  DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/"

  # Run the Python script with variables
  python training.py \
    --experiment_type_general "$EXPERIMENT_TYPE" \
    --experiment_name_general "$EXPERIMENT_NAME" \
    --name_dataset_general "$DATASET_NAME" \
    --data_root_general "$DATA_ROOT" \
    --patience_general 15 \
    --lr_general 1e-4 \
    --batch_size_general 512 \
    --use_sampler_general 1 \
    --num_encoders 1 \
    --embedding_size 128 \
    --embedding_size_sub 512 \
    --num_heads 4 \
    --num_epochs_general 35
done


#embedding_size = 64*numbands
#embedding_size_sub = 256*band
#num_harmonics = 4

