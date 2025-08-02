#!/bin/bash

cd ../../../
export CUDA_VISIBLE_DEVICES=2 #,1

for SEED in {0..4}; do

experiment_name=TF_GELU_NORM_EXP
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP.yaml"
 
export EXPERIMENT_OUTPUT_PATH=./results/${DIRECTORY}/LC/${EXPERIMENT_NAME}/ 
export LOG_FILENAME=$EXPERIMENT_OUTPUT_PATH/${EXPERIMENT_NAME}.log
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/${DIRECTORY}/LC/${experiment_name}/

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1 

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name $CONFIG_FILE_DIRECTORY \
  ++ATATConfig.experiment_type=${EXPERIMENT_TYPE}\
  ++ATATConfig.experiment_name=${EXPERIMENT_NAME}\
  ++ATATConfig.log_filename=$LOG_FILENAME\
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=${EXPERIMENT_TYPE}\
  ++ATATConfig.datamodule.dataset.seed=$SEED\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\ 
done

