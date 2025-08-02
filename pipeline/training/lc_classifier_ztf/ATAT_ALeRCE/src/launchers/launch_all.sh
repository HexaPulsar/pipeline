#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3

for SEED in {0..0}; do

experiment_name=BASELINE
experiment_name=FINAL_v16
DIRECTORY=PRETRAIN
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml"

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
   ++ATATConfig.online_transforms.use_window_select=True\
    ++ATATConfig.online_transforms.use_max_window_select=True\
      ++ATATConfig.online_transforms.use_time_gauss_factor=True\
      ++ATATConfig.online_transforms.use_gauss_factor=True\
      ++ATATConfig.online_transforms.use_simple_time_factor=True\
      ++ATATConfig.online_transforms.use_simple_data_factor=True\
      ++ATATConfig.online_transforms.use_band_permute=True\
      ++ATATConfig.online_transforms.use_roll=True\
      ++ATATConfig.online_transforms.use_gauss_noise=False\
    ++ATATConfig.online_transforms.p_=1\
    ++ATATConfig.lc.embedding_size=64\
    ++ATATConfig.lc.embedding_size_sub=64\
    ++ATATConfig.lc.num_encoders=2\
    ++ATATConfig.lc.dropout=0.01\
    ++ATATConfig.learning_rate=1e-3 ++ATATConfig.lc.checkpoint=$CHECKPOINT
done

