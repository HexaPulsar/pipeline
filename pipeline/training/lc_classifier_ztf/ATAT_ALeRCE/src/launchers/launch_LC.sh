#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,1
 

for seed in {0..0}; do

expname=3enc_128
expname=all_augs_v6
# Set experiment variables correctly
export EXPERIMENT_TYPE='LC'
export DIR=LINEAR
export EXPERIMENT_NAME=class_${expname}_${seed}

export DIRECTORY=PRETRAIN
export EXPERIMENT_OUTPUT_PATH="./results/$DIRECTORY/LC/$EXPERIMENT_NAME/"  
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifsier_ztf/ATAT_ALeRCE/src/configs"

export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/$DIRECTORY/LC/${expname}/
# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1 
# Run the Python script with Hydra
export early_stopping='loss_validation/total'
export early_stoppin_mode=min
export checkpoint='validation/LC/f1_macro'
export checkpoint_mode=max

python training.py \
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.datamodule.dataset.seed=$seed\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.monitor=$checkpoint\
  ++ATATConfig.callbacks.model_checkpoint.mode=$checkpoint_mode\
  ++ATATConfig.callbacks.early_stopping.mode=$early_stoppin_mode\
  ++ATATConfig.callbacks.early_stopping.monitor=$early_stopping\
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.lc.use_acceleration=True\
  ++ATATConfig.lc.use_velocity=True\
  ++ATATConfig.lc.use_stats=True\
  ++ATATConfig.lc.use_metadata=False\
  ++ATATConfig.lc.use_features=False\
  ++ATATConfig.lc.sequence_norm=True\
  ++ATATConfig.lc.timefilm_gelu=True\
  ++ATATConfig.lc.timefilm_norm=True\
    ++ATATConfig.online_transforms.use_window_select=True\
    ++ATATConfig.online_transforms.use_max_window_select=True\
      ++ATATConfig.online_transforms.use_time_gauss_factor=True\
      ++ATATConfig.online_transforms.use_gauss_factor=True\
      ++ATATConfig.online_transforms.use_simple_time_factor=True\
      ++ATATConfig.online_transforms.use_simple_data_factor=True\
      ++ATATConfig.online_transforms.use_band_permute=True\
      ++ATATConfig.online_transforms.use_roll=True\
      ++ATATConfig.online_transforms.use_gauss_noise=True\
      ++ATATConfig.online_transforms.p_=1\
        ++ATATConfig.lc.checkpoint=$CHECKPOINT
done

