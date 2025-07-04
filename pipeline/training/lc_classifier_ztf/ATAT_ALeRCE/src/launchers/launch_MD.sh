#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0 #,1,2,3

# Define variables


for seed in {0..0}; do

  expname=baseline_${seed}_softmaxtoken_gaussnoise
  # Set experiment variables correctly
  export EXPERIMENT_TYPE='MD'
  export EXPERIMENT_NAME=class_${expname} #_linear_frozen
  export EXPERIMENT_OUTPUT_PATH="./results/AUGS/MD/$EXPERIMENT_NAME/"
  export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
  export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"

  export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/MD/${expname}/
  # Ensure directories exist
  mkdir -p "$EXPERIMENT_OUTPUT_PATH"
  export HYDRA_FULL_ERROR=1
  # Run the Python script with Hydra


  export early_stopping='loss_validation/total'
  export early_stoppin_mode=min
  export checkpoint='validation/TAB/f1_macro'
  export checkpoint_mode=max
  python training.py \
    ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
    ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
    ++ATATConfig.log_filename=$LOG_FILENAME \
    ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
    ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
    ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
    ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
    ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
    ++ATATConfig.callbacks.model_checkpoint.monitor=$checkpoint\
    ++ATATConfig.callbacks.model_checkpoint.mode=$checkpoint_mode\
    ++ATATConfig.callbacks.early_stopping.mode=$early_stoppin_mode\
    ++ATATConfig.callbacks.early_stopping.monitor=$early_stopping\
    ++ATATConfig.datamodule.dataset.seed=$seed\
    ++ATATConfig.tab.length_size=6\
    hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
    ++ATATConfig.learning_rate=1e-4\

    #++ATATConfig.checkpoint=$CHECKPOINT
done

