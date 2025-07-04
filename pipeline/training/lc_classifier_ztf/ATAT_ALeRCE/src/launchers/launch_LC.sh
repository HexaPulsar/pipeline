#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define variables
 
 
for seed in {0..0}; do

expname=baseline_${seed}_augs_elastic
#expname=multiview_v1

# Set experiment variables correctly
export EXPERIMENT_TYPE='LC'
export DIR=LINEAR
export EXPERIMENT_NAME=class_${expname}

export EXPERIMENT_OUTPUT_PATH="./results/AUGS/LC/$EXPERIMENT_NAME/"  
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifsier_ztf/ATAT_ALeRCE/src/configs"

export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/AUGS/LC/${expname}/
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
  #++ATATConfig.lc.checkpoint=$CHECKPOINT\
  #++ATATConfig.learning_rate=1e-3\
done



#la señal cos(exp(t)) convierte al tiempo en una señal acotadata pero no periodica! Evita acoplarse con el objeto periodico
