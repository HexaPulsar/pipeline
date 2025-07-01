#!/bin/bash
#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Define variables
 
#!/bin/bash
for seed in {0..0}; do
# Set experiment variables correctly
export EXPERIMENT_TYPE='LC_MD_FEAT'
export EXPERIMENT_NAME=baseline_${seed}_dxdtdt
export EXPERIMENT_OUTPUT_PATH="./results/200/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"


# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

export MONITOR='validation/MIX/f1_macro'

python training.py \
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.monitor=$MONITOR\
  ++ATATConfig.callbacks.early_stopping.monitor=$MONITOR\
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.seed=$seed\
  #++ATATConfig.lc.checkpoint='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/200/LC/just_roll_v2/'\

done
