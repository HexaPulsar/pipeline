#!/bin/bash
#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=1 #,2,3

# Define variables
 
#!/bin/bash
for seed in {1..4}; do
# Set experiment variables correctly
export EXPERIMENT_TYPE='LC_MD_FEAT'
export EXPERIMENT_NAME=baseline_${seed}_
export EXPERIMENT_OUTPUT_PATH="./results/AUGS/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"


# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

export early_stopping='loss_validation/total'
export early_stoppin_mode=min
export checkpoint='validation/MIX/f1_macro'
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
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.seed=$seed\
  #++ATATConfig.lc.checkpoint='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/200/LC/just_roll_v2/'\

done
