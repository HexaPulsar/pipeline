#!/bin/bash
#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,3 

# Define variables
 
#!/bin/bash

# Set experiment variables correctly
export EXPERIMENT_TYPE='LC_MD'
export EXPERIMENT_NAME='class_baseline'
export EXPERIMENT_OUTPUT_PATH="./results/ZTF_ff/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"

export CHECKPOINT='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/ZTF_ff/LC/ZTF_augs_V5_1e3/'
# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

python training.py \
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.train_use_sampler=1\
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  #++ATATConfig.checkpoint=$CHECKPOINT


