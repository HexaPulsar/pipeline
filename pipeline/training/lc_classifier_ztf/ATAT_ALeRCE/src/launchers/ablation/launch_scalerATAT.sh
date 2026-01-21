#!/bin/bash

cd ../../../
export CUDA_VISIBLE_DEVICES=0,1,2,3 #,1


LEARNING_RATE=5e-4
DROPOUT=0.1
PATIENCE=10

VERSION=0
DIRECTORY=SCALING
EXPERIMENT_TYPE='LC_MD_FEAT'

for SEED in {0..4}; do
  for DEPTH in {3..3}; do
    for i in 32 64 128; do
      experiment_name=${DEPTH}_${i}
      EXPERIMENT_NAME=class_${experiment_name}_${SEED}

      export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml"

      export EXPERIMENT_OUTPUT_PATH=./results/${DIRECTORY}/${EXPERIMENT_TYPE}/${EXPERIMENT_NAME}/
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
          ++ATATConfig.tab.embedding_size=32\
          ++ATATConfig.tab.embedding_size_sub=${i}\
          ++ATATConfig.tab.num_encoders=3\
          ++ATATConfig.lc.embedding_size=32\
          ++ATATConfig.lc.embedding_size_sub=${i}\
          ++ATATConfig.lc.num_encoders=3\
          ++ATATConfig.online_transforms.use_window_select=False\
          ++ATATConfig.online_transforms.use_max_window_select=False\
          ++ATATConfig.tab.dropout=$DROPOUT\
          ++ATATConfig.lc.dropout=$DROPOUT\
          ++ATATConfig.learning_rate=$LEARNING_RATE\
          ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE\
          ++ATATConfig.callbacks.model_checkpoint.monitor='validation/MIX/f1_macro'\
          ++ATATConfig.loggers.tensorboard.version=$VERSION\
          ++ATATConfig.loggers.csv.version=$VERSION
    done
  done
done


