#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3

for SEED in {2..2}; do

LEARNING_RATE=1e-3
MAX_EPOCHS=500
PATIENCE=20




experiment_name=FACTOR
#experiment_name=test_2019_v7
DIRECTORY=AUGMENTATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED} # _with_random_mask_p001_d001 #_1e5


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
        ++ATATConfig.online_transforms.use_window_select=False\
        ++ATATConfig.online_transforms.use_max_window_select=False\
        ++ATATConfig.online_transforms.use_time_gauss_factor=False\
        ++ATATConfig.online_transforms.use_gauss_factor=False\
        ++ATATConfig.online_transforms.use_simple_time_factor=True\
        ++ATATConfig.online_transforms.use_simple_data_factor=True\
        ++ATATConfig.online_transforms.use_band_permute=False\
        ++ATATConfig.online_transforms.use_roll=False\
        ++ATATConfig.online_transforms.use_gauss_noise=False\
        ++ATATConfig.online_transforms.p_=1\
    ++ATATConfig.lc.embedding_size=64\
    ++ATATConfig.lc.embedding_size_sub=128\
    ++ATATConfig.lc.num_encoders=3\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE\
    ++ATATConfig.trainer.max_epochs=$MAX_EPOCHS\
    ++ATATConfig.learning_rate=$LEARNING_RATE\
    ++ATATConfig.lc.dropout=0.1\
    ++ATATConfig.datamodule.batch_size=128 # ++ATATConfig.lc.checkpoint=$CHECKPOINT

done

