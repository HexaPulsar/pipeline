#!/bin/bash

# Navigate to ATAT_ALeRCE directory (where AR_training.py lives)
cd "$(dirname "$0")/../../"
export CUDA_VISIBLE_DEVICES=0 #,1,2,3

# Set experiment variables
export EXPERIMENT_TYPE='LC'
export EXPERIMENT_NAME='ar_pretrain_v3'
export EXPERIMENT_OUTPUT_PATH="./results/PRETRAIN/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="$(pwd)/src/configs"

LEARNING_RATE=1e-5
WARMUP_STEPS=0
ETA_MIN_FACTOR=1e-2
TOTAL_STEPS=100000
VERSION=0

mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1

python AR_training.py hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.datamodule.dataset.data_root="['/home/magdalena/Desktop/sambashare/H5_files/2019/200_2019.h5','/home/magdalena/Desktop/sambashare/H5_files/2020/200_2020.h5','/home/magdalena/Desktop/sambashare/H5_files/2021/200_2021.h5','/home/magdalena/Desktop/sambashare/H5_files/2022/200_2022.h5']"\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.lc.use_acceleration=True\
  ++ATATConfig.lc.use_velocity=True\
  ++ATATConfig.lc.use_stats=False\
  ++ATATConfig.lc.use_metadata=False\
  ++ATATConfig.lc.use_features=False\
  ++ATATConfig.lc.use_sequence_norm=True\
  ++ATATConfig.lc.use_timefilm_norm=True\
  ++ATATConfig.lc.use_exp=True\
  ++ATATConfig.lc.use_conv=True\
  ++ATATConfig.lc.use_causal=True\
  ++ATATConfig.lc.num_bands=1\
  ++ATATConfig.lc.embedding_size=128\
  ++ATATConfig.lc.embedding_size_sub=512\
  ++ATATConfig.lc.num_encoders=3\
  ++ATATConfig.lc.dropout=0.01\
  ++ATATConfig.datamodule.batch_size=512\
  ++ATATConfig.datamodule.num_workers=4\
  ++ATATConfig.datamodule.eval_probe=True\
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.pretrain.warmup_steps=$WARMUP_STEPS\
  ++ATATConfig.pretrain.total_steps=$TOTAL_STEPS\
  ++ATATConfig.pretrain.eta_min_factor=$ETA_MIN_FACTOR\
  ++ATATConfig.loggers.tensorboard.version=$VERSION\
  ++ATATConfig.loggers.csv.version=$VERSION\
  ++ATATConfig.callbacks.early_stopping.monitor=val/loss\
  ++ATATConfig.callbacks.early_stopping.patience=10\
  ++ATATConfig.trainer.check_val_every_n_epoch=10\
  ++ATATConfig.trainer.log_every_n_steps=5\
  ++ATATConfig.context_size=200 #\
  #++ATATConfig.datamodule.dataset.norm_stats_path=/path/to/norm_stats.h5
