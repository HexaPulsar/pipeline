#!/bin/bash
#!/bin/bash

cd ../../
export CUDA_VISIBLE_DEVICES=0 #,1,2,3

# Set experiment variables correctly
export EXPERIMENT_TYPE='LC'
export EXPERIMENT_NAME='20260314_ema_optim_change_1491_1e3_v2'
export EXPERIMENT_OUTPUT_PATH="./results/PRETRAIN/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/magdalena/rpos/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"
LEARNING_RATE=1e-3

VERSION=0

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

python SSL_training.py hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
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
    ++ATATConfig.online_transforms.use_window_select=True\
    ++ATATConfig.online_transforms.use_max_window_select=True\
    ++ATATConfig.online_transforms.use_time_gauss_factor=True\
      ++ATATConfig.online_transforms.use_gauss_factor=True\
      ++ATATConfig.online_transforms.use_simple_time_factor=True\
      ++ATATConfig.online_transforms.use_simple_data_factor=True\
      ++ATATConfig.online_transforms.use_band_permute=True\
      ++ATATConfig.online_transforms.use_roll=False\
      ++ATATConfig.online_transforms.use_gauss_noise=False\
      ++ATATConfig.online_transforms.p_=1\
    ++ATATConfig.lc.embedding_size=64\
    ++ATATConfig.lc.embedding_size_sub=128\
    ++ATATConfig.lc.num_encoders=3\
      ++ATATConfig.lc.dropout=0.1\
          ++ATATConfig.learning_rate=$LEARNING_RATE\
          ++ATATConfig.loggers.tensorboard.version=$VERSION\
          ++ATATConfig.loggers.csv.version=$VERSION\
          ++ATATConfig.callbacks.early_stopping.patience=3\
          ++ATATConfig.vicreg.inv_coeff=1\
          ++ATATConfig.vicreg.var_coeff=49\
          ++ATATConfig.vicreg.cov_coeff=1\
          ++ATATConfig.vicreg.shape_projector_1='64-128-128'\
          ++ATATConfig.vicreg.shape_projector_2='64-128-128'
  #++ATATConfig.trainer.val_check_interval=0.2

