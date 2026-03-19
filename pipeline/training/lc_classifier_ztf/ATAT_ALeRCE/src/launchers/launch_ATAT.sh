#!/bin/bash
cd ../../
export CUDA_VISIBLE_DEVICES=2,3

PATIENCE=20
for SEED in {0..1}; do


experiment_name=all_v4_band_permute_time_normalization
#experiment_name=v0000_MULTIMODAL_64128_0 ##${SEED}

EXPERIMENT_NAME=class_${experiment_name}_${SEED}
DIRECTORY=ZTF
LEARNING_RATE=1e-3
EXPERIMENT_TYPE='LC'

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml"
#export CONFIG_FILE_DIRECTORY="supervised_training.yaml"

export EXPERIMENT_OUTPUT_PATH="./results/$DIRECTORY/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/FINAL/${EXPERIMENT_TYPE}/${experiment_name}/

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

export early_stopping='loss_validation/total'
export early_stoppin_mode='min'
export checkpoint='validation/LC/f1_macro'
export checkpoint_mode='max'

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name $CONFIG_FILE_DIRECTORY\
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME\
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
  ++ATATConfig.datamodule.dataset.seed=$SEED\
  ++ATATConfig.lc.use_acceleration=True\
  ++ATATConfig.lc.use_velocity=True\
  ++ATATConfig.lc.use_stats=False\
  ++ATATConfig.lc.use_metadata=False\
  ++ATATConfig.lc.use_features=False\
  ++ATATConfig.lc.use_sequence_norm=True\
  ++ATATConfig.lc.use_timefilm_gelu=True\
  ++ATATConfig.lc.use_timefilm_norm=True\
    ++ATATConfig.online_transforms.use_window_select=True\
    ++ATATConfig.online_transforms.use_max_window_select=True\
    ++ATATConfig.online_transforms.use_gauss_factor=False\
    ++ATATConfig.online_transforms.use_time_gauss_factor=False\
      ++ATATConfig.online_transforms.use_simple_time_factor=True\
      ++ATATConfig.online_transforms.use_simple_data_factor=True\
      ++ATATConfig.online_transforms.use_band_permute=True\
      ++ATATConfig.online_transforms.use_roll=False\
      ++ATATConfig.online_transforms.use_gauss_noise=False\
      ++ATATConfig.online_transforms.p_=1\
        ++ATATConfig.learning_rate=$LEARNING_RATE\
        ++ATATConfig.lc.dropout=0.01\
        ++ATATConfig.tab.dropout=0.01\
          ++ATATConfig.tab.embedding_size=64\
          ++ATATConfig.tab.embedding_size_sub=128\
          ++ATATConfig.tab.num_encoders=3\
          ++ATATConfig.lc.embedding_size=64\
          ++ATATConfig.lc.embedding_size_sub=128\
          ++ATATConfig.lc.num_encoders=3\
          ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE\
          ++ATATConfig.callbacks.model_checkpoint.monitor='validation/LC/f1_macro'\
          ++ATATConfig.datamodule.batch_size=512\
          ++ATATConfig.loggers.tensorboard.version=$VERSION\
          ++ATATConfig.loggers.csv.version=$VERSION #\
  #++ATATConfig.lc.checkpoint=$CHECKPOINT\
  #++ATATConfig.tab.checkpoint=$CHECKPOINT
done
  #++ATATConfig.lc.checkpoint='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/200/LC/just_roll_v2/'\

