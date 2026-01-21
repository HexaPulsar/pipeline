#!/bin/bash
cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3 #0,1,2,3

PATIENCE=40

for SEED in {0..0}; do


experiment_name=BASELINE1
#experiment_name=MULTIMODAL_v7_twobranch
EXPERIMENT_NAME=class_${experiment_name}_${SEED} #_p10
DIRECTORY=ATAT
LEARNING_RATE=5e-4
EXPERIMENT_TYPE='MD_FEAT' #_FEAT'

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL_ACC_SEQNORM.yaml"

export EXPERIMENT_OUTPUT_PATH="./results/$DIRECTORY/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/${DIRECTORY}/${EXPERIMENT_TYPE}/${experiment_name}/

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name $CONFIG_FILE_DIRECTORY \
  ++ATATConfig.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.experiment_name=$EXPERIMENT_NAME \
  ++ATATConfig.log_filename=$LOG_FILENAME \
  ++ATATConfig.save_dir_path=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.experiment_type=$EXPERIMENT_TYPE\
  ++ATATConfig.loggers.tensorboard.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.loggers.csv.save_dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=$EXPERIMENT_OUTPUT_PATH\
  hydra.run.dir=$EXPERIMENT_OUTPUT_PATH\
  ++ATATConfig.datamodule.dataset.seed=$SEED\
    ++ATATConfig.online_transforms.use_window_select=False\
    ++ATATConfig.online_transforms.use_max_window_select=False\
    ++ATATConfig.online_transforms.use_gauss_factor=False\
      ++ATATConfig.online_transforms.use_simple_time_factor=False\
      ++ATATConfig.online_transforms.use_simple_data_factor=False\
      ++ATATConfig.online_transforms.use_band_permute=False\
      ++ATATConfig.online_transforms.use_roll=False\
      ++ATATConfig.online_transforms.use_gauss_noise=False\
      ++ATATConfig.online_transforms.p_=1\
        ++ATATConfig.learning_rate=$LEARNING_RATE\
        ++ATATConfig.tab.dropout=0.1\
          ++ATATConfig.tab.embedding_size=32\
          ++ATATConfig.tab.embedding_size_sub=32\
          ++ATATConfig.tab.num_encoders=3\
          ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE\
          ++ATATConfig.callbacks.model_checkpoint.monitor='validation/TAB/f1_macro'\
          ++ATATConfig.loggers.tensorboard.version=$VERSION\
          ++ATATConfig.loggers.csv.version=$VERSION #\
      #++ATATConfig.lc.checkpoint=$CHECKPOINT\
      #++ATATConfig.tab.checkpoint=$CHECKPOINT
done