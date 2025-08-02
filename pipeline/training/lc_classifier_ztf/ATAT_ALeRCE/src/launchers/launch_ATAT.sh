#!/bin/bash
cd ../../
export CUDA_VISIBLE_DEVICES=0,1 #0,1,2,3
 

for seed in {0..0}; do


expname=test_mm 
#expname=baseline_v2_correct3d_with_augs_roll_xt_normx_notmx_patience30_v3 #norom_embeddingdim
# Set experiment variables correctly
export EXPERIMENT_TYPE='LC_MD_FEAT' #_FEAT'
export EXPERIMENT_NAME=class_${expname}


export EXPERIMENT_OUTPUT_PATH="./results/BASELINE/$EXPERIMENT_TYPE/$EXPERIMENT_NAME/"
export LOG_FILENAME="$EXPERIMENT_OUTPUT_PATH/$EXPERIMENT_NAME.log"
export CONFIGS_PATH="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs"


export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/BASELINE/${EXPERIMENT_TYPE}/${expname}/

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
  ++ATATConfig.lc.use_acceleration=True\
  ++ATATConfig.lc.use_velocity=True\
  ++ATATConfig.lc.use_stats=True\
  ++ATATConfig.lc.use_metadata=False\
  ++ATATConfig.lc.use_features=False\
  ++ATATConfig.lc.sequence_norm=True\
  ++ATATConfig.lc.timefilm_gelu=True\
  ++ATATConfig.lc.timefilm_norm=True\
    ++ATATConfig.online_transforms.use_window_select=True\
    ++ATATConfig.online_transforms.use_max_window_select=True\
    ++ATATConfig.online_transforms.use_gauss_factor=False\
      ++ATATConfig.online_transforms.use_simple_time_factor=False\
      ++ATATConfig.online_transforms.use_simple_data_factor=False\
      ++ATATConfig.online_transforms.use_band_permute=False\
      ++ATATConfig.online_transforms.use_roll=False\
      ++ATATConfig.online_transforms.use_gauss_noise=False\
      ++ATATConfig.online_transforms.p_=0.5\
       # ++ATATConfig.lc.checkpoint=$CHECKPOINT\
        #++ATATConfig.tab.checkpoint=$CHECKPOINT
  #++ATATConfig.lc.checkpoint=$CHECKPOINT\
  #++ATATConfig.learning_rate=1e-3\
done
  #++ATATConfig.lc.checkpoint='/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/200/LC/just_roll_v2/'\

