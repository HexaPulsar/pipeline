
cd ../../
export CUDA_VISIBLE_DEVICES=0,1,2,3


for SEED in {0..0}; do

# Set experiment variables correctly
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=diciembre_8_${SEED}
export HYDRA_FULL_ERROR=1
# Run the Python script with Hydra
EXPERIMENT_OUTPUT_PATH=./results/PLUS/${EXPERIMENT_TYPE}/${EXPERIMENT_NAME}/
LOG_FILENAME=${EXPERIMENT_OUTPUT_PATH}/${EXPERIMENT_NAME}.log
CONFIGS_PATH=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
python SSL_training.py hydra.run.dir=${EXPERIMENT_OUTPUT_PATH}\
  ++ATATConfig.experiment_type=${EXPERIMENT_TYPE}\
  ++ATATConfig.experiment_name=${EXPERIMENT_NAME} \
  ++ATATConfig.log_filename=${LOG_FILENAME} \
  ++ATATConfig.save_dir_path=${EXPERIMENT_OUTPUT_PATH}\
  ++ATATConfig.datamodule.dataset.experiment_type=${EXPERIMENT_TYPE}\
  ++ATATConfig.loggers.tensorboard.save_dir=${EXPERIMENT_OUTPUT_PATH}\
  ++ATATConfig.loggers.csv.save_dir=${EXPERIMENT_OUTPUT_PATH}\
  ++ATATConfig.callbacks.model_checkpoint.dirpath=${EXPERIMENT_OUTPUT_PATH}\
  ++ATATConfig.online_transforms.use_window_select=False\
    ++ATATConfig.online_transforms.use_window_select=False\
    ++ATATConfig.online_transforms.use_max_window_select=False\
    ++ATATConfig.online_transforms.use_gauss_factor=False\
    ++ATATConfig.online_transforms.use_time_gauss_factor=False\
      ++ATATConfig.online_transforms.use_simple_time_factor=False\
      ++ATATConfig.online_transforms.use_simple_data_factor=False\
      ++ATATConfig.online_transforms.use_band_permute=False\
      ++ATATConfig.online_transforms.p_=0.5\
      ++ATATConfig.lc.dropout=0.1\
      ++ATATConfig.tab.dropout=0.1\
          ++ATATConfig.tab.embedding_size=64\
          ++ATATConfig.tab.embedding_size_sub=128\
          ++ATATConfig.tab.num_encoders=3\
          ++ATATConfig.lc.embedding_size=64\
          ++ATATConfig.lc.embedding_size_sub=128\
          ++ATATConfig.lc.num_encoders=3\
          ++ATATConfig.learning_rate=1e-3\
          ++ATATConfig.loggers.tensorboard.version=${VERSION}\
          ++ATATConfig.loggers.csv.version=${VERSION}\
          ++ATATConfig.callbacks.early_stopping.patience=5\
          ++ATATConfig.vicreg.shape_projector_1='64-128-128'\
          ++ATATConfig.vicreg.shape_projector_2='64-128-128'\
          ++ATATConfig.callbacks.early_stopping.patience=10\
          ++ATATConfig.datamodule.batch_size=256\
          ++ATATConfig.vicreg.inv_coeff=25\
          ++ATATConfig.vicreg.var_coeff=25\
          ++ATATConfig.vicreg.cov_coeff=1
          #++ATATConfig.trainer.val_check_interval=0.2
done