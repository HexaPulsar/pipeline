#!/bin/bash

cd ../../../
export CUDA_VISIBLE_DEVICES=0,1,2,3 #,1

for SEED in {0..4}; do

experiment_name=BASELINE_TF
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}
DROPOUT=0.01
PATIENCE=20
EMBEDDING_SIZE=32
EMBEDDING_SIZE_SUB=32
DEPTH=1
LEARNING_RATE=1e-3

export early_stopping='loss_validation/total'
export early_stoppin_mode=min
export checkpoint='validation/LC/f1_macro'
export checkpoint_mode=max

export EXPERIMENT_OUTPUT_PATH=./results/${DIRECTORY}/LC/${EXPERIMENT_NAME}/
export LOG_FILENAME=$EXPERIMENT_OUTPUT_PATH/${EXPERIMENT_NAME}.log
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/${DIRECTORY}/LC/${experiment_name}/
# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name TF.yaml\
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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.dropout=$DROPOUT\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
  ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done
exit


for SEED in {0..4}; do

experiment_name=CONV_TF
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export EXPERIMENT_OUTPUT_PATH=./results/${DIRECTORY}/LC/${EXPERIMENT_NAME}/
export LOG_FILENAME=$EXPERIMENT_OUTPUT_PATH/${EXPERIMENT_NAME}.log
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/${DIRECTORY}/LC/${experiment_name}/

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name TF_CONV.yaml\
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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
    ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done



for SEED in {0..4}; do

experiment_name=TF_GELU_NORM
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export EXPERIMENT_OUTPUT_PATH=./results/${DIRECTORY}/LC/${EXPERIMENT_NAME}/
export LOG_FILENAME=$EXPERIMENT_OUTPUT_PATH/${EXPERIMENT_NAME}.log
export CHECKPOINT=/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/results/${DIRECTORY}/LC/${experiment_name}/

# Ensure directories exist
mkdir -p "$EXPERIMENT_OUTPUT_PATH"
export HYDRA_FULL_ERROR=1

python training.py \
  --config-dir /home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/src/configs/\
  --config-name TF_GELU_NORM.yaml\
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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done


for SEED in {0..4}; do

experiment_name=TF_GELU_NORM_EXP
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP.yaml"

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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done



for SEED in {0..4}; do

experiment_name=TF_GELU_NORM_EXP_VEL
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL.yaml"

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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done



for SEED in {0..4}; do

experiment_name=TF_GELU_NORM_EXP_VEL_ACC
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

export CONFIG_FILE_DIRECTORY="TF_GELU_NORM_EXP_VEL_ACC.yaml"

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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done


for SEED in {0..4}; do

experiment_name=TF_GELU_NORM_EXP_VEL_ACC_SEQNORM
DIRECTORY=ABLATION
EXPERIMENT_TYPE='LC'
EXPERIMENT_NAME=class_${experiment_name}_${SEED}

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
  ++ATATConfig.learning_rate=$LEARNING_RATE\
  ++ATATConfig.lc.embedding_size=$EMBEDDING_SIZE\
    ++ATATConfig.lc.embedding_size_sub=$EMBEDDING_SIZE_SUB\
    ++ATATConfig.lc.num_encoders=$DEPTH\
    ++ATATConfig.callbacks.early_stopping.patience=$PATIENCE
done


