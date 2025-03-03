
cd ../../
export CUDA_VISIBLE_DEVICES=0,3 
pwd
# Define variables
EXPERIMENT_TYPE="lc"
EXPERIMENT_NAME="all_data_v2"
DATASET_NAME="ztf_ff"

DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/h5file/"
 
# Run the Python script with variables
python SSL_training.py \
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
  --patience_general 10 \
  --lr_general 1e-5 \
  --num_harmonics 4 \
  --batch_size_general 256 \
  --use_sampler_general 0 \
  --num_encoders 3 \
  --embedding_size 192 \
  --embedding_size_sub 384 \
  --num_heads 4 \
  --num_epochs_general 300
