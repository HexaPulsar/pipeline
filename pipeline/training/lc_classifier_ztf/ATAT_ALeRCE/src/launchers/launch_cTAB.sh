
cd ../../
export CUDA_VISIBLE_DEVICES=1,2 
pwd
# Define variables
EXPERIMENT_TYPE="md"
EXPERIMENT_NAME="pretrain_tab"
DATASET_NAME="ztf_ff"

DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/h5file/"
 
# Run the Python script with variables
python SSL_training_tab.py \
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
  --batch_size_general 128 \
  --patience_general 10 \
  --lr_general 1e-03 \
  --use_sampler_general 0 
