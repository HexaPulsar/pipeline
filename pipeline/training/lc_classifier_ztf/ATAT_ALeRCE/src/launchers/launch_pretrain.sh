
cd ../../
export CUDA_VISIBLE_DEVICES=0#1,2 
pwd
# Define variables
EXPERIMENT_TYPE="lc"
EXPERIMENT_NAME="v1_scaleshift"
DATASET_NAME="ztf_ff"

DATA_ROOT="/home/magdalena/pipeline/data_preprocessing/sixplusdets/final_datasets/"
 
# Run the Python script with variables
python SSL_training.py \
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
  --patience_general 10 \
  --lr_general 1e-05 \
  --num_harmonics 4 \
  --use_sampler_general 0

