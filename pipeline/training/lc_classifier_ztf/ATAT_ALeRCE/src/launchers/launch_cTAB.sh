
cd ../../
export CUDA_VISIBLE_DEVICES=1,2  
<<<<<<< Updated upstream
export CUDA_VISIBLE_DEVICES=0 #1,2  
pwd
# Define variables
EXPERIMENT_TYPE="md"
EXPERIMENT_NAME="pretrain_tab"
DATASET_NAME="ztf_ff"

EXPERIMENT_OUTPUT_PATH=''

DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/h5file/"
 
=======
export CUDA_VISIBLE_DEVICES=0,3 
pwd
# Define variables
EXPERIMENT_TYPE="md_feat"
EXPERIMENT_NAME="pretrain_gauss001mask_md_ft_32"
DATASET_NAME="ztf_ff"

DATA_ROOT="/home/mdelafuente/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/h5file/"

>>>>>>> Stashed changes
# Run the Python script with variables
python SSL_training_tab.py \
  --experiment_type_general "$EXPERIMENT_TYPE" \
  --experiment_name_general "$EXPERIMENT_NAME" \
  --name_dataset_general "$DATASET_NAME" \
  --data_root_general "$DATA_ROOT" \
<<<<<<< Updated upstream
  --batch_size_general 128 \
  --patience_general 10 \
  --lr_general 1e-03 \
=======
  --batch_size_general 256 \
<<<<<<< Updated upstream
  --patience_general 100\
  --lr_general 1e-05 \
>>>>>>> Stashed changes
  --use_sampler_general 0 
=======
  --patience_general 35 \
  --lr_general 1e-04 \
  --use_sampler_general 0
>>>>>>> Stashed changes
