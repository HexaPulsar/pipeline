
cd ../../
export CUDA_VISIBLE_DEVICES=0 #1,2 
# Define variables
for seed in {0..0}; do
  EXPERIMENT_TYPE="lc_mta"
  EXPERIMENT_NAME="lv_v1_FINE"
  DATASET_NAME="ztf_ff"

  DATA_ROOT="/home/magdalena/pipeline/pipeline/training/lc_classifier_ztf/ATAT_ALeRCE/data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12/"


  # Run the Python script with variables
  python finetune.py \
    --experiment_type_general "$EXPERIMENT_TYPE" \
    --experiment_name_general "$EXPERIMENT_NAME" \
    --name_dataset_general "$DATASET_NAME" \
    --data_root_general "$DATA_ROOT" \
    --patience_general 15 \
    --num_harmonics 4 \
    --lr_general 5e-04 \
    --use_sampler_general 1
done
