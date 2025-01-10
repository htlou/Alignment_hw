MODEL_DIR=$1
MODEL_NAME=$2
LORA_RANK=$3

python generate.py \
    --model_name_or_path ${MODEL_DIR} \
    --max_length 512 \
    --trust_remote_code True \
    --use_lora False \
    --lora_dim ${LORA_RANK} \
    --lora_scaling 32 \
    --lora_module_name h. \
    --lora_load_path ${MODEL_DIR} \
    --seed 42 \
    --use_cuda True \
    --output_dir_name ${MODEL_NAME}-eval