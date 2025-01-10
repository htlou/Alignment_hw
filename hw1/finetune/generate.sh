MODEL_DIR=$1
MODEL_NAME=$2
LORA_RANK=$3

if [ $LORA_RANK -gt 0 ]; then
    USE_LORA=True
    LORA_DIR=${MODEL_DIR}
    MODEL_DIR=/data/align-anything/hantao/models/gpt2
else
    USE_LORA=False
    LORA_DIR=None
fi



python generate.py \
    --model_name_or_path ${MODEL_DIR} \
    --max_length 512 \
    --trust_remote_code True \
    --use_lora ${USE_LORA} \
    --lora_dim ${LORA_RANK} \
    --lora_scaling 32 \
    --lora_module_name h. \
    --lora_load_path ${LORA_DIR} \
    --seed 42 \
    --use_cuda True \
    --output_dir_name ${MODEL_NAME}-eval