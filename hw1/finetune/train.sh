MODEL_NAME_OR_PATH=/data/align-anything/hantao/models/gpt2

# get LoRA rank from input
LORA_RANK=$1

OUTPUT_DIR_NAME=outputs/gpt2-alpaca-lora-rank-$LORA_RANK
DATA_PATH=./data/alpaca_data.json

# if LORA_RANK > 0, use LoRA
USE_LORA=False
if [ $LORA_RANK -gt 0 ]; then
    USE_LORA=True
fi

python train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --max_length 512 \
    --trust_remote_code True \
    --use_lora $USE_LORA \
    --lora_dim $LORA_RANK \
    --lora_scaling 32 \
    --lora_module_name h. \
    --data_path $DATA_PATH \
    --epochs 4 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4 \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --seed 42 \
    --eval_batch_size 16 \
    --eval_ratio 0.01 \
    --eval_interval 100 \
    --output_dir_name $OUTPUT_DIR_NAME