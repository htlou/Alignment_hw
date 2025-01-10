MODEL_DIRS=(
    "/data/align-anything/hantao/models/gpt2 gpt2-original"
    # "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-0-20250110-151349/ gpt2-full-parameter"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-1-20250110-151350/lora.pt gpt2-lora-rank-1"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-2-20250110-151349/lora.pt gpt2-lora-rank-2"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-4-20250110-151349/lora.pt gpt2-lora-rank-4"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-8-20250110-151349/lora.pt gpt2-lora-rank-8"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-16-20250110-151349/lora.pt gpt2-lora-rank-16"
    "/data/align-anything/hantao/Alignment_hw/hw1/finetune/results/outputs/gpt2-alpaca-lora-rank-32-20250110-151350/lora.pt gpt2-lora-rank-32"
)

LORA_RANK=(0 1 2 4 8 16 32)
GPU_IDS=(0 1 2 3 4 5 6 7)
for i in {0..6}; do
    model=${MODEL_DIRS[$i]}
    model_name=${model[1]}
    model_dir=${model[0]}
    echo "Generating for ${model_name}..."
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} bash generate.sh ${model_dir} ${model_name} ${LORA_RANK[$i]} &
done

wait
