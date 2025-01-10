LORA_RANK=(0 1 2 4 8 16 32)

GPU_IDS=(0 1 2 3 4 5 6 7)

for i in {0..6}; do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$i]} bash train.sh ${LORA_RANK[$i]} &
done

wait
