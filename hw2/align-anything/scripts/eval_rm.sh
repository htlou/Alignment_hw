export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
MODEL_NAME_OR_PATH="/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm/slice_end" # model path

TRAIN_DATASETS="" # dataset path
TRAIN_TEMPLATE="HOMEWORK" # dataset template
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="/data/align-anything/hantao/data/PKU-SafeRLHF-single-dimension" # dataset path
EVAL_TEMPLATE="HOMEWORK" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm" # output dir

# For wandb online logging
export WANDB_API_KEY="7e2dcc0c310ebcb7cdcafd5e9320d6be55cf1a33"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 1000000 \
     --epochs 3