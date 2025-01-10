# path to your trained reward model
# for example: ~/output/rm
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
MODEL_NAME_OR_PATH="/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm/slice_end" # model path

# for example: ~/align-anything/generate_scripts/test/Qwen-0.5B-Instruct_num_4_time_20241103_133249.json
# EVAL_DATASETS="/data/align-anything/hantao/Alignment_hw/hw2/data/chosen" # dataset path
EVAL_DATASETS="/data/align-anything/hantao/Alignment_hw/hw2/align-anything/generate_scripts/baseline" # dataset path
EVAL_TEMPLATE="HOMEWORK" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm_score/baseline" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm_score \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \

# for example: ~/align-anything/generate_scripts/test/Qwen-0.5B-Instruct_num_4_time_20241103_133249.json
# EVAL_DATASETS="/data/align-anything/hantao/Alignment_hw/hw2/data/rejected" # dataset path
EVAL_DATASETS="/data/align-anything/hantao/Alignment_hw/hw2/align-anything/generate_scripts/test" # dataset path
EVAL_TEMPLATE="HOMEWORK" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="/data/align-anything/hantao/Alignment_hw/hw2/outputs/rm_score/ours" # output dir

# For wandb online logging
export WANDB_API_KEY=""

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm_score \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \