file_list=(
    /data/align-anything/hantao/models/Beaver-0.5B-Instruct
    # you can replace it with other models
    # or add more models below
)

# set gpu to what you want
export CUDA_VISIBLE_DEVICES="1"

INPUTFILE="./test.json"

NUM_RESPONSES=4

for model in "${file_list[@]}"; do

    model_name="${model##*/}"
    echo ${model_name}'.json'
    OUTPUT_DIR="./baseline/" # model name to change
    OUTPUT_NAME=${model_name} # model name to change

    bash generation.sh \
        --model_name_or_path ${model} \
        --output_dir ${OUTPUT_DIR} \
        --output_name ${OUTPUT_NAME} \
        --input_path ${INPUTFILE} \
        --num_responses ${NUM_RESPONSES}
done 