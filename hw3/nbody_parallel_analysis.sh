#!/bin/bash

nvcc -arch=sm_80 -ccbin=/data/align-anything/miniconda3/envs/hantao_stable/bin/gcc -o nbody_parallel_analysis nbody_parallel_analysis.cu

echo "Finished compiling nbody_parallel_analysis.cu, starting sweep..."

# Define the list of nBodies and BLOCK_SIZE values
NBODIES_LIST=(1024 2048 4096 8192 16384)
BLOCK_SIZE_LIST=(8 16 32 64 128 256)

# Specify the log file
LOG_FILE="outputs/nbody_parallel_analysis.log"

# Clear the log file if it already exists
> "$LOG_FILE"

# Print the header to the log file
echo "GPU_Name,Total_Global_Memory,Shared_Memory_Per_Block,Max_Threads_Per_Block,nBodies,BLOCK_SIZE,Average_Kernel_Execution_Time_ms,Total_Simulation_Time_ms,Interactions_Per_Second_Billion" >> "$LOG_FILE"

# Loop through each combination of nBodies and BLOCK_SIZE
for nBodies in "${NBODIES_LIST[@]}"; do
    for BLOCK_SIZE in "${BLOCK_SIZE_LIST[@]}"; do
        # Run the simulation and capture the output
        OUTPUT=$(./nbody_parallel_analysis "$nBodies" "$BLOCK_SIZE")

        # Extract GPU information
        GPU_NAME=$(echo "$OUTPUT" | grep "GPU Name" | awk -F: '{print $2}' | xargs)
        TOTAL_GLOBAL_MEM=$(echo "$OUTPUT" | grep "Total Global Memory" | awk -F: '{print $2}' | xargs)
        SHARED_MEM=$(echo "$OUTPUT" | grep "Shared Memory Per Block" | awk -F: '{print $2}' | xargs)
        MAX_THREADS=$(echo "$OUTPUT" | grep "Max Threads Per Block" | awk -F: '{print $2}' | xargs)

        # Extract performance metrics
        AVG_KERNEL_TIME=$(echo "$OUTPUT" | grep "Average Kernel Execution Time" | awk -F: '{print $2}' | xargs | sed 's/ ms//')
        TOTAL_SIM_TIME=$(echo "$OUTPUT" | grep "Total Simulation Time" | awk -F: '{print $2}' | xargs | sed 's/ ms//')
        INTERACTIONS=$(echo "$OUTPUT" | grep "Interactions Per Second" | awk -F: '{print $2}' | xargs | sed 's/ Billion//')
        INTERACTIONS_KERNEL=$(echo "$OUTPUT" | grep "Interactions Per Kernel Second" | awk -F: '{print $2}' | xargs | sed 's/ Billion//')
        
        # Append the results to the log file
        echo "$GPU_NAME,$TOTAL_GLOBAL_MEM,$SHARED_MEM,$MAX_THREADS,$nBodies,$BLOCK_SIZE,$AVG_KERNEL_TIME,$TOTAL_SIM_TIME,$INTERACTIONS" >> "$LOG_FILE"
    done
done

# Notify the user
echo "Sweep complete. Results written to $LOG_FILE"