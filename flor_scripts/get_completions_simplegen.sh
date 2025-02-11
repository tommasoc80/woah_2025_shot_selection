#!/bin/bash

# to surpress annoyingly verbose warning
export TOKENIZERS_PARALLELISM=true

export CUDA_VISIBLE_DEVICES=0,1,2

# meta-llama/Llama-2-7b-chat-hf
# meta-llama/Llama-2-13b-chat-hf
# meta-llama/Meta-Llama-3-8B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-3.2-1B-Instruct


BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME=$(echo "$BASE_MODEL" | awk -F '/' '{print $2}')
TESTSET="raw_data"

TASK='offensiveness'

# Record the start time
start_time=$(date +%s)


python get_completions_simplegen_chat.py \
    --model_name_or_path $BASE_MODEL \
    --lora_weights "" \
    --test_data_input_path ../../data/splits/popquorn/$TASK/$TESTSET.csv \
    --n_test_samples 0 \
    --batch_size 8 \
    --load_in_8bit False \
    --log_level "debug" \
    --test_data_output_path "../../../evaluation/data/model_completions/$TASK/$MODEL_NAME/$TESTSET.csv"

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))
echo "Elapsed Time: $elapsed_time seconds"