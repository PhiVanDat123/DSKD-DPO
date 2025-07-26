#!/bin/bash
set -e

# wandb login (disabled)
# export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export LM_EVAL_LOGLEVEL=DEBUG
export VLLM_LOGLEVEL=INFO

# Define variables
MODEL_NAME="/mnt/nfs-nlp/models/Llama-3.1-8B/KD_tisdpo/stdAllV1/700"
TASKS="truthfulqa_mc"
TP_SIZE=4
DTYPE="auto"
GPU_UTIL=0.9
BATCH_SIZE="auto:4"
MAX_LEN=131072

# Construct model arguments
MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_UTIL},max_model_len=${MAX_LEN}"

# Execute lm_eval
lm_eval --model vllm \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASKS}" \
        --num_fewshot=0 \
        --batch_size "${BATCH_SIZE}" \
        --output_path "output/${MODEL_NAME}/${TASKS}" \
        --log_samples \
        2>&1 | tee /tmp/lm_eval_debug.log
