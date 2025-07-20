#!/bin/bash
set -e
set -x

echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

model_name_1="tonyshelby/Qwen2.5_0.5B_SFT_sample"
model_name_2="openai-community/gpt2"
input_dir="datasets/ultra-feedback"
data_path="tonyshelby/ultra-feedback_checking"
output_dir="generated-data/ultra-feedback-tisdpo"
batch_size=4
num_gpus=2
force_sequential=true  # Set to true if multiprocessing causes issues
split="train"

${CONDA_PREFIX}/bin/python -u ../code/weight.py \
  --positive_model_name $model_name_1 \
  --negative_model_name $model_name_2 \
  --split=$split \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $( [ "$force_sequential" = true ] && echo "--force_sequential" )

