#!/bin/bash
set -e

echo "Current directory: $(pwd)"


model_name_1="Qwen/Qwen2.5-14B" #thay bằng path checkpoint dpo
model_name_2="Qwen/Qwen2.5-14B" #thay bằng path checkpoint dpo đảo
#input_dir="datasets/ultra-feedback"
data_path="tonyshelby/ultra-feedback_new_v1"
output_dir="generated-data/ultra-feedback-tisdpo"
batch_size=4
num_gpus=2
force_sequential=false  # Set to true if multiprocessing causes issues
split="train"

echo "[DEBUG] script running:"
${CONDA_PREFIX}/bin/python -u ../code/weight.py \
  --positive_model_name $model_name_1 \
  --negative_model_name $model_name_2 \
  --data_path $data_path \
  --split=$split \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi)