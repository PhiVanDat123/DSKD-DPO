#!/bin/zsh
set -e


#cd code/weight
echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

model_name_1="tonyshelby/Qwen2.5_0.5B_SFT_sample"
model_name_2="openai-community/gpt2"
# input_dir="datasets/ultra-feedback"
data_path="tonyshelby/ultra-feedback_checking"
output_dir="generated-data/ultra-feedback-tisdpo"
# model1_template="normal"
# model2_template="normal"
batch_size=4
num_gpus=2
force_sequential=false  # Set to true if multiprocessing causes issues
split="train"

# Create output directory if it doesn't exist
# mkdir -p $output_dir

# Run the parallel processing script
${CONDA_PREFIX}/bin/python -u ../code/weight.py \
  --positive_model_name $model_name_1 \
  --negative_model_name $model_name_2 \
  --spilt=$split \
  --input_dir $input_dir \
  --output_dir $output_dir \
  --batch_size $batch_size \
  --num_gpus $num_gpus \
  $(if $force_sequential; then echo "--force_sequential"; fi) 
