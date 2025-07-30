#!/bin/bash
set -e

echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"  #thay bằng path trên server của chị

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model=DSKD-DPO \
  model.policy_name_or_path=Qwen/Qwen2.5-0.5B \  #thay bằng path cp llama 8B sft
  model.reference_name_or_path=bigscience/bloom-560m \      #thay bằng path cp qwen 14B sft
  model.teacher_tokenizer_name_or_path=bigscience/bloom-560m \ 
  model.student_tokenizer_name_or_path=Qwen/Qwen2.5-0.5B \
  model.teacher_name_or_path=bigscience/bloom-560m \         #thay bằng path cp qwen 14B sft
  model.student_name_or_path=Qwen/Qwen2.5-0.5B \  #thay bằng path cp llama 8B sft
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=BloomBlock \
  loss=tisdpo \
  policy_mode=student \
  reference_mode=teacher \
  eval_every=5 \
  datasets=pvdhihihi/14B-weight-trasformed \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
#  save_repo=tonyshelby/Qwen2.5_0.5B_TDPO_DSKD \
#   model.policy_name_or_path=openai-community/gpt2 \
#   model.reference_name_or_path=openai-community/gpt2 \
#   model.teacher_tokenizer_name_or_path=openai-community/gpt2 \
#   model.student_tokenizer_name_or_path=openai-community/gpt2 \
#   model.policy_block_name=Qwen2DecoderLayer \
#   model.reference_block_name=Qwen2DecoderLayer \