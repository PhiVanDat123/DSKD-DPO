#!/bin/bash
set -e

echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"  #thay bằng path trên server của chị

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model=DSKD-DPO \
  model.policy_name_or_path=meta-llama/Llama-3.1-8B \  #thay bằng path cp llama 8B sft
  model.reference_name_or_path=Qwen/Qwen2.5-14B \      #thay bằng path cp qwen 14B sft
  model.teacher_tokenizer_name_or_path=Qwen/Qwen2.5-14B \ 
  model.student_tokenizer_name_or_path=meta-llama/Llama-3.1-8B \
  model.teacher_name_or_path=Qwen/Qwen2.5-14B \         #thay bằng path cp qwen 14B sft
  model.student_name_or_path=meta-llama/Llama-3.1-8B \  #thay bằng path cp llama 8B sft
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=tisdpo \
  policy_mode=student \
  reference_mode=teacher \
  eval_every=500 \
  datasets=pvdhihihi/7B-weight-transformed-v3-15k \
  gradient_accumulation_steps=1 batch_size=8 eval_batch_size=8 \
  trainer=FSDPTrainer sample_during_eval=false \
#  save_repo=tonyshelby/Qwen2.5_0.5B_TDPO_DSKD \
#   model.policy_name_or_path=openai-community/gpt2 \
#   model.reference_name_or_path=openai-community/gpt2 \
#   model.teacher_tokenizer_name_or_path=openai-community/gpt2 \
#   model.student_tokenizer_name_or_path=openai-community/gpt2 \
#   model.policy_block_name=Qwen2DecoderLayer \
#   model.reference_block_name=Qwen2DecoderLayer \