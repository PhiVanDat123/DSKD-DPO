#!/bin/bash
set -e

echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model=DSKD-DPO \
  model.policy_name_or_path=meta-llama/Llama-3.2-1B \
  model.reference_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.teacher_tokenizer_name_or_path=meta-llama/Llama-3.2-1B \
  model.student_tokenizer_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.teacher_name_or_path=meta-llama/Llama-3.2-1B \
  model.student_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=tisdpo \
  policy_mode=student \
  reference_mode=teacher \
  eval_every=5 \
  datasets=tonyshelby/ultra-feedback_checking \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
#  save_repo=tonyshelby/Qwen2.5_0.5B_TDPO_DSKD \
#   model.policy_name_or_path=openai-community/gpt2 \
#   model.reference_name_or_path=openai-community/gpt2 \
#   model.teacher_tokenizer_name_or_path=openai-community/gpt2 \
#   model.student_tokenizer_name_or_path=openai-community/gpt2 \
#   model.policy_block_name=Qwen2DecoderLayer \
#   model.reference_block_name=Qwen2DecoderLayer \
