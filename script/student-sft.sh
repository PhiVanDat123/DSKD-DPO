#!/bin/bash
set -e

#cd ..
#cd ../script
echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model.policy_name_or_path=meta-llama/Llama-3.1-8B \
  model.reference_name_or_path=meta-llama/Llama-3.1-8B \
  model.teacher_tokenizer_name_or_path=meta-llama/Llama-3.1-8B \
  model.student_tokenizer_name_or_path=meta-llama/Llama-3.1-8B \
  model.policy_block_name=LlamaDecoderLayer \
  model.reference_block_name=LlamaDecoderLayer \
  loss=sft \
  eval_every=5 \
  policy_mode=student \
  datasets=tonyshelby/ultra-feedback_new_v1 \
  gradient_accumulation_steps=4 batch_size=2 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
  debug=false \
