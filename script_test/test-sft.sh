#!/bin/bash
set -e

#cd ..
#cd ../script
echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model.policy_name_or_path=Qwen/Qwen2.5-0.5B \
  model.reference_name_or_path=Qwen/Qwen2.5-0.5B \
  model.teacher_tokenizer_name_or_path=Qwen/Qwen2.5-0.5B \
  model.student_tokenizer_name_or_path=Qwen/Qwen2.5-0.5B \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=sft \
  eval_every=5 \
  policy_mode=student \
  datasets=tonyshelby/ultra-feedback_checking \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
  debug=false \
