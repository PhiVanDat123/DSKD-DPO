#!/bin/bash
set -e

#cd ..
#cd ../script
echo "Current directory: $(pwd)"

CONFIG_DIR="/home/hungpv/projects/DSKD-DPO/config"

${CONDA_PREFIX}/bin/python -u ../code/train.py \
  --config-dir $CONFIG_DIR \
  --config-name config.yaml \
  model.policy_name_or_path=Qwen/Qwen2.5-7B \
  model.reference_name_or_path=Qwen/Qwen2.5-7B \
  model.teacher_tokenizer_name_or_path=Qwen/Qwen2.5-7B \
  model.student_tokenizer_name_or_path=Qwen/Qwen2.5-7B \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=Qwen2DecoderLayer \
  loss=sft \
  eval_every=5 \
  policy_mode=teacher \
  datasets=tonyshelby/ultra-feedback_checking \
  gradient_accumulation_steps=1 batch_size=2 eval_batch_size=2 \
  trainer=FSDPTrainer sample_during_eval=false \
  debug=false \
