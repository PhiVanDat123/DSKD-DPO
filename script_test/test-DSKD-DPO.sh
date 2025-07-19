#!/bin/bash
set -e

cd code
echo "Current directory: $(pwd)"

/usr/local/envs/DSKD-DPO_train/bin/python -u train.py \
  model=DSKD-DPO \
  model.policy_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.reference_name_or_path=openai-community/gpt2 \
  model.teacher_tokenizer_name_or_path=openai-community/gpt2 \
  model.student_tokenizer_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.teacher_name_or_path=openai-community/gpt2 \
  model.student_name_or_path=tonyshelby/Qwen2.5_0.5B_SFT_sample \
  model.policy_block_name=Qwen2DecoderLayer \
  model.reference_block_name=GPT2Block \
  loss=tisdpo \
  policy_mode=student \
  reference_mode=teacher \
  eval_every=5 \
  datasets=pvdhihihi/tis-dpo-5k \
  gradient_accumulation_steps=2 batch_size=4 eval_batch_size=4 \
  trainer=FSDPTrainer sample_during_eval=false \
#  save_repo=tonyshelby/Qwen2.5_0.5B_TDPO_DSKD \
#   model.policy_name_or_path=openai-community/gpt2 \
#   model.reference_name_or_path=openai-community/gpt2 \
#   model.teacher_tokenizer_name_or_path=openai-community/gpt2 \
#   model.student_tokenizer_name_or_path=openai-community/gpt2 \
#   model.policy_block_name=Qwen2DecoderLayer \
#   model.reference_block_name=Qwen2DecoderLayer \
