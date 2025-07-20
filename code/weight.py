import argparse
import json
import multiprocessing as mp
import os
import time
import math

import torch
import tqdm
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import login
from utils.loss_utils import get_token_logps, prompt_remove
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.preference_datasets import get_collate_fn
from utils.distill_datasets import DistillDataset
from distiller import Distiller
from typing import Dict, List, Union
from utils.utils import pad_to_length

# Replace 'your_token_here' with the token you got from Hugging Face
#login(token=".....")

import logging

# Cấu hình logging
logging.basicConfig(
    filename='log.txt',              # Tên file log
    filemode='w',                    # Ghi đè mỗi lần chạy. Dùng 'a' để nối tiếp nếu cần
    level=logging.INFO,              # Mức độ log (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def fast_pad_tensor(input_tensor, max_token, max_span, pad_value=-1):
    batch_size, token_size, span_size = input_tensor.shape

    # Create the output tensor filled with pad_value
    output = input_tensor.new_full((batch_size, max_token, max_span), pad_value)

    # Copy the original values into the top-left part
    output[:, :token_size, :span_size] = input_tensor

    return output

def concatenated_inputs(batch: Dict, mode: str) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    ''''''
    max_length = max(
        batch[f"chosen_{mode}_input_ids"].shape[1], batch[f"rejected_{mode}_input_ids"].shape[1]
    )
    max_num_parents = max(
        batch[f"chosen_{mode}_parent_list"].shape[1], batch[f"rejected_{mode}_parent_list"].shape[1]
    )
    max_span = max(
        batch[f"chosen_{mode}_parent_list"].shape[2], batch[f"rejected_{mode}_parent_list"].shape[2]
    )
    concatenated_batch = {}
    #keys = [k for k in batch if mode in k]
    #keys.extend([k for k in batch if "weight" in k])
    keys = [k for k in batch if k.startswith(f"chosen_{mode}") or k.startswith(f"rejected_{mode}")]
    for k in keys:
        # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
        if k.startswith(f"chosen_{mode}"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("chosen", "concatenated")
            if "weight" in k:
                # print(k)
                # print(concatenated_key)
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_num_parents, pad_value=pad_value
                )
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = fast_pad_tensor(
                    batch[k], max_num_parents, max_span, pad_value=-1
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] = batch[k]
            else:
                # print(k)
                # print(type(batch[k]))
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
    for k in keys:
        # if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
        if k.startswith(f"rejected_{mode}"):
            pad_value = -100 if "labels" in k else 0
            concatenated_key = k.replace("rejected", "concatenated")
            if "weight" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_num_parents, pad_value=pad_value),
                    ),
                    dim=0,
                )
            elif "parent_list" in k:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        fast_pad_tensor(batch[k], max_num_parents, max_span, pad_value=-1),
                    ),
                    dim=0,
                )
            elif ("parent_dict" in k) or ("offset_mapping" in k):
                concatenated_batch[concatenated_key] += batch[k]
            else:
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                )
    return concatenated_batch

def compute_logits(
        batch, teacher_model, mode, config
    ):  
        distiller = Distiller(config)
        model = distiller.student_model
        teacher_model = teacher_model
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                batch[f"{mode}_teacher_input_ids"],
                attention_mask=batch[f"{mode}_teacher_attention_mask"],
                #position_ids=concat_input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        #concat_output_data = concatenated_inputs(output_data)
        target = batch[f"{mode}_student_labels"]
        teacher_target = batch[f"{mode}_teacher_labels"]
        
        pad_mask = target.ne(-100)
        teacher_pad_mask = teacher_target.ne(-100)

        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, batch[f"{mode}_student_input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, batch[f"{mode}_teacher_input_ids"], torch.zeros_like(teacher_target))
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float()
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        
        return t2s_logits, target

def token_weight(
    positive_model,
    negative_model,
    tokenizer,  # mistralai/Mistral-7B-v0.3
    samples,
    config,
    distiller,
    mode="chosen",  # rejected
    batch_size=8,
    mu=1.0,
    k=1.0,
    L=-0.5,
    U=1.5,
    device=None,
    process_id=None,
):  # batch_size*max_num_ptokens
    # assert
    # Get the device from the model if not provided
    if device is None:
        device = next(positive_model.parameters()).device
    # Create a descriptive prefix for the progress bar
    desc = f"GPU-{process_id}" if process_id is not None else "Processing"
    
    '''
    dataset = DistillDataset(
        config=config,
        split="train",  # hoặc "dev", "test"
        student_tokenizer=...,   # instance của tokenizer student
        teacher_tokenizers={"mistral": tokenizer}  # teacher model name map → tokenizer
    )  
    '''

    collate_fn = get_collate_fn(tokenizer)
    '''
    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    '''
    dataloader = DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,  # <- dùng hàm collate đã xử lý đầy đủ teacher/student
        num_workers=2,
        pin_memory=True,
    )

    all_weights = []

    # input_ids = [list(chain.from_iterable(d.values())) for d in sample['tea']]

    for batch in tqdm(dataloader, desc=desc, mininterval=1.0, ncols=80):
        # input_ids = samples[f"{mode}_teacher_input_ids"][i : i + batch_size]
        # input_ids_list = [torch.tensor(i) for i in input_ids]
        # input_ids_tensor = pad_sequence(
        #     input_ids_list, batch_first=True, padding_value=tokenizer.eos_token
        # )

        # attention_mask = samples[f"{mode}_teacher_attention_mask"][i : i + batch_size]
        # attention_mask_list = [torch.tensor(i) for i in attention_mask]
        # attention_mask_tensor = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        logits_1 = compute_logits(batch, positive_model, mode, config)
        logits_2 = compute_logits(batch, negative_model, mode, config)

        logits_1 = torch.log_softmax(logits_1, dim=-1).cpu()
        logits_2 = torch.log_softmax(logits_2, dim=-1).cpu()

        labels = batch[f"{mode}_teacher_labels"].cpu()

        no_prompt_logits_1, no_prompt_labels = prompt_remove(logits_1, labels)
        no_prompt_logits_2, _ = prompt_remove(logits_2, labels)

        # labels = samples[f"{mode}_teacher_labels"][i : i + batch_size]
        # labels_list = [torch.tensor(i) for i in labels]
        # labels_tensor = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        # masked = [reformat_tensor(i) for i in labels_list]
        # masked_tensor = pad_sequence(masked, batch_first=True, padding_value=0)

        # masked_labels = masked_tensor * input_ids_tensor
        # masked_non_zero = drop_leading_zeros_batch(masked_labels)
        # masked_non_zero_tensor = pad_sequence(
        #     masked_non_zero, batch_first=True, padding_value=0
        # )  # batch, max_response_length

        # masked_logits_1 = masked_tensor.unsqueeze(-1) * logits_1
        # masked_logits_2 = masked_tensor.unsqueeze(-1) * logits_2

        # padded_logits_1 = drop_zr_cols_and_padded(
        #     masked_logits_1
        # )  # batch, max_response_length, vocab_size
        # padded_logits_2 = drop_zr_cols_and_padded(masked_logits_2)

        # parent_token_dicts = samples[f"{mode}_teacher_parent_dict"][i : i + batch_size]
        # parent_token_dicts_list = [
        #     json.loads(i) for i in parent_token_dicts
        # ]  # since each dict is a string

        token_logps_1 = get_token_logps(
            no_prompt_logits_1, no_prompt_labels
        )
        token_logps_2 = get_token_logps(
            no_prompt_logits_2, no_prompt_labels
        )

        weights = torch.clamp(token_logps_1 - token_logps_2, L, U)
        weights = k * torch.exp(mu * weights)
        weights = torch.round(weights * 100) / 100
        all_weights.extend(weights.tolist())

    return all_weights


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def process_dataset_shard(
    gpu_id,
    input_file,
    positive_model,
    negative_model,
    data_shard,
    batch_size=8,
):
    # Set the GPU device - directly select device instead of using environment variable
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    #print(f"Process using device: {device}")
    logger.info(f"Process using device: {device}")

    # Load models and tokenizer for this process
    tokenizer = AutoTokenizer.from_pretrained(positive_model)
    # tokenizer.pad_token = tokenizer.eos_token

    # # convert data share from dict to Datasets
    # data_shard = Dataset.from_dict(data_shard)

    # Load models to the specific device
    model_1 = AutoModelForCausalLM.from_pretrained(positive_model).to(device)
    model_2 = AutoModelForCausalLM.from_pretrained(negative_model).to(device)

    #print(f"GPU {gpu_id}: Processing {len(data_shard)} examples")
    logger.info(f"GPU {gpu_id}: Processing {len(data_shard)} examples")

    # Pass the device and process ID to calculate_probability_differences
    rejected_weights = token_weight(
        model_1,
        model_2,
        tokenizer,
        data_shard,
        mode="rejected",
        batch_size=batch_size,
        device=device,
        process_id=gpu_id,
        mu=-1.0,
    )
    chosen_weights = token_weight(
        model_1,
        model_2,
        tokenizer,
        data_shard,
        mode="chosen",
        batch_size=batch_size,
        device=device,
        process_id=gpu_id,
        mu=1.0,
    )

    def add_weight_col(example, index):
        example["rejected_true_weight"] = rejected_weights[index]
        example["chosen_true_weight"] = chosen_weights[index]
        return example

    data_shard = data_shard.map(
        lambda example, index: add_weight_col(example, index),
        with_indices=True,
        batched=False,
        num_proc=8,
    )
    # Add weights to the data
    # for i, item in enumerate(data_shard):
    #     item["rejected_weight"] = rejected_weights[i]
    #     item["chosen_weight"] = chosen_weights[i]

    # Clean up to free GPU memory
    del model_1
    del model_2
    torch.cuda.empty_cache()

    return data_shard


def get_output_file(output_dir, file_path):
    """Get the output file path based on the input file and output directory."""
    # Extract just the filename without extension
    file_name = os.path.basename(file_path).split(".")[0]
    # Create the output file path
    output_file = os.path.join(output_dir, f"{file_name}.jsonl")
    return output_file


def parallel_process_file(file_path, args):
    #print(f"Processing file: {file_path}")
    logger.info(f"Processing file: {file_path}")
    # data = load_jsonl(file_path)

    data = load_dataset(file_path, split=args.split)

    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")

    #print(f"Using {num_gpus} GPUs for parallel processing (available: {available_gpus})")
    logger.info(f"Using {num_gpus} GPUs for parallel processing (available: {available_gpus})")

    # Split data into approximately equal shards
    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus  # Ceiling division
    for i in range(0, len(data), shard_size):
        part = data.select(range(i, min(i + shard_size, len(data))))
        shards.append(part)
        # shards.append(data[i : i + shard_size])

    shards = shards[:num_gpus]  # Make sure we don't have more shards than GPUs
    #print(f"Split data into {len(shards)} shards")
    logger.info(f"Split data into {len(shards)} shards")

    # Force sequential or handle single shard case
    if args.force_sequential or len(shards) == 1:
        # Sequential processing
        #print("Using sequential processing")
        logger.info("Using sequential processing")
        results = []
        for i in range(len(shards)):
            result = process_dataset_shard(
                i % available_gpus,
                file_path,
                args.model_name_1,
                args.model_name_2,
                shards[i],
                args.batch_size,
            )
            results.append(result)
        processed_shards = results
    else:
        # Process shards in parallel
        #print("Using parallel processing with multiprocessing Pool")
        logger.info("Using parallel processing with multiprocessing Pool")
        with mp.Pool(num_gpus) as pool:
            # Start workers with respective GPU IDs and data shards
            results = []
            for i in range(len(shards)):
                result = pool.apply_async(
                    process_dataset_shard,
                    args=(
                        i % available_gpus,
                        file_path,
                        args.model_name_1,
                        args.model_name_2,
                        shards[i],
                        args.batch_size,
                    ),
                )
                results.append(result)

            # Get results from all workers
            processed_shards = [r.get() for r in results]

    # Flatten results
    # processed_data = []
    # for result in processed_shards:
    #     processed_data.extend(result)
    processed_data = concatenate_datasets(processed_shards)

    # save to HF
    processed_data.push_to_hub("tonyshelby/ultra-feedback_weight", split=args.split)
    #print("Saved processed data to HF")
    logger.info("Saved processed data to HF")
    # Save combined results
    # output_file = get_output_file(args.output_dir, file_path)
    output_dir = os.path.join(args.output_dir, file_path, args.split)
    os.makedirs(output_dir, exist_ok=True)
    processed_data.save_to_disk(output_dir)
    # save_jsonl(processed_data, output_file)
    #print(f"Saved processed data to {output_dir}")
    logger.info(f"Saved processed data to {output_dir}")

    return output_dir


def main():
    #print("[DEBUG] main started")
    logger.info("[DEBUG] main started")
    # Try setting multiprocessing start method to spawn for better CUDA compatibility
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        #print("Multiprocessing start method already set, continuing with existing method")
        logger.info("Multiprocessing start method already set, continuing with existing method")

    parser = argparse.ArgumentParser(description="Process dataset with models in parallel.")
    parser.add_argument(
        "--positive_model_name", type=str, required=True, help="Path to the first model."
    )
    parser.add_argument(
        "--negative_model_name", type=str, required=True, help="Path to the second model."
    )
    # parser.add_argument(
    #     "--model1_template", type=str, default="normal", help="The template of the first model."
    # )
    # parser.add_argument(
    #     "--model2_template", type=str, default="normal", help="The template of the second model."
    # )
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     default="prefKD/data/ultra_feedback",
    #     required=True,
    #     help="Input directory containing JSONL files.",
    # )
    parser.add_argument(
        "--data_path",
        type=str,
        default="pvdhihihi/ultra-feedback_v6",
        required=True,
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights",
        required=True,
        help="Output directory for processed files.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing.")
    parser.add_argument(
        "--num_gpus", type=int, default=8, help="Number of GPUs to use for parallel processing."
    )
    parser.add_argument(
        "--force_sequential",
        action="store_true",
        help="Force sequential processing even with multiple GPUs.",
    )

    args = parser.parse_args()

    # Verify GPU availability
    available_gpus = torch.cuda.device_count()
    #print(f"Found {available_gpus} available GPUs")
    logger.info(f"Found {available_gpus} available GPUs")
    if available_gpus == 0:
        raise RuntimeError("No GPU devices available, but GPUs are required for this script")
    if args.num_gpus > available_gpus:
        #print(
        #    f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus} GPUs."
        #)
        logger.info(
            f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus} GPUs."
        )
        args.num_gpus = available_gpus

    # Process all files in the input directory
    start_time = time.time()
    # all_files = [
    #     os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".jsonl")
    # ]
    file_path = args.data_path.split("/")[-1]
    processed_files = []
    # for file_path in all_files:
    #print("[DEBUG] before parallel_process_file")
    logger.info("[DEBUG] before parallel_process_file")
    output_dir = parallel_process_file(file_path, args)
    #print("[DEBUG] after parallel_process_file")
    logger.info("[DEBUG] after parallel_process_file")
    processed_files.append(output_dir)

    elapsed_time = time.time() - start_time
    #print(f"Finished processing all files in {elapsed_time:.2f} seconds")
    logger.info(f"Finished processing all files in {elapsed_time:.2f} seconds")
    #print("Processed dirs:")
    logger.info("Processed dirs:")
    for file in processed_files:
        #print(f"  {file}")
        logger.info(f"  {file}")
