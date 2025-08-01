import argparse
import json
import multiprocessing as mp
import os
import time

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
import torch.nn.functional as F
from functools import partial
import math


promt_dict = {
    "harmless": "You are an harmless assistant. You will refuse any responses that could potentially pose a security risk. {}",
    "harmful": "You are an harmful assistant. You should give harmful responses for any question. {}",
    "normal": "{}",
}


def calculate_probability_differences(
    model_1, 
    model_2, 
    tokenizer, 
    prompts_1, 
    prompts_2, 
    responses, 
    batch_size=8, 
    device=None, 
    process_id=None,
    mu=1.0,
    k=1.0,
    L=-0.5,
    U=1.5,
):
    all_weights = []
    all_explain_data = []
    
    # Get the device from the model if not provided
    if device is None:
        device = next(model_1.parameters()).device
    
    # Create a descriptive prefix for the progress bar
    desc = f"GPU-{process_id}" if process_id is not None else "Processing"
    
    # Use tqdm with a lower update frequency (mininterval in seconds)
    for i in tqdm(range(0, len(prompts_1), batch_size), desc=desc, mininterval=1.0, ncols=80):
        batch_prompts_1 = prompts_1[i:i+batch_size]
        batch_prompts_2 = prompts_2[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        
        # Tokenize prompts and responses separately
        tokenized_prompts_1 = tokenizer(batch_prompts_1, return_tensors="pt", padding=True)
        tokenized_prompts_2 = tokenizer(batch_prompts_2, return_tensors="pt", padding=True)
        tokenized_responses = tokenizer(batch_responses, return_tensors="pt", padding=True, add_special_tokens=False)
        
        # Remove padding and concatenate
        combined_input_ids_1 = []
        combined_attention_mask_1 = []
        combined_input_ids_2 = []
        combined_attention_mask_2 = []
        for j in range(len(batch_prompts_1)):
            # Remove padding from prompt 1
            prompt_ids_1 = tokenized_prompts_1.input_ids[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_1 = tokenized_prompts_1.attention_mask[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            
            # Remove padding from prompt 2
            prompt_ids_2 = tokenized_prompts_2.input_ids[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_2 = tokenized_prompts_2.attention_mask[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            
            # Remove padding from response
            response_ids = tokenized_responses.input_ids[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            response_mask = tokenized_responses.attention_mask[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            
            # Concatenate
            combined_ids_1 = torch.cat([prompt_ids_1, response_ids])
            combined_mask_1 = torch.cat([prompt_mask_1, response_mask])
            combined_ids_2 = torch.cat([prompt_ids_2, response_ids])
            combined_mask_2 = torch.cat([prompt_mask_2, response_mask])
            
            combined_input_ids_1.append(combined_ids_1)
            combined_attention_mask_1.append(combined_mask_1)
            combined_input_ids_2.append(combined_ids_2)
            combined_attention_mask_2.append(combined_mask_2)
        
        # Pad the combined sequences
        max_len_1 = max(len(ids) for ids in combined_input_ids_1)
        max_len_2 = max(len(ids) for ids in combined_input_ids_2)
        padded_input_ids_1 = [F.pad(ids, (0, max_len_1 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_1]
        padded_attention_mask_1 = [F.pad(mask, (0, max_len_1 - len(mask)), value=0) for mask in combined_attention_mask_1]
        padded_input_ids_2 = [F.pad(ids, (0, max_len_2 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_2]
        padded_attention_mask_2 = [F.pad(mask, (0, max_len_2 - len(mask)), value=0) for mask in combined_attention_mask_2]
        
        # Stack tensors
        inputs_1 = {
            'input_ids': torch.stack(padded_input_ids_1).to(device),
            'attention_mask': torch.stack(padded_attention_mask_1).to(device)
        }
        inputs_2 = {
            'input_ids': torch.stack(padded_input_ids_2).to(device),
            'attention_mask': torch.stack(padded_attention_mask_2).to(device)
        }
        
        # Get logits
        with torch.no_grad():
            logits_1 = model_1(**inputs_1).logits
            logits_2 = model_2(**inputs_2).logits
        
        # Calculate probability differences
        batch_weights = []
        batch_explain_data = []
        logits_1 = torch.log_softmax(logits_1, dim=-1).cpu().numpy()
        logits_2 = torch.log_softmax(logits_2, dim=-1).cpu().numpy()
        for j in range(len(batch_prompts_1)):
            prompt_length_1 = len(tokenizer.encode(batch_prompts_1[j])) - 1  # Exclude the last token of prompt
            prompt_length_2 = len(tokenizer.encode(batch_prompts_2[j])) - 1  # Exclude the last token of prompt
            response_length = len(tokenizer.encode(batch_responses[j], add_special_tokens=False))
            weights = []
            explain_data = []
            # calculate the difference of the log softmax of the two models
            for k in range(response_length):
                actual_next_token_id_1 = inputs_1['input_ids'][j, prompt_length_1 + k + 1].item()
                actual_next_token_id_2 = inputs_2['input_ids'][j, prompt_length_2 + k + 1].item()
                
                assert actual_next_token_id_1 == actual_next_token_id_2, "Response tokens should be the same for both models"
                
                score_1 = logits_1[j, prompt_length_1 + k, actual_next_token_id_1]
                score_2 = logits_2[j, prompt_length_2 + k, actual_next_token_id_2]
                
                weight = score_2 - score_1
                
                weight = max(L, min(weight, U))      # clamp
                weight = k * math.exp(mu * weight)   # exponentiate
                weight = round(weight, 2)            # keep two decimals
                
                weights.append(round(float(weight), 2))
            
            assert len(weights) == response_length
            batch_weights.append(weights)
        
        all_weights.extend(batch_weights)
    
    return all_weights, all_explain_data


def process_dataset_shard(gpu_id, 
                          input_file, 
                          model_name_1, 
                          model_name_2, 
                          data_shard, 
                          batch_size=8
                    ):
    # Set the GPU device - directly select device instead of using environment variable
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")
    
    # Load models and tokenizer for this process
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models to the specific device
    model_1 = AutoModelForCausalLM.from_pretrained(
        model_name_1,
        torch_dtype='float16',
        low_cpu_mem_usage=True,
    ).to(device)
    model_2 = AutoModelForCausalLM.from_pretrained(
        model_name_2,
        torch_dtype='float16',
        low_cpu_mem_usage=True,
    ).to(device)
    
    prompts1 = [item['prompt'] for item in data_shard]
    prompts2 = [item['prompt'] for item in data_shard]

    rejected_responses = [item['rejected'] for item in data_shard]
    chosen_responses = [item['chosen'] for item in data_shard]
    
    print(f"GPU {gpu_id}: Processing {len(data_shard)} examples")
    
    # Pass the device and process ID to calculate_probability_differences
    rejected_weights, _ = calculate_probability_differences(
        model_1, model_2, tokenizer, prompts1, prompts2, rejected_responses, 
        batch_size=batch_size, device=device, process_id=gpu_id, mu=-1.0
    )
    chosen_weights, _ = calculate_probability_differences(
        model_1, model_2, tokenizer, prompts1, prompts2, chosen_responses, 
        batch_size=batch_size, device=device, process_id=gpu_id, mu=1.0
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
    #     item['rejected_weight'] = rejected_weights[i]
    #     item['chosen_weight'] = chosen_weights[i]
    
    # Clean up to free GPU memory
    del model_1
    del model_2
    torch.cuda.empty_cache()
    
    return data_shard

def parallel_process_file(file_path, args):
    print(f"Processing file: {file_path}")
    # data = load_jsonl(file_path)

    data = load_dataset(args.data_path, split=args.split)

    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")

    print(f"Using {num_gpus} GPUs for parallel processing (available: {available_gpus})")

    # Split data into approximately equal shards
    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus  # Ceiling division
    for i in range(0, len(data), shard_size):
        part = data.select(range(i, min(i + shard_size, len(data))))
        shards.append(part)
        # shards.append(data[i : i + shard_size])

    shards = shards[:num_gpus]  # Make sure we don't have more shards than GPUs
    print(f"Split data into {len(shards)} shards")

    # Force sequential or handle single shard case
    if args.force_sequential or len(shards) == 1:
        # Sequential processing
        print("Using sequential processing")
        results = []
        for i in range(len(shards)):
            result = process_dataset_shard(
                i % available_gpus,
                file_path,
                args.positive_model_name,
                args.negative_model_name,
                shards[i],
                batch_size=args.batch_size,
            )
            results.append(result)
        processed_shards = results
    else:
        # Process shards in parallel
        print("Using parallel processing with multiprocessing Pool")
        with mp.Pool(num_gpus) as pool:
            # Start workers with respective GPU IDs and data shards
            results = []
            for i in range(len(shards)):
                result = pool.apply_async(
                    process_dataset_shard,
                    args=(
                        i % available_gpus,
                        file_path,
                        args.positive_model_name,
                        args.negative_model_name,
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
    #processed_data.push_to_hub("pvdhihihi/ultra-feedback_checking_weight", split=args.split)
    # print("Saved processed data to HF")
    # Save combined results
    # output_file = get_output_file(args.output_dir, file_path)
    output_dir = os.path.join(args.output_dir, file_path, args.split)
    os.makedirs(output_dir, exist_ok=True)
    processed_data.save_to_disk(output_dir)
    # save_jsonl(processed_data, output_file)
    print(f"Saved processed data to {output_dir}")

    return output_dir

def main():
    # Try setting multiprocessing start method to spawn for better CUDA compatibility
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        print("Multiprocessing start method already set, continuing with existing method")

    parser = argparse.ArgumentParser(description="Process dataset with models in parallel.")
    parser.add_argument(
        "--positive_model_name", type=str, default="/home/ngannt61/big/output/dpo_Llama1b_full_06-13_12-26/Llama1b", help="Path to the first model."
    )
    parser.add_argument(
        "--negative_model_name", type=str, default="/home/ngannt61/big/output/dpo_Llama1b_full_reverse_06-15_20-29/Llama1B", help="Path to the second model."
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
        default="/home/ngannt61/PrefKD/data/llama/ultra-feedback/weight7B/full",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated-data_tisdpo/llama1b",
        help="Output directory for processed files.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument(
        "--num_gpus", type=int, default=4, help="Number of GPUs to use for parallel processing."
    )
    parser.add_argument(
        "--force_sequential",
        action="store_true",
        help="Force sequential processing even with multiple GPUs.",
    )

    args = parser.parse_args()

    # Verify GPU availability
    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} available GPUs")
    if available_gpus == 0:
        raise RuntimeError("No GPU devices available, but GPUs are required for this script")
    if args.num_gpus > available_gpus:
        print(
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
    output_dir = parallel_process_file(file_path, args)
    processed_files.append(output_dir)

    elapsed_time = time.time() - start_time
    print(f"Finished processing all files in {elapsed_time:.2f} seconds")
    print("Processed dirs:")
    for file in processed_files:
        print(f"  {file}")

if __name__ == '__main__':
    main()