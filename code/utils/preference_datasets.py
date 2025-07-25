import datasets
import torch
import warnings

# Suppress torchvision warning about image loading
warnings.filterwarnings('ignore', message='Failed to load image Python extension')

from torch.utils.data import DataLoader, Dataset
from .utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import json
import random
from datasets import load_dataset
from .distill_datasets import DistillDataset
from transformers import default_data_collator
from distiller import Distiller

'''
def binary_weight_transform(nums, top_percent=100):
    sorted_nums = sorted(nums, reverse=True)
    
    threshold_index = int(len(nums) * top_percent / 100)
    
    num_to_value = {num: 1 if i < threshold_index else 0 for i, num in enumerate(sorted_nums)}
    
    transformed_list = [num_to_value[num] for num in nums]
    
    return transformed_list

def threshold_weight_transform(nums, upper_threshold=1, lower_threshold=-1):
    result = []
    for num in nums:
        if num > upper_threshold:
            result.append(upper_threshold)
        elif num < lower_threshold:
            result.append(lower_threshold)
        else:
            result.append(num)
    return result
    
def threshold_and_scale_transform(nums, min_val=-1.5, max_val=1.5, min_scale=0.7, max_scale=1.3):
    result = []
    for num in nums:
        num = max(min_val, min(max_val, num))
        scaled = min_scale + (num - min_val) * (max_scale - min_scale) / (max_val - min_val)
        result.append(scaled)
    return result

def random_weight_transform(nums, min_val=0.7, max_val=1.3):
    return [random.uniform(min_val, max_val) for _ in nums]

def rank_based_transform(nums, min_scale=0.7, max_scale=1.3):
    sorted_indices = sorted(range(len(nums)), key=lambda k: nums[k])
    result = []
    for i, idx in enumerate(sorted_indices):
        if len(nums) == 1:
            result.append(1.0)
        else:
            scaled = min_scale + (i / (len(nums) - 1)) * (max_scale - min_scale)
            result.append(scaled)
    final_result = [result[sorted_indices.index(i)] for i in range(len(nums))]
    return final_result

weight_transform_methods = {
    'origin': lambda x: x,
    'binary': binary_weight_transform,
    'threshold': threshold_weight_transform,
    'threshold_and_scale': threshold_and_scale_transform,
    'random': random_weight_transform,
    'rank_based': rank_based_transform
}
'''

def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, reverse_dataset: bool = False):
    """Load the dataset and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
       
       Args:
           transform_config: A structured configuration for the weight transformation.

    """
    #transform_method = transform_config.get('method', 'origin')
    #transform = TransformConfig()
    #transform_config = transform.get_transform_config(transform_config)
    #transform_method = transform_config.method  

    # Get parameters for the specific method
    #if transform_method in transform_config:
    #    transform_params = transform_config.get(transform_method, {})
    '''
    def apply_weight_transform(weight_values, negate=False):
        """Helper function to apply weight transformation with the configured parameters"""
        if weight_values is None:
            return None
            
        # Apply negation if needed (for chosen weights)
        if negate:
            weight_values = [-x for x in weight_values]
            
        # Apply the transform method with parameters
        transform_func = weight_transform_methods[transform_method]
        if transform_method == 'binary' and 'top_percent' in transform_params:
            return transform_func(weight_values, top_percent=transform_params['top_percent'])
        elif transform_method == 'threshold' and ('upper_threshold' in transform_params or 'lower_threshold' in transform_params):
            return transform_func(
                weight_values, 
                upper_threshold=transform_params.get('upper_threshold', 1),
                lower_threshold=transform_params.get('lower_threshold', -1)
            )
        elif transform_method == 'threshold_and_scale' and any(param in transform_params for param in ['min_val', 'max_val', 'min_scale', 'max_scale']):
            return transform_func(
                weight_values,
                min_val=transform_params.get('min_val', -1.5),
                max_val=transform_params.get('max_val', 1.5),
                min_scale=transform_params.get('min_scale', 0.7),
                max_scale=transform_params.get('max_scale', 1.3)
            )
        elif transform_method == 'random' and ('min_val' in transform_params or 'max_val' in transform_params):
            return transform_func(
                weight_values,
                min_val=transform_params.get('min_val', 0.7),
                max_val=transform_params.get('max_val', 1.3)
            )
        elif transform_method == 'rank_based' and ('min_scale' in transform_params or 'max_scale' in transform_params):
            return transform_func(
                weight_values,
                min_scale=transform_params.get('min_scale', 0.7),
                max_scale=transform_params.get('max_scale', 1.3)
            )
        else:
            # Default case with no special parameters
            return transform_func(weight_values)
        '''
    '''
    file_path = f'{base_data_dir}/{name}/{split}.jsonl'
    print(f'Loading {name} dataset ({split} split) from {file_path}...')
    data = defaultdict(lambda: defaultdict(list))
    '''
    raw_dataset = load_dataset(name, split=split, cache_dir=cache_dir)
    data = defaultdict(lambda: defaultdict(list))
    #with open(file_path, 'r', encoding='utf-8') as f:
    for example in tqdm.tqdm(raw_dataset, desc=f"Processing {name}", disable=silent):
        #example = json.loads(line)
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        chosen_teacher_parent_dict = json.loads(example["chosen_teacher_parent_dict"])
        rejected_teacher_parent_dict = json.loads(example["rejected_teacher_parent_dict"])
        chosen_student_parent_dict = json.loads(example["chosen_student_parent_dict"])
        rejected_student_parent_dict = json.loads(example["rejected_student_parent_dict"])
            
        # Get weights
        rejected_weight_exists = 'rejected_weight' in example
        chosen_weight_exists = 'chosen_weight' in example
            
        rejected_weight = example.get('rejected_weight', None)
        chosen_weight = example.get('chosen_weight', None)
            
            # Swap chosen and rejected responses and weights if reverse_dataset is True
        if reverse_dataset:
            chosen, rejected = rejected, chosen
            if rejected_weight_exists and chosen_weight_exists:
                rejected_weight, chosen_weight = chosen_weight, rejected_weight
            
        responses = [chosen, rejected]
            
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen
        data[prompt]["chosen_teacher_parent_dict"].append(chosen_teacher_parent_dict)
        data[prompt]["rejected_teacher_parent_dict"].append(rejected_teacher_parent_dict)
        data[prompt]["chosen_student_parent_dict"].append(chosen_student_parent_dict)
        data[prompt]["rejected_student_parent_dict"].append(rejected_student_parent_dict)
            
        # Process weights
        #data[prompt]['rejected_weight'].append(apply_weight_transform(rejected_weight, negate=False))
        if rejected_weight is None:
            data[prompt]['rejected_weight'] = None
                
            #data[prompt]['chosen_weight'].append(apply_weight_transform(chosen_weight, negate=True))
        if chosen_weight is None:
            data[prompt]['chosen_weight'] = None
                
    return data

'''
def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        #print("[debug] Raw input batch:", batch)
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels') or k.endswith('_weight'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    if k.endswith('_weight'):
                        to_pad = [torch.FloatTensor(ex[k]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels') :
                    padding_value = -100
                elif k.endswith('_attention_mask') or k.endswith('_weight'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
            
        # import ipdb; ipdb.set_trace()

        return padded_batch
    return collate_fn
'''

def get_collate_fn(tokenizer: Dict) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
      ints [tokens] or strings [the original texts]) and returns a batch of examples,
      PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
                or k.endswith("_weight")
            ):
                if "prompt" in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    if k.endswith("_weight"):
                        to_pad = [torch.FloatTensor(ex[k]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    if "teacher" in k:
                        padding_value = tokenizer["teacher"].eos_token_id
                    else:
                        padding_value = tokenizer["student"].eos_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask") or k.endswith("_weight"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if "prompt" in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_parent_list"):
                padding_value = -1
                batch_size = len(batch_dict[k])
                max_outer = max(len(sample) for sample in batch_dict[k])
                max_inner = max(len(sublist) for sample in batch_dict[k] for sublist in sample)
                # Preallocate big padded tensor
                result = torch.full(
                    (batch_size, max_outer, max_inner), padding_value, dtype=torch.long
                )

                for i, sample in enumerate(batch_dict[k]):
                    for j, sublist in enumerate(sample):
                        length = len(sublist)
                        if length > 0:
                            result[i, j, :length] = torch.LongTensor(sublist)
                # new_key = k.replace('list', 'tensor')
                # padded_batch[new_key] = result
                padded_batch[k] = result
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        # import ipdb; ipdb.set_trace()

        return padded_batch

    return collate_fn


'''
def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int, rejected_weight=None, chosen_weight=None) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    # len(chosen_tokens['input_ids'])  104
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    if rejected_weight is not None:
        assert len(rejected_weight) == len(rejected_tokens['input_ids']) 
    
    if chosen_weight is not None:
        assert len(chosen_weight) == len(chosen_tokens['input_ids'])
    
    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        # print('truncate=====', len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    
    batch = {}

    if rejected_weight is not None:
        batch['rejected_weight'] =  [0] * len(prompt_tokens['input_ids']) + rejected_weight[:len(rejected_tokens['input_ids'])-1] + [0]
    else:
        batch['rejected_weight'] =  [0] * len(prompt_tokens['input_ids']) + [1]*(len(rejected_tokens['input_ids'])-1) + [0]

    if chosen_weight is not None:
        batch['chosen_weight'] =  [0] * len(prompt_tokens['input_ids']) + chosen_weight[:len(chosen_tokens['input_ids'])-1] + [0]
    else:
        batch['chosen_weight'] =  [0] * len(prompt_tokens['input_ids']) + [1]*(len(chosen_tokens['input_ids'])-1) + [0]

    assert len(batch['chosen_weight']) == len(chosen_sequence_tokens['labels'])
    assert len(batch['rejected_weight']) == len(rejected_sequence_tokens['labels'])

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    
    return batch
'''

def tokenize_batch_element(
    prompt: str,
    chosen: str,
    rejected: str,
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int,
    chosen_teacher_parent_dict: Dict,
    rejected_teacher_parent_dict: Dict,
    chosen_student_parent_dict: Dict,
    rejected_student_parent_dict: Dict,
    rejected_weight=None,
    chosen_weight=None,
) -> Dict:
    """Tokenize a single batch element.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
      in case the prompt + chosen or prompt + rejected responses is/are too long. First
      we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
      the sum of the length of the prompt and the chosen/rejected response, with -100 for the
      prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False, return_offsets_mapping=True)
    # len(chosen_tokens['input_ids'])  104
    rejected_tokens = tokenizer(rejected, add_special_tokens=False, return_offsets_mapping=True)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    if rejected_weight is not None:
        assert len(rejected_weight) == len(rejected_teacher_parent_dict)

    if chosen_weight is not None:
        assert len(chosen_weight) == len(chosen_teacher_parent_dict)

    assert tokenizer.eos_token_id not in prompt_tokens["input_ids"], (
        f"Prompt contains EOS token: {prompt}"
    )
    assert tokenizer.eos_token_id not in chosen_tokens["input_ids"], (
        f"Chosen response contains EOS token: {chosen}"
    )
    assert tokenizer.eos_token_id not in rejected_tokens["input_ids"], (
        f"Rejected response contains EOS token: {rejected}"
    )

    chosen_tokens["input_ids"].append(tokenizer.eos_token_id)
    chosen_tokens["attention_mask"].append(1)

    rejected_tokens["input_ids"].append(tokenizer.eos_token_id)
    rejected_tokens["attention_mask"].append(1)

    longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        if truncation_mode == "keep_start":
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == "keep_end":
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f"Unknown truncation mode: {truncation_mode}")

    # if that's still too long, truncate the response
    if len(prompt_tokens["input_ids"]) + longer_response_length > max_length:
        # print('truncate=====', len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))
        chosen_tokens = {k: v[: max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {
            k: v[: max_length - max_prompt_length] for k, v in rejected_tokens.items()
        }

    # Create labels
    chosen_sequence_tokens = {
        k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens if k != "offset_mapping"
    }
    rejected_sequence_tokens = {
        k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens if k != "offset_mapping"
    }
    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
    chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    chosen_sequence_tokens["offset_mapping"] = chosen_tokens["offset_mapping"]
    chosen_sequence_tokens["teacher_parent_dict"] = chosen_teacher_parent_dict
    chosen_sequence_tokens["student_parent_dict"] = chosen_student_parent_dict
    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [-100] * len(
        prompt_tokens["input_ids"]
    )
    rejected_sequence_tokens["offset_mapping"] = rejected_tokens["offset_mapping"]
    rejected_sequence_tokens["teacher_parent_dict"] = rejected_teacher_parent_dict
    rejected_sequence_tokens["student_parent_dict"] = rejected_student_parent_dict

    batch = {}

    if rejected_weight is not None:
        batch["rejected_weight"] = rejected_weight[: len(rejected_teacher_parent_dict) - 1] + [
            0
        ]  ## ? why, needs further testing!!
    else:
        batch["rejected_weight"] = [1] * (len(rejected_teacher_parent_dict))

    if chosen_weight is not None:
        batch["chosen_weight"] = chosen_weight[: len(chosen_teacher_parent_dict) - 1] + [0]
    else:
        batch["chosen_weight"] = [1] * (len(chosen_teacher_parent_dict))

    assert len(batch["chosen_weight"]) == len(chosen_sequence_tokens["teacher_parent_dict"])
    assert len(batch["rejected_weight"]) == len(rejected_sequence_tokens["teacher_parent_dict"])

    batch["prompt"] = prompt
    batch["chosen"] = prompt + " " + chosen
    batch["rejected"] = prompt + " " + rejected
    batch["chosen_response_only"] = chosen
    batch["rejected_response_only"] = rejected

    for k, toks in {
        "chosen": chosen_sequence_tokens,
        "rejected": rejected_sequence_tokens,
        "prompt": prompt_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}_{type_key}"] = tokens

    return batch

'''
def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       reverse_dataset: bool = False) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        transform_config: Configuration for weight transformation. Can be a string (method name) or 
                          a dict with a 'method' field and parameters for that method.
        base_data_dir: Base directory for the dataset.
        cache_dir: Directory to cache the datasets in.
        reverse_dataset: Whether to reverse the dataset.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, reverse_dataset=reverse_dataset).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], data['rejected_weight'], data['chosen_weight'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, sft_target, rejected_weight, chosen_weight, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target, truncation_mode, tokenizer, max_length, max_prompt_length)
                #batch_element = {k: v for k, v in batch_element.items() if 'rejected' not in k}
                # Lọc ra các key liên quan đến "chosen" hoặc "prompt", giữ lại input_ids và attention_mask (không cần labels hoặc weight nếu không dùng)
                keep_keys = ['prompt', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'chosen_weight']
                batch_element = {k: v for k, v in batch_element.items() if any(k.startswith(prefix) for prefix in keep_keys)}
                batch.append(batch_element)
                #print("[debug]", batch_element.keys())
                example_idx += 1
                if len(batch) == batch_size:
                    out_batch = collate_fn(batch)
                    if 'chosen_input_ids' not in out_batch:
                        print("[ERROR] Missing chosen_input_ids in collated batch. Keys:", out_batch.keys())
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            else:
                for index, p in enumerate(pairs):
                    if done:
                        break
                    rejected_weight_item = rejected_weight[index] if rejected_weight else None
                    chosen_weight_item = chosen_weight[index] if chosen_weight else None
                    batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length, rejected_weight_item, chosen_weight_item)
                    if 'chosen_input_ids' not in batch_element:
                        print("[ERROR] Missing chosen_input_ids in batch_element:", batch_element.keys())
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1
'''

def get_batch_iterator(
    names: List[str],
    tokenizer,
    split: str = "train",
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 0,
    silent: bool = False,
    transform_config=None,
    base_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    reverse_dataset: bool = False,
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        transform_config: Configuration for weight transformation. Can be a string (method name) or
                          a dict with a 'method' field and parameters for that method.
        base_data_dir: Base directory for the dataset.
        cache_dir: Directory to cache the datasets in.
        reverse_dataset: Whether to reverse the dataset.
    """
    assert n_epochs is not None or n_examples is not None, (
        "Must specify either n_epochs or n_examples"
    )
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = "keep_end" if name == "hh" else "keep_start"
            for prompt, data in get_dataset(
                name,
                split,
                silent=silent,
                cache_dir=cache_dir,
                transform_config=transform_config,
                base_data_dir=base_data_dir,
                reverse_dataset=reverse_dataset,
            ).items():
                flat_data.append(
                    (
                        prompt,
                        data["responses"],
                        data["pairs"],
                        data["sft_target"],
                        data["rejected_weight"],
                        data["chosen_weight"],
                        data["chosen_teacher_paren_dict"],
                        data["rejected_teacher_paren_dict"],
                        data["chosen_student_paren_dict"],
                        data["rejected_student_paren_dict"],
                        truncation_mode,
                    )
                )

        # truncation_mode = "keep_start"
        # for prompt, data in get_dataset(
        #             #name,
        #             split,
        #             silent=silent,
        #             cache_dir=cache_dir,
        #             transform_config=transform_config,
        #             base_data_dir=base_data_dir,
        #             reverse_dataset=reverse_dataset,
        #         ).items():
        #             flat_data.append(
        #                 (
        #                     prompt,
        #                     data["responses"],
        #                     data["pairs"],
        #                     data["sft_target"],
        #                     data["rejected_weight"],
        #                     data["chosen_weight"],
        #                     data["chosen_teacher_paren_dict"],
        #                     data["rejected_teacher_paren_dict"],
        #                     data["chosen_student_paren_dict"],
        #                     data["rejected_student_paren_dict"],
        #                     truncation_mode,
        #                 )
        #             )

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f"Finished generating {n_epochs} epochs on {split} split")
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for (
            prompt,
            responses,
            pairs,
            sft_target,
            rejected_weight,
            chosen_weight,
            chosen_teacher_parent_dict,
            rejected_teacher_parent_dict,
            chosen_student_parent_dict,
            rejected_student_parent_dict,
            truncation_mode,
        ) in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(
                    prompt,
                    sft_target,
                    sft_target,
                    truncation_mode,
                    tokenizer,
                    max_length,
                    max_prompt_length,
                    chosen_teacher_parent_dict,
                    rejected_teacher_parent_dict,
                    chosen_student_parent_dict,
                    rejected_student_parent_dict,
                )
                batch_element = {k: v for k, v in batch_element.items() if "rejected" not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f"Finished generating {n_examples} examples on {split} split")
                        done = True

                    batch = []
            else:
                for index, p in enumerate(pairs):
                    if done:
                        break
                    rejected_weight_item = rejected_weight[index] if rejected_weight else None
                    chosen_weight_item = chosen_weight[index] if chosen_weight else None
                    batch_element = tokenize_batch_element(
                        prompt,
                        responses[p[0]],
                        responses[p[1]],
                        truncation_mode,
                        tokenizer,
                        max_length,
                        max_prompt_length,
                        chosen_teacher_parent_dict,
                        rejected_teacher_parent_dict,
                        chosen_student_parent_dict,
                        rejected_student_parent_dict,
                        rejected_weight_item,
                        chosen_weight_item,
                    )
                    batch.append(batch_element)
                    example_idx += 1
                    if len(batch) == batch_size:
                        yield collate_fn(batch)
                        if n_examples is not None and example_idx >= n_examples:
                            if not silent:
                                print(f"FINISHED {n_examples} EXAMPLES on {split} split")
                            done = True
                        batch = []
        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True

class CustomCollate:
    def __init__(self, tokenizer: Dict):
        self.tokenizer = tokenizer
        print("[DEBUG] Tokenizer type:", type(self.tokenizer))


    def __call__(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        batch_dict = {k: [d[k] for d in batch] for k in batch[0]}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
                or k.endswith("_weight")
            ):
                if "prompt" in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    if k.endswith("_weight"):
                        to_pad = [torch.FloatTensor(ex[k]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    if "teacher" in k:
                        padding_value = self.tokenizer["teacher"].eos_token_id
                    else:
                        padding_value = self.tokenizer["student"].eos_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask") or k.endswith("_weight"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if "prompt" in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_parent_list"):
                padding_value = -1
                batch_size = len(batch_dict[k])
                max_outer = max(len(sample) for sample in batch_dict[k])
                max_inner = max(len(sublist) for sample in batch_dict[k] for sublist in sample)
                # Preallocate big padded tensor
                result = torch.full(
                    (batch_size, max_outer, max_inner), padding_value, dtype=torch.long
                )

                for i, sample in enumerate(batch_dict[k]):
                    for j, sublist in enumerate(sample):
                        length = len(sublist)
                        if length > 0:
                            result[i, j, :length] = torch.LongTensor(sublist)
                # new_key = k.replace('list', 'tensor')
                # padded_batch[new_key] = result
                padded_batch[k] = result
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        # import ipdb; ipdb.set_trace()

        return padded_batch


class PrefData(Dataset):
    def __init__(
        self,
        data_path: str,
        sft_mode: bool,
        train_test_split: str,
        reverse_dataset: bool = False,
    ):
        self.data = datasets.load_dataset(data_path, split=train_test_split)
        self.sft_mode = sft_mode
        self.reverse_dataset = reverse_dataset

    def __len__(self):
        return len(self.data)

    def safe_swap_chosen_rejected(self, d):
        temp = {}
        for k, v in d.items():
            if "teacher" in k or "student" in k:
                if "chosen" in k:
                    new_k = (
                        k.replace("chosen", "temp")
                        .replace("rejected", "chosen")
                        .replace("temp", "rejected")
                    )
                elif "rejected" in k:
                    new_k = (
                        k.replace("rejected", "temp")
                        .replace("chosen", "rejected")
                        .replace("temp", "chosen")
                    )
                else:
                    new_k = k
            else:
                new_k = k
            temp[new_k] = v
        d.clear()
        d.update(temp)

    '''
    def apply_weight_transform(
        self, weight_values, transform_method, transform_params, negate=False
    ):
        """Helper function to apply weight transformation with the configured parameters"""
        if weight_values is None:
            return None

        # Apply negation if needed (for chosen weights)
        if negate:
            weight_values = [-x for x in weight_values]

        
        # Apply the transform method with parameters
        transform_func = weight_transform_methods[transform_method]
        if transform_method == "binary" and "top_percent" in transform_params:
            return transform_func(weight_values, top_percent=transform_params["top_percent"])
        elif transform_method == "threshold" and (
            "upper_threshold" in transform_params or "lower_threshold" in transform_params
        ):
            return transform_func(
                weight_values,
                upper_threshold=transform_params.get("upper_threshold", 1),
                lower_threshold=transform_params.get("lower_threshold", -1),
            )
        elif transform_method == "threshold_and_scale" and any(
            param in transform_params for param in ["min_val", "max_val", "min_scale", "max_scale"]
        ):
            return transform_func(
                weight_values,
                min_val=transform_params.get("min_val", -1.5),
                max_val=transform_params.get("max_val", 1.5),
                min_scale=transform_params.get("min_scale", 0.7),
                max_scale=transform_params.get("max_scale", 1.3),
            )
        elif transform_method == "random" and (
            "min_val" in transform_params or "max_val" in transform_params
        ):
            return transform_func(
                weight_values,
                min_val=transform_params.get("min_val", 0.7),
                max_val=transform_params.get("max_val", 1.3),
            )
        elif transform_method == "rank_based" and (
            "min_scale" in transform_params or "max_scale" in transform_params
        ):
            return transform_func(
                weight_values,
                min_scale=transform_params.get("min_scale", 0.7),
                max_scale=transform_params.get("max_scale", 1.3),
            )
        else:
            # Default case with no special parameters
            return transform_func(weight_values)
        
    '''
    def __getitem__(self, idx):
        for k in self.data[idx].keys():
            if "parent_dict" in k:
                self.data[idx][k] = json.loads(self.data[idx][k])

        self.data[idx]["rejected_weight"] = self.data[idx].get("rejected_weight", None)
        self.data[idx]["chosen_weight"] = self.data[idx].get("chosen_weight", None)
        '''
        if self.transform_method:
            self.data[idx]["chosen_weight"] = self.apply_weight_transform(
                self.data[idx]["chosen_weight"],
                self.transform_method,
                self.transform_params,
                negate=False,
            )
            self.data[idx]["rejected_weight"] = self.apply_weight_transform(
                self.data[idx]["rejected_weight"], self.transform_method, self.transform_params
            )
        '''
        if self.reverse_dataset:
            self.safe_swap_chosen_rejected(self.data[idx])
            self.data[idx]["chosen"], self.data[idx]["rejected"] = (
                self.data[idx]["rejected"],
                self.data[idx]["chosen"],
            )
            self.data[idx]["chosen_weight"], self.data[idx]["rejected_weight"] = (
                self.data[idx]["rejected_weight"],
                self.data[idx]["chosen_weight"],
            )
        if self.sft_mode:
            batch_element = {k: v for k, v in self.data[idx].items() if "rejected" not in k}
            return batch_element
        return self.data[idx]
