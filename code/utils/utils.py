import importlib.util
import inspect
import os
import random
import socket
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Type, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

def log_rank(content, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        logging.info(content)

def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to all interfaces and use an OS provided port
        return s.getsockname()[1]  # return only the port number


def build_exp_name(
    loss_name: str,
    student_model_name: str,
    datasets: Union[str, List[str]],
    reverse_dataset: bool,
    teacher_model_name: str,
) -> str:
    """Build experiment name by combining loss name, model name, and dataset name(s)."""
    # Extract the model name without path
    student_model_short_name = student_model_name.split("/")[-1]
    teacher_model_short_name = teacher_model_name.split("/")[-1]

    dataset_part = datasets.split("/")[-1]

    # Add 'reverse' suffix if loss is dpo and reverse_dataset is True
    if loss_name == "dpo" and reverse_dataset:
        return f"{loss_name}_{student_model_short_name}_{dataset_part}_reverse"

    # Process transform info

    #method = transform.get("method", "origin")
    #params = transform.get(method, {})

    # Include key parameters in name based on transform method
    '''
    if method == "binary":
        top_percent = params.get("top_percent", 100)
        transform_str = f"{method}{top_percent}"
    elif method == "threshold":
        upper = params.get("upper_threshold", 1.0)
        lower = params.get("lower_threshold", -1.0)
        transform_str = f"{method}{upper}_{lower}"
    elif method == "threshold_and_scale":
        min_scale = params.get("min_scale", 0.7)
        max_scale = params.get("max_scale", 1.3)
        transform_str = f"{method}{min_scale}_{max_scale}"
    elif method == "random":
        min_val = params.get("min_val", 0.7)
        max_val = params.get("max_val", 1.3)
        transform_str = f"{method}{min_val}_{max_val}"
    elif method == "rank_based":
        min_scale = params.get("min_scale", 0.7)
        max_scale = params.get("max_scale", 1.3)
        transform_str = f"{method}{min_scale}_{max_scale}"
    else:
        transform_str = method
    '''
    # import ipdb; ipdb.set_trace()
    if loss_name == "tisdpo":
        #return f"{loss_name}_{student_model_short_name}_{dataset_part}_{transform_str}"
        return f"{loss_name}_{student_model_short_name}_{dataset_part}"

    if loss_name == "KD_tisdpo":
        #return f"{loss_name}_{student_model_short_name}_{teacher_model_short_name}_{dataset_part}_{transform_str}"
        return f"{loss_name}_{student_model_short_name}_{teacher_model_short_name}_{dataset_part}"

    return f"{loss_name}_{student_model_short_name}_{dataset_part}"


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if (not dist.is_initialized()) or (dist.get_rank() == 0):
        print(*args, **kwargs)


def get_local_dir(path: str) -> str:
    """Return the path to the cache directory."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_local_run_dir(exp_name: str, local_dir: str) -> str:
    """Create a local directory to store outputs for this run, and return its path."""
    now = datetime.now(timezone(timedelta(hours=8)))  # China Standard Time (UTC+8)
    timestamp = now.strftime("%m-%d_%H-%M")
    run_dir = f"{get_local_dir(local_dir)}/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

'''
def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    #chunk_size = len(list(batch.values())[0]) // world_size
    if isinstance(batch, dict):
        chunk_size = len(list(batch.values())[0]) // world_size
    elif isinstance(batch, tuple):
        chunk_size = len(batch[0]) // world_size
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    print(f"[slice_and_move_batch_for_device] rank {rank} {batch.items()}")
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device
'''
'''
def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    #print("Input batch before slicing:", batch)
    """Slice only `model_data` and `no_model_data` from batch, and move to device."""
    sliced = {}
    
    for key in ["model_data", "no_model_data"]:
        if key not in batch:
            continue

        v = batch[key]
        if isinstance(v, dict):
            # slice each tensor in the nested dict
            chunk_size = len(list(v.values())[0]) // world_size
            start = chunk_size * rank
            end = chunk_size * (rank + 1)
            sliced[key] = {
                sub_k: (sub_v[start:end].to(device) if isinstance(sub_v, torch.Tensor) else sub_v)
                for sub_k, sub_v in v.items()
            }
        elif isinstance(v, torch.Tensor):
            chunk_size = v.shape[0] // world_size
            start = chunk_size * rank
            end = chunk_size * (rank + 1)
            sliced[key] = v[start:end].to(device)
        elif isinstance(v, list):
            chunk_size = len(v) // world_size
            start = chunk_size * rank
            end = chunk_size * (rank + 1)
            sliced[key] = v[start:end]
        else:
            raise TypeError(f"Unsupported value type under key '{key}': {type(v)}")
    
    return sliced
'''

def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """
    Slice each Tensor or List in the batch by rank and move tensors to the correct device.
    """
    if not batch:
        return {}

    sliced = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            chunk_size = value.shape[0] // world_size
            start = chunk_size * rank
            end = chunk_size * (rank + 1)
            sliced[key] = value[start:end].to(device)
        elif isinstance(value, list):
            chunk_size = len(value) // world_size
            start = chunk_size * rank
            end = chunk_size * (rank + 1)
            sliced[key] = value[start:end]
        elif isinstance(value, dict):
            # Assume dicts (e.g., parent_dict) are not batched, just copy as is
            sliced[key] = value
        else:
            # For string, int, None, etc.
            sliced[key] = value

    return sliced

    
def pad_to_length(
    tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],
            dim=dim,
        )


def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)


def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) is float else v) for k, v in d.items()}


def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_gpu_memory(rank: int = None, message: str = ""):
    """Print the amount of GPU memory currently allocated for each GPU."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")
            allocated_bytes = torch.cuda.memory_allocated(device)
            if allocated_bytes == 0:
                continue
            print("*" * 40)
            print(f"[{message} rank {rank} ] GPU {i}: {allocated_bytes / 1024**2:.2f} MB")
        print("*" * 40)


def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
    """Get the class of a block from a model, using the block's class name."""
    for module in model.modules():
        if module.__class__.__name__ == block_class_name:
            return module.__class__
    raise ValueError(f"Could not find block class {block_class_name} in model {model}")


def get_block_class_from_model_class_and_block_name(
    model_class: Type, block_class_name: str
) -> Type:
    filepath = inspect.getfile(model_class)
    assert filepath.endswith(".py"), f"Expected a .py file, got {filepath}"
    assert os.path.exists(filepath), f"File {filepath} does not exist"
    assert "transformers" in filepath, f"Expected a transformers model, got {filepath}"

    module_name = filepath[filepath.find("transformers") :].replace("/", ".")[:-3]
    print(f"Searching in file {filepath}, module {module_name} for class {block_class_name}")

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class dynamically
    class_ = getattr(module, block_class_name)
    print(f"Found class {class_} in module {module_name}")
    return class_


def init_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    port: int = 12355,
    backend: str = "nccl",
):
    print(rank, "initializing distributed")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_DEBUG"] = "INFO"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def reformat_tensor(t):
    t = t.clone()  # in case you don't want to modify in-place
    mask = torch.ones_like(t)

    # Find where the first non -100 value appears
    first_valid_idx = (t != -100).nonzero(as_tuple=True)[0][0]

    # Set leading -100s to 0 except the last one
    if first_valid_idx >= 2:
        mask[: first_valid_idx - 1] = 0
    if first_valid_idx >= 1:
        mask[first_valid_idx - 1] = 1  # explicitly set, for clarity

    return mask


def drop_leading_zeros_batch(t2d):
    # Create mask where non-zero elements are marked
    mask = t2d != 0

    # Find index of first non-zero element per row
    first_nonzero_indices = mask.float().argmax(dim=1)

    # Slice each row starting from its first non-zero index
    result = [row[idx:] for row, idx in zip(t2d, first_nonzero_indices)]
    return result


def drop_zr_cols_and_padded(t3d):
    batch, _, _ = t3d.shape

    # Step 1: Compute mask of non-zero rows
    nonzero_mask = t3d.abs().sum(dim=2) != 0  # (batch, num)

    # Step 2: Find first non-zero row index for each batch
    first_nonzero_idx = nonzero_mask.float().argmax(dim=1)  # (batch,)

    # Step 3: Slice from first non-zero row onward
    trimmed = [t3d[i, first_nonzero_idx[i] :] for i in range(batch)]

    # Step 4: Pad to max valid rows across batch
    max_non_zero_rows = max(t.shape[0] for t in trimmed)

    # Step 5: Pad and stack
    padded = torch.stack(
        [F.pad(t, (0, 0, 0, max_non_zero_rows - t.shape[0])) for t in trimmed]
    )  # shape: (batch, max_valid_rows, length)
    return padded


class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(71)
        np.random.seed(71)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)
