import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils.utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, build_exp_name
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set, List, Union
import resource
import sys
import copy
import torch.distributed as dist
from huggingface_hub import login

mp.set_start_method("fork", force=True)

os.environ["WANDB_SILENT"] = "true"

wandb.login(key="c029bf5e12185949e8e5745af223f868adf63ee4")

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver(
    "build_exp_name", 
    lambda loss_name, model_name, datasets, reverse_dataset, reference_model_name: 
        build_exp_name(loss_name, model_name, datasets, reverse_dataset, reference_model_name)
)


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    print(f"[worker_main] config type: {type(config)}")
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    
    # === 🔧 Init torch.distributed for FSDP ===
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl",  # Hoặc "gloo" nếu không có GPU
        rank=rank,
        world_size=world_size
    )

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    # Convert transform configuration to a proper object if needed
    # if 'transform' in config and isinstance(config.transform, (dict, str)):
    #transform_config = get_transform_config(config.transform_config)
    
    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config.seed, config.local_run_dir, config, reference_model=reference_model, 
                         rank=rank, world_size=world_size)

    trainer.train(config)
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    '''
    # Load transform configuration before resolving experiment name
    if isinstance(config.transform_config, str):
        # Check if it's a path to a configuration file
        if os.path.exists(config.transform_config) and config.transform_config.endswith('.yaml'):
            transform_config = TransformConfig.from_file(config.transform_config)
            print(f"Loaded transform configuration from {config.transform_config}")
        # Check if it's a preset name
        elif os.path.exists(f"config/transform/{config.transform_config}.yaml"):
            transform_config = TransformConfig.from_preset(config.transform_config)
            print(f"Loaded transform configuration preset: {config.transform_config}")
        # Otherwise it's just a method name
        else:
            transform_config = TransformConfig(method=config.transform_config)
            print(f"Using transform method: {config.transform_config}")
    else:
        # Using the default configuration from OmegaConf
        transform_config = config.transform_config
        print("Using transform configuration from config file")
    '''
    # Update config.transform with the full config object for experiment naming
    #config.transform_config = transform_config.to_dict() if hasattr(transform_config, 'to_dict') else transform_config

    # Now resolve hydra references with the updated transform config
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    # Print transform configuration details
    '''
    method = transform_config.get('method', 'origin')
    print(f"Transform method: {method}")
    if method in transform_config:
        print(f"Transform parameters: {transform_config[method]}")
    '''
    print(OmegaConf.to_yaml(config))

    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.policy_name_or_path, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.reference_name_or_path, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None
    
    print(f"[main] config type: {type(config)}")

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)

    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
