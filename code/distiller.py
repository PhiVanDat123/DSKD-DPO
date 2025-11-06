import os
import json
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from utils.utils import log_rank


class Distiller(nn.Module):
    def __init__(self, config, device):
        super(Distiller, self).__init__()
        self.config = config
        self.device = device
        self.student_model_type = config.model_type
        self.student_model, self.student_tokenizer = self.load_student_model()
        
        if self.config.reference_name_or_path is not None:
            self.teacher_model, self.teacher_tokenizers = self.load_teacher_model()
        else:
            self.teacher_model, self.teacher_tokenizers = None, {}
        self.teacher_model_type = config.teacher_model_type

        if self.teacher_model and config.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")
            self.to(self.device)
        
        
    def load_tokenizer(self, model_type, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif model_type == "qwen":
            # tokenizer.pad_token_id = 151646
            tokenizer.eos_token_id = 151643
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer

    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.config.projector_config_path))
        '''
        name_dict = {
            "s": self.student_hidden_size, 
            "t": self.teacher_hidden_size,
            "relu": nn.ReLU()
        }
        '''
        '''
        name_dict = {
            "s": 4096, #8B 
            "t": 5120, #14B
            "relu": nn.ReLU()
        }
        '''
        
        name_dict = {
            "s": 2048, #1B 
            "t": 3584, #7B
            "relu": nn.ReLU()
        }
        
        '''
        name_dict = {
            "s": 896, #1B 
            "t": 1024, #7B
            "relu": nn.ReLU()
        }
        '''
        # auto-parse projector config strings to construct nn.Module
        for projector_name in projector_config:
            # for d in projector_config[loc]:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for i in range(len(structure)):
                    if structure[i] not in ["relu"]:
                        coef = 1 if not len(structure[i][:-1]) else int(structure[i][:-1])
                        base_size = name_dict[structure[i][-1]]
                        structure[i] = coef * base_size

                for i in range(len(structure) - 1):
                    if isinstance(structure[i], int) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[i], structure[i+1])
                        )
                    elif isinstance(structure[i], int) and isinstance(structure[i+1], str):
                        self.projectors[projector_name].append(
                            name_dict[structure[i+1]]
                        )
                        last_size = structure[i]
                    elif isinstance(structure[i], str) and isinstance(structure[i+1], int):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[i+1])
                        )
                    else:
                        raise NotImplementedError(f"Invalid structure for '{structure}'")
                        
        # load existing projectors if already have
        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.config.projector_path is not None:
            #projector_path = os.path.join(self.config.projector_path, "projector.pt")
            projector_path = self.config.projector_path
        else:
            projector_path = os.path.join(self.config.policy_name_or_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(projector_path, map_location=f"cuda:{self.device}")
            log_rank("Existing projector params: {}".format(list(projector_params.keys())))
            for key in self.projectors:
                try:
                    state_dict = {
                        n.split('.', 1)[1]: projector_params[n] for n in projector_params if n.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank("Load projector '{}' from current path.".format(key))
                except:
                    log_rank("Not compatible for projector '{}'".format(key))
                    continue
    
    def load_student_model(self):
        log_rank("Loading student model...")
        config = AutoConfig.from_pretrained(self.config.policy_name_or_path, trust_remote_code=True)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.config.model_type, self.config.policy_name_or_path)
        
        if hasattr(config, "n_embed"):
            self.student_hidden_size = config.n_embed
        else:
            self.student_hidden_size = config.hidden_size
        
        if self.config.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.config.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.config.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError("Invalid model_dtype for f`{self.config.model_dtype}`")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_name_or_path, 
            config=config, 
            device_map=None, 
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer
    
    def load_teacher_model(self):
        log_rank("Loading teacher model...")
        config = AutoConfig.from_pretrained(self.config.reference_name_or_path)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.config.teacher_model_type, self.config.reference_name_or_path)

        if hasattr(config, "n_embed"):
            self.teacher_hidden_size = config.n_embed
        else:
            self.teacher_hidden_size = config.hidden_size

        model = AutoModelForCausalLM.from_pretrained(
            self.config.reference_name_or_path, 
            config=config, 
            device_map=None, 
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        for params in model.parameters():
            params.requires_grad = False
        return model, {self.config.teacher_model_type: tokenizer}
    
    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors"):
            if self.config.projector_lr:
                pretrained_proj = self.config.pretrained_projector.split(",") if self.config.pretrained_projector is not None else []
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b not in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.config.projector_lr
                })
                optimizer.add_param_group({
                    "params": [p for b in self.projectors if b in pretrained_proj for p in self.projectors[b].parameters()],
                    "lr": self.config.pretrained_projector_lr
                })
            else:
                optimizer.add_param_group({
                    "params": [p for b in self.projectors for p in self.projectors[b].parameters()],
                })
        return optimizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data, 
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
