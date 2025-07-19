import math
import torch
from .various_divergence import VariousDivergence
from typing import Dict, List, Union
from utils.utils import pad_to_length
from datasets import load_dataset

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
    keys = [k for k in batch if mode in k]
    keys.extend([k for k in batch if "weight" in k])
    for k in keys:
        # if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
        if k.startswith("chosen"):
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
        if k.startswith("rejected"):
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

class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, config, padding_id=-100) -> None:
        super().__init__(config, padding_id=padding_id)
        self.distiller = None

    '''
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return loss / batch_denom, logging_output
    
    '''
    def compute_dual_space_kd_loss_with_cma(
        self, distiller
    ):  
        ds = load_dataset("tonyshelby/ultra-feedback_checking")
        self.distiller = distiller
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        concat_data = concatenated_inputs(ds["train"])
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                concat_data["concatenated_teacher_input_ids"],
                attention_mask=concat_data[f"concatenated_teacher_attention_mask"],
                #position_ids=concat_input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        #concat_output_data = concatenated_inputs(output_data)
        target = concat_data["concatenated_student_labels"]
        teacher_target = concat_data["concatenated_teacher_labels"]
        
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

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
        formal_input = torch.where(pad_mask, concat_data["concatenated_student_input_ids"], torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, concat_data[f"concatenated_teacher_input_ids"], torch.zeros_like(teacher_target))
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