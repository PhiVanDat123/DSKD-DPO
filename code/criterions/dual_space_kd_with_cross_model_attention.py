import math
import torch
from .various_divergence import VariousDivergence
from ..trainers import concatenated_inputs

class DualSpaceKDWithCMA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def forward(
        self, 
        distiller, 
        input_data,
        attention_mask=None,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        with torch.no_grad():
            teacher_model.eval()
            t2s_logits = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)

        return t2s_logits
    
    def compute_dual_space_kd_loss_with_cma(
        self, input_data, attention_mask, distiller
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        batch = {**input_data}
        concat_batch = concatenated_inputs(batch)
        #target = output_data["label"]
        #teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
        target = concat_batch["concatenated_labels"]
        teacher_target = concat_batch[f"concatenated_teacher_{distiller.teacher_model_type}_labels"]
        
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
        #formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))
        formal_input = torch.where(
        pad_mask,
        concat_batch["concatenated_input_ids"],
        torch.zeros_like(target)
        )
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        #formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(
        teacher_pad_mask,
        concat_batch[f"concatenated_teacher_{distiller.teacher_model_type}_input_ids"],
        torch.zeros_like(teacher_target),
        )
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
   
        return t2s_logits
