import math
import torch
from .various_divergence import VariousDivergence
from typing import Dict, List, Union
from utils.utils import pad_to_length
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext, ExitStack

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
    def compute_batch_dual_space_kd_loss_with_cma(
        self, concat_student_data, concat_teacher_data, distiller, model, reference_model
    ):  
        device = next(model.parameters()).device  # Lấy device của mô hình
        #batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}  # Di chuyển các tensor trong batch
        self.distiller = distiller
        model = model.to(device)
        print("[dskd] Model device:", model.device)
        teacher_model = reference_model.to(device)
        print("[dskd] Teacher model device:", teacher_model.device)
        #print("[DEBUG] Max input ID:", concat_student_data["concatenated_student_input_ids"].max().item())
        #print("[DEBUG] Vocab size:", model.config.vocab_size)
        #print("[DEBUG] Input IDs shape:", concat_student_data["concatenated_student_input_ids"].shape)


        teacher_model.eval()
        teacher_outputs = teacher_model(
            concat_teacher_data["concatenated_teacher_input_ids"].to(device),
            attention_mask=concat_teacher_data[f"concatenated_teacher_attention_mask"].to(device),
            #position_ids=concat_input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
            output_hidden_states=True)
            
        #concat_output_data = concatenated_inputs(output_data)
        target = concat_student_data["concatenated_student_labels"].to(device)
        teacher_target = concat_teacher_data["concatenated_teacher_labels"].to(device)
            
        pad_mask = target.ne(self.padding_id).to(device)
        teacher_pad_mask = teacher_target.ne(self.padding_id).to(device)

        teacher_hiddens = teacher_outputs.hidden_states[-1].to(device)

        if hasattr(model, "model") \
            and hasattr(model.model, "embed_tokens"):
            stu_embed_tokens = model.model.embed_tokens
        elif hasattr(model, "model") \
            and hasattr(model.model, "model") \
            and hasattr(model.model.model, "embed_tokens"):
            stu_embed_tokens = model.model.model.embed_tokens
        elif hasattr(model, "transformer") \
            and hasattr(model.transformer, "word_embeddings"):
            stu_embed_tokens = model.transformer.word_embeddings
        else:
            raise NotImplementedError

        stu_embed_tokens = stu_embed_tokens.to(device)


        
        if hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens
        elif hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "model") \
            and hasattr(teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.model.embed_tokens
        elif hasattr(teacher_model, "transformer") \
            and hasattr(teacher_model.model, "wte"):
            tea_embed_tokens = teacher_model.transformer.wte
        else:
            raise NotImplementedError
        
        '''
        tea_embed_tokens = getattr(
            getattr(teacher_model, "module", teacher_model).transformer,
            "word_embeddings"
        )
        '''
        tea_embed_tokens = tea_embed_tokens.to(device)
            

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target)).to(device)
        print("[dskd] formal_target device:", formal_target.device)
        formal_input = torch.where(pad_mask, concat_student_data["concatenated_student_input_ids"].to(device), torch.zeros_like(target)).to(device)
        print("[dskd] formal_input device:", formal_input.device)
        print("formal_input:", formal_input)
        stu_input_embeds = stu_embed_tokens(formal_input).detach()   
        stu_target_embeds = stu_embed_tokens(formal_target).detach().to(device)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target)).to(device)
        formal_teacher_input = torch.where(teacher_pad_mask, concat_teacher_data[f"concatenated_teacher_input_ids"].to(device), torch.zeros_like(teacher_target)).to(device)
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach().to(device)
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach().to(device)

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1).to(device)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1).to(device)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        distiller.projectors["query"] = distiller.projectors["query"].to(device)
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float().to(device)
        tea_k_hiddens = norm_tea_index_embeds.float().to(device)

        distiller.projectors["t2s"] = distiller.projectors["t2s"].to(device)
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float().to(device)
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)      
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            model.lm_head.weight.detach().transpose(-1, -2).to(device)
        ).to(device)
        
        return t2s_logits, target
    
    
    def compute_dual_space_kd_loss_with_cma(
        self, batch, distiller, model, reference_model
    ):  
        device = next(model.parameters()).device  # Lấy device của mô hình
        #batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}  # Di chuyển các tensor trong batch
        self.distiller = distiller
        model = model.to(device)
        print("[dskd] Model device:", model.device)
        teacher_model = reference_model.to(device)
        print("[dskd] Teacher model device:", teacher_model.device)
        #print("[DEBUG] Max input ID:", concat_student_data["concatenated_student_input_ids"].max().item())
        #print("[DEBUG] Vocab size:", model.config.vocab_size)
        #print("[DEBUG] Input IDs shape:", concat_student_data["concatenated_student_input_ids"].shape)

        
        teacher_model.eval()
        teacher_outputs = teacher_model(
            batch["chosen_teacher_input_ids"].to(device),
            attention_mask=batch[f"chosen_teacher_attention_mask"].to(device),
            #position_ids=concat_input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
            output_hidden_states=True)
         
        '''
        with torch.no_grad():
            model.eval()
            outputs = model(
                concat_student_data["concatenated_student_input_ids"].to(device),
                attention_mask=concat_student_data[f"concatenated_student_attention_mask"].to(device),
                output_hidden_states=True).logits.to(device)
            print("[DEBUG] outputs logits shape:", outputs.shape)
        '''
    
        #concat_output_data = concatenated_inputs(output_data)
        target = batch["chosen_student_labels"].to(device)
        teacher_target = batch["chosen_teacher_labels"].to(device)
            
        pad_mask = target.ne(self.padding_id).to(device)
        teacher_pad_mask = teacher_target.ne(self.padding_id).to(device)

        teacher_hiddens = teacher_outputs.hidden_states[-1].to(device)

        if hasattr(model, "model") \
            and hasattr(model.model, "embed_tokens"):
            stu_embed_tokens = model.model.embed_tokens
        elif hasattr(model, "model") \
            and hasattr(model.model, "model") \
            and hasattr(model.model.model, "embed_tokens"):
            stu_embed_tokens = model.model.model.embed_tokens
        elif hasattr(model, "transformer") \
            and hasattr(model.transformer, "word_embeddings"):
            stu_embed_tokens = model.transformer.word_embeddings
        else:
            raise NotImplementedError

        stu_embed_tokens = stu_embed_tokens.to(device)
    
        
        
        if hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.embed_tokens
        elif hasattr(teacher_model, "model") \
            and hasattr(teacher_model.model, "model") \
            and hasattr(teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = teacher_model.model.model.embed_tokens
        elif hasattr(teacher_model, "transformer") \
            and hasattr(teacher_model.model, "wte"):
            tea_embed_tokens = teacher_model.transformer.wte
        else:
            raise NotImplementedError
        
        '''
        tea_embed_tokens = getattr(
            getattr(teacher_model, "module", teacher_model).transformer,
            "word_embeddings"
        )
        '''
        tea_embed_tokens = tea_embed_tokens.to(device)
            

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target)).to(device)
        print("[dskd] formal_target device:", formal_target.device)
        formal_input = torch.where(pad_mask, batch["chosen_student_input_ids"].to(device), torch.zeros_like(target)).to(device)
        print("[dskd] formal_input device:", formal_input.device)
        print("formal_input:", formal_input)
        stu_input_embeds = stu_embed_tokens(formal_input).detach()   
        stu_target_embeds = stu_embed_tokens(formal_target).detach().to(device)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target)).to(device)
        formal_teacher_input = torch.where(teacher_pad_mask, batch[f"chosen_teacher_input_ids"].to(device), torch.zeros_like(teacher_target)).to(device)
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach().to(device)
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach().to(device)

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1).to(device)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1).to(device)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        distiller.projectors["query"] = distiller.projectors["query"].to(device)
        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds).float().to(device)
        tea_k_hiddens = norm_tea_index_embeds.float().to(device)

        distiller.projectors["t2s"] = distiller.projectors["t2s"].to(device)
        tea_v_hiddens = distiller.projectors["t2s"](
            norm_teacher_hiddens + norm_tea_target_embeds
        ).float().to(device)
        
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        t2s_weight = torch.softmax(align, -1)        
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            model.lm_head.weight.detach().transpose(-1, -2).to(device)
        ).to(device)
        
        return t2s_logits, target
    
    def compute_dtw_and_alignment_kd_losses(self, concat_student_data, concat_teacher_data, distiller, model, reference_model):
        device = next(model.parameters()).device  
        model = model.to(device)
        teacher_model = reference_model.to(device)
        teacher_model.eval()

        # === Forward student ===
        outputs = model(
            concat_student_data["chosen_student_input_ids"].to(device),
            attention_mask=concat_student_data["chosen_student_attention_mask"].to(device),
            output_hidden_states=True
        )

        # === Forward teacher ===
        teacher_outputs = teacher_model(
            concat_teacher_data["chosen_teacher_input_ids"].to(device),
            attention_mask=concat_teacher_data["chosen_teacher_attention_mask"].to(device),
            output_hidden_states=True
        )

        target = concat_student_data["chosen_student_labels"].to(device)
        pad_mask = target.ne(self.padding_id)
        teacher_target = concat_student_data["chosen_teacher_labels"].to(device)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        dtw_proj_total = torch.tensor(0.0, device=hiddens.device, requires_grad=True)
        align_kd_total = torch.tensor(0.0, device=hiddens.device, requires_grad=True)

        for i in range(hiddens.size(0)):
            s_len = pad_mask[i].sum().item()
            t_len = teacher_pad_mask[i].sum().item()
            if s_len == 0 or t_len == 0:
                continue

            s_seq = hiddens[i, :s_len, :]   
            t_seq = teacher_hiddens[i, :t_len, :]

            t_proj = distiller.projectors["dtw_hidden_t2s"](t_seq)  # (t_len, d_stu)

            C = 1.0 - torch.cosine_similarity(
                s_seq.detach().unsqueeze(1), t_proj.unsqueeze(0), dim=-1
            )  # (s_len, t_len)

            # soft-DTW divergence in student space: dtw(S,T') - 1/2(dtw(S,S) + dtw(T',T'))
            dtw_xy_stu, A = self.dtw.forward_with_cost_matrix(C.unsqueeze(0), return_alignment=True)
            C_ss = 1.0 - torch.cosine_similarity(
                s_seq.detach().unsqueeze(1), s_seq.detach().unsqueeze(0), dim=-1
            )  # (s_len, s_len)
            C_tt = 1.0 - torch.cosine_similarity(
                t_proj.unsqueeze(1), t_proj.unsqueeze(0), dim=-1
            )  # (t_len, t_len)
            dtw_ss_stu = self.dtw.forward_with_cost_matrix(C_ss.unsqueeze(0))
            dtw_tt_stu = self.dtw.forward_with_cost_matrix(C_tt.unsqueeze(0))
            dtw_div_stu = dtw_xy_stu - 0.5 * (dtw_ss_stu + dtw_tt_stu)
            dtw_proj_total = dtw_proj_total + dtw_div_stu.squeeze()

            A = A.detach()
            t_aligned = torch.einsum("bnm,bmd->bnd", A, t_proj.unsqueeze(0).detach())  # (1, s_len, d_stu)

            with torch.no_grad():
                W_s = distiller.student_model.lm_head.weight  # (V_s, d_stu)
                l_ts = t_aligned.matmul(W_s.transpose(-1, -2))  # (1, s_len, V_s)

            l_s = outputs.logits[i, :s_len, :].unsqueeze(0)

            kd_i = self.compute_forward_kl_divergence(
                l_s, l_ts, target[i, :s_len].unsqueeze(0), reduction="sum"
            )
            align_kd_total = align_kd_total + kd_i

            proj_mod = distiller.projectors["dtw_hidden_t2s"]
            W = proj_mod[0].weight  # (d_stu, d_tea)
            s_proj = torch.matmul(s_seq, W.detach())        # (s_len, d_tea)
            s_proj_det = torch.matmul(s_seq.detach(), W)    # (s_len, d_tea)

            # soft-DTW divergence in teacher space: dtw(T, S_proj) - 1/2(dtw(T,T) + dtw(S_proj,S_proj))
            C_teach_xy = 1.0 - torch.cosine_similarity(
                t_seq.unsqueeze(1), s_proj_det.unsqueeze(0), dim=-1
            )  # (t_len, s_len)
            dtw_xy_teach, B = self.dtw.forward_with_cost_matrix(C_teach_xy.unsqueeze(0), return_alignment=True)
            C_teach_xx = 1.0 - torch.cosine_similarity(
                t_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
            )  # (t_len, t_len)
            C_teach_yy = 1.0 - torch.cosine_similarity(
                s_proj_det.unsqueeze(1), s_proj_det.unsqueeze(0), dim=-1
            )  # (s_len, s_len)
            dtw_xx_teach = self.dtw.forward_with_cost_matrix(C_teach_xx.unsqueeze(0))
            dtw_yy_teach = self.dtw.forward_with_cost_matrix(C_teach_yy.unsqueeze(0))
            dtw_div_teach = dtw_xy_teach - 0.5 * (dtw_xx_teach + dtw_yy_teach)
            dtw_proj_total = dtw_proj_total + dtw_div_teach.squeeze()

            B = B.detach()
            s_proj_b = s_proj.unsqueeze(0)
            s_aligned = torch.einsum("bnm,bmd->bnd", B, s_proj_b)  # (1, t_len, d_tea)

            W_t = distiller.teacher_model.lm_head.weight.detach()  # (V_t, d_tea)
            l_st = s_aligned.matmul(W_t.transpose(-1, -2))  # (1, t_len, V_t)

            l_t = teacher_outputs.logits[i, :t_len, :].unsqueeze(0)

            kd_t = self.compute_forward_kl_divergence(
                l_st, l_t, teacher_target[i, :t_len].unsqueeze(0), reduction="sum"
            )
            align_kd_total = align_kd_total + kd_t

        return dtw_proj_total, align_kd_total

        
    def _get_target_embeddings(self, distiller, input_data, output_data, pad_mask, teacher_pad_mask):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        
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
        stu_target_embeds = stu_embed_tokens(formal_target)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        return stu_target_embeds, tea_target_embeds
