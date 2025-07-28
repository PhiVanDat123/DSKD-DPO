import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from criterions.cross_entropy_loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict
from utils.utils import slice_and_move_batch_for_device
from huggingface_hub import HfApi, HfFolder, Repository


# 2. Load mô hình và tokenizer
student_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B")

teacher_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Dataset
dataset = load_dataset("pvdhihihi/7B-weight-trasformed-v3")
train_dataset = dataset["train"]

# 4. Sử dụng cross entropy loss
criterion = CrossEntropyLoss(padding_id = -100)

tokenizer = {
    "teacher": teacher_tokenizer,
    "student": student_tokenizer,
}

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

train_iterator = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            #collate_fn=get_collate_fn(self.tokenizer),
            collate_fn=CustomCollate(tokenizer),
            pin_memory=True,
            num_workers=2,
            drop_last=True,
        )


# 6. Huấn luyện mô hình
optimizer = torch.optim.Adam(
    model.parameters() , lr= 1e-2
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, (step + 1) / ( 151))
)

gradient_accumulation_steps = 1
for batch in train_iterator:
    model.train()
    optimizer.zero_grad()
    for microbatch_idx in range(gradient_accumulation_steps):
        global_microbatch = slice_and_move_batch_for_device(
            batch, microbatch_idx, gradient_accumulation_steps, 0
        )
        local_microbatch = slice_and_move_batch_for_device(
            global_microbatch, 0, 8, 0
        )

        loss, _ =  criterion.compute_cross_entropy_loss(
            model(local_microbatch["chosen_student_input_ids"], local_microbatch["chosen_student_attention_mask"]).logits, 
            local_microbatch["chosen_student_labels"]
        )
        (loss / gradient_accumulation_steps).backward()
    optimizer.step()
    scheduler.step()

save_directory = "./llama_student_model"
model.save_pretrained(save_directory)
student_tokenizer.save_pretrained(save_directory)

print(f"Mô hình và tokenizer đã được lưu tại: {save_directory}")