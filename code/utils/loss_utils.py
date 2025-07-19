from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .utils import drop_leading_zeros_batch, drop_zr_cols_and_padded, pad_to_length

def mask_from_neg100(x):
    """
    Replace padding (-100) with 0, but set the last padding token
    before real tokens start to 1.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_tokens).
        pad_val (int): The padding value to replace. Default is -100.

    Returns:
        torch.Tensor: The processed tensor.
    """
    # Find the first real token index
    pad_val = -100
    out = (x != pad_val).long()

    first_real_token_idx = (x != pad_val).float().argmax(dim=1)

    marker_col_idx = (first_real_token_idx - 1).clamp(min=0)

    # Set the last padding (before real tokens) to 1
    row_idx = torch.arange(x.size(0), device=x.device)
    if x.size(1) > 0:
        out[row_idx, marker_col_idx] = 1

    return out

def prompt_remove(logits, labels, input_ids):
    masked_labels = mask_from_neg100(labels)  # (-1,n) becomes 1, else becomse 0
    masked_logits = masked_labels.unsqueeze(-1) * logits

    no_prompt_logits = drop_zr_cols_and_padded(masked_logits)

    print(f"No prompt logits shape {no_prompt_logits.shape}")

    masked_input_ids = masked_labels * input_ids
    no_prompt_labels = pad_sequence(
        drop_leading_zeros_batch(masked_input_ids), batch_first=True, padding_value=0
    )
    print(f"No prompt labels shape {no_prompt_labels.shape}")

    return no_prompt_logits, no_prompt_labels

def get_token_logps(
    logits: torch.FloatTensor,  # shape: (batch_size, num_tokens, vocab_size) # already normalized (-1, n)
    labels: torch.LongTensor,  # shape: (batch_size, num_tokens) # only the tokens in the response (-1, n)
    #p_tokens_list: torch.LongTensor,  # batch_size * num_parent_tokens *  # data: offset of ptoken: index of token in seq len
) -> torch.Tensor:  # shape: batch_size * max_parent_tokens
    assert logits.shape[:-1] == labels.shape

    # mask?
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    # loss_mask = labels == -100

    labels[labels == -100] = 0

    rank = torch.distributed.get_rank()  # Get rank if in distributed setting
    print(f"[Rank {rank}] logits shape: {logits.shape}")
    print(f"[Rank {rank}] labels shape (before gather): {labels.shape}")
    print(f"[Rank {rank}] labels min value: {labels.min()}")
    print(f"[Rank {rank}] labels max value: {labels.max()}")
    print(f"[Rank {rank}] Expected max label index: {logits.shape[-1] - 1}")  # vocab_size - 1

    per_token_prob = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(
        2
    )  # shape: (batch_size, chosen response)
    # rank = torch.distributed.get_rank()
    # print(f"Rank {rank}: per_token_prob shape: {per_token_prob.shape}")
    # print(f"Rank {rank}: p_tokens_list shape: {p_tokens_list.shape}")
    # print(f"Rank {rank}: p_tokens_list min: {p_tokens_list.min()}, max: {p_tokens_list.max()}")
    # # Crucially, check the condition directly:
    # response_len = per_token_prob.shape[1]
    # invalid_mask = (p_tokens_list >= response_len) & (p_tokens_list != -1)
    # if torch.any(invalid_mask):
    #     print(f"Rank {rank}: FOUND INVALID INDICES!")
    #     print(f"Rank {rank}: Response len: {response_len}")
    #     offending_indices = p_tokens_list[invalid_mask]
    #     print(f"Rank {rank}: Offending indices: {offending_indices.unique().tolist()}")
    #     # You might want to raise an error here immediately for debugging
    #     # raise ValueError("Invalid indices detected before gather_with_padding")
    # else:
    #     print(f"Rank {rank}: All indices seem valid relative to response_len {response_len}.")

    #p_token_with_prob = gather_with_padding(
    #    per_token_prob, p_tokens_list, padding_value=-1, fill_value=0.0
    #)

    #result = torch.sum(p_token_with_prob, dim=-1)

    #return result
    return per_token_prob

    # # Prepare batch processing
    # batch_size = len(p_tokens_dict)
    # max_parent_tokens = max(len(d) for d in p_tokens_dict)

    # # Create tensors to hold results
    # # results = torch.ones(batch_size, max_parent_tokens, device=per_token_prob.device)
    # results = torch.ones(batch_size, max_parent_tokens)
    # mask = torch.zeros(batch_size, max_parent_tokens, dtype=torch.bool)

    # # Process each batch item
    # for i, p_dict in enumerate(p_tokens_dict):
    #     for j, (_, token_indices) in enumerate(p_dict.items()):
    #         # Use indexing to get all probabilities for this parent token
    #         token_probs = per_token_prob[i, token_indices]
    #         # Multiply all probabilities (product along dimension 0)
    #         results[i, j] = torch.sum(token_probs)
    #         mask[i, j] = True

    # # Apply mask to handle variable number of parent tokens per batch
    # masked_results = results * mask

    # # If you need a ragged tensor (list of tensors with different lengths)
    # # res = [masked_results[i, :len(p_dict)] for i, p_dict in enumerate(p_tokens_dict)]
    # return masked_results
    # # return torch.stack([r[r != 0] for r in res])
