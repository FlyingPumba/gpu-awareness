"""Padded physical batches with equal per-problem solution-token loss weights."""
import torch
import torch.nn.functional as F


def solution_loss(model, encoded_batch, pad_token_id):
    device = next(model.parameters()).device
    length = max(len(row["input_ids"]) for row in encoded_batch)
    ids = torch.full((len(encoded_batch), length), pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(encoded_batch):
        size = len(row["input_ids"])
        ids[index, :size] = torch.tensor(row["input_ids"], device=device)
        labels[index, :size] = torch.tensor(row["labels"], device=device)
        mask[index, :size] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    targets = labels[:, 1:]
    per_problem = []
    for row_logits, row_targets in zip(logits, targets):
        keep = row_targets != -100
        per_problem.append(F.cross_entropy(row_logits[keep].float(), row_targets[keep]))
    losses = torch.stack(per_problem)
    return losses.mean(), losses
