import torch


def generate_pos_neg_index(pseudo_label):
    batch_size = pseudo_label.size(0)
    invalid_index = pseudo_label == -1

    mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(pseudo_label.device)
    mask[invalid_index, :] = False
    mask[:, invalid_index] = False

    mask_eye = torch.eye(batch_size, dtype=torch.bool, device=pseudo_label.device)
    mask |= mask_eye

    valid_neg_choices = ~mask
    if valid_neg_choices.sum() == 0:
        neg_indices = torch.randperm(batch_size, device=pseudo_label.device)
        pos_indices = torch.arange(batch_size, device=pseudo_label.device)
        return pos_indices, neg_indices

    initial_indices = torch.randperm(batch_size, device=pseudo_label.device)
    global_non_cluster_mask = valid_neg_choices.float()
    replace_indices = torch.multinomial(global_non_cluster_mask, num_samples=1, replacement=True).squeeze(1)

    need_replacement = ~valid_neg_choices[torch.arange(batch_size, device=pseudo_label.device), initial_indices]
    initial_indices[need_replacement] = replace_indices[need_replacement]
    neg_indices = initial_indices

    pos_choices = (~valid_neg_choices).float()
    pos_indices = torch.multinomial(pos_choices, num_samples=1, replacement=True).squeeze(1)
    return pos_indices, neg_indices
