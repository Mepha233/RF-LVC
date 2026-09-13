"""Module 3 - layer-wise contrastive training (RF-LVC).

Implements the representation-learning stage of the paper:
  * two-view cropping and augmentation (see also module 1),
  * temporal-mode and instance-mode InfoNCE losses,
  * receptive-field-aware contrast scheduling across encoder levels,
  * final-layer hierarchical contrasting,
  * the grouped training objective and the AdamW pretraining loop with a
    parameter-averaged (SWA/Polyak) encoder for feature extraction.

Cropping, losses and the training loop are methods of FK1Model in the
original code base and are therefore kept in a single file instead of
being split artificially across modules.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import ReduceLROnPlateau

from module1_crop_aug.augmentations import DataTransform
from module2_encoder import FK1Encoder
from module2_encoder.Metrics import acc, nmi
from module3_contrastive_training.negatives import generate_pos_neg_index


class FK1Model:
    """Cross-layer contrastive pretraining with final-layer-only clustering evaluation."""

    def __init__(
        self,
        data_loader,
        eval_loader,
        dataset_size,
        timesteps_len,
        batch_size,
        pretraining_epoch,
        n_cluster,
        dataset_name,
        input_dims,
        output_dims=32,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        hard_w=0.2,
        instance_activation_ratio=0.8,
        log_dir='logs',
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.num_cluster = n_cluster
        self.batch_size = batch_size
        self.pretraining_epoch = pretraining_epoch
        self.train_loader = data_loader
        self.eval_loader = eval_loader if eval_loader is not None else data_loader
        self.dataset_size = dataset_size
        self.timesteps_len = timesteps_len
        self.input_dims = input_dims
        self.dataset_name = dataset_name
        self.hard_w = hard_w
        self.instance_activation_ratio = instance_activation_ratio
        self.logdir = log_dir
        os.makedirs(self.logdir, exist_ok=True)
        self.scaling_rate = 0.8

        self.encoder = FK1Encoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
        ).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self.encoder)
        self.net.update_parameters(self.encoder)
        self.layer_output_dims = list(self.encoder.layer_output_dims)
        self.layer_receptive_fields = list(self.encoder.layer_receptive_fields)
        self.instance_start_layer = self._find_instance_start_layer()
        self.group_budgets = self._group_loss_budgets()

        threshold = int(np.ceil(self.timesteps_len * self.instance_activation_ratio))
        start_rf = self.layer_receptive_fields[self.instance_start_layer]
        print(
            f"[Loss Schedule] receptive_fields={self.layer_receptive_fields} | "
            f"instance_loss_starts=L{self.instance_start_layer:02d} "
            f"(rf={start_rf}, threshold={threshold})"
        )
        print(
            "[Loss Budget] "
            f"shallow_temp={self.group_budgets['shallow_temp']:.3f} "
            f"deep_temp={self.group_budgets['deep_temp']:.3f} "
            f"deep_inst={self.group_budgets['deep_inst']:.3f} "
            f"final_pyramid={self.group_budgets['final_pyramid']:.3f}"
        )

    def get_inference_state_dict(self):
        return self.net.module.state_dict()

    def _find_instance_start_layer(self):
        threshold = int(np.ceil(self.timesteps_len * self.instance_activation_ratio))
        for idx, receptive_field in enumerate(self.layer_receptive_fields):
            if receptive_field >= threshold:
                return idx
        return len(self.layer_receptive_fields) - 1

    def _layer_loss_weights(self):
        num_layers = len(self.layer_output_dims)
        if num_layers == 1:
            return [(0.5, 0.5)]

        weights = []
        deep_count = num_layers - self.instance_start_layer
        for idx in range(num_layers):
            if idx < self.instance_start_layer:
                temp_weight = 1.0
                inst_weight = 0.0
            else:
                if deep_count == 1:
                    progress = 1.0
                else:
                    progress = (idx - self.instance_start_layer) / float(deep_count - 1)
                temp_weight = 0.8 - 0.3 * progress
                inst_weight = 0.2 + 0.3 * progress
            weights.append((temp_weight, inst_weight))
        return weights

    def _group_loss_budgets(self):
        budgets = {
            'shallow_temp': 0.2 if self.instance_start_layer > 0 else 0.0,
            'deep_temp': 0.2 if self.instance_start_layer < len(self.layer_output_dims) else 0.0,
            'deep_inst': 0.4 if self.instance_start_layer < len(self.layer_output_dims) else 0.0,
            'final_pyramid': 0.2,
        }
        total = sum(budgets.values())
        if total == 0:
            budgets['final_pyramid'] = 1.0
            total = 1.0
        return {key: value / total for key, value in budgets.items()}

    @staticmethod
    def _mean_loss(losses, reference):
        if not losses:
            return reference.sum() * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _weighted_mean_loss(losses, weights, reference):
        if not losses:
            return reference.sum() * 0.0
        weight_tensor = torch.tensor(weights, device=reference.device, dtype=reference.dtype)
        loss_tensor = torch.stack(losses)
        return (loss_tensor * weight_tensor).sum() / weight_tensor.sum()

    def Pretraining(self):
        print('Pretraining...')
        self.encoder.train()
        for param in self.encoder.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.encoder.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=12)

        loss_log = []
        acc_log = []
        nmi_log = []

        for epoch in range(self.pretraining_epoch):
            print('Pretraining Epoch: ', epoch + 1)
            total_loss = 0.0
            num_batches = 0

            for x, _, _ in self.train_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                view1, view2 = self.crops_and_extract(x, scaling_rate=self.scaling_rate)
                outputs_view1 = self.encoder(view1, return_all_layers=True)
                outputs_view2 = self.encoder(view2, return_all_layers=True)
                loss, _ = self.multilayer_contrastive_loss(outputs_view1, outputs_view2)
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.encoder)
                total_loss += loss.item()
                num_batches += 1

            average_loss = total_loss / max(1, num_batches)
            loss_log.append(average_loss)
            scheduler.step(average_loss)

            epoch_acc, epoch_nmi = self.Kmeans_model_evaluation(epoch)
            acc_log.append(epoch_acc)
            nmi_log.append(epoch_nmi)
            print(f"Epoch #{epoch + 1}: loss={average_loss}")

        log_path = os.path.join(self.logdir, f'pretraining_{self.dataset_name}.csv')
        pd.DataFrame.from_dict(
            {'pretraining': loss_log, 'ACC': acc_log, 'NMI': nmi_log},
            orient='index',
        ).to_csv(log_path, index=False)
        print(f'Training log saved to: {log_path}')
        return self.encoder

    def _collect_final_pooled_embeddings(self, data_loader, model):
        model.eval()
        embeddings = []
        label_true = []
        with torch.no_grad():
            for x, target, _ in data_loader:
                x = x.to(self.device)
                final_output = model(x, return_all_layers=True)[-1]
                pooled = F.max_pool1d(
                    final_output.transpose(1, 2),
                    kernel_size=final_output.size(1),
                ).transpose(1, 2).squeeze(1)
                embeddings.append(pooled.detach().cpu().numpy())
                label_true.append(target.numpy())
        model.train()
        return np.concatenate(embeddings, axis=0), np.concatenate(label_true, axis=0)

    def _evaluate_final_layer(self, data_loader, model):
        embeddings, label_true = self._collect_final_pooled_embeddings(data_loader, model)
        label_pred = KMeans(n_clusters=self.num_cluster, random_state=0, n_init=10).fit(embeddings).labels_
        return {
            'ACC': acc(label_true, label_pred, self.num_cluster),
            'NMI': nmi(label_true, label_pred),
        }

    def Kmeans_model_evaluation(self, epoch_idx):
        metrics = self._evaluate_final_layer(self.eval_loader, self.net)
        print('ACC', metrics['ACC'])
        print('NMI', metrics['NMI'])
        return metrics['ACC'], metrics['NMI']

    def mask_instance_loss_with_mixup(self, z1, z2, pseudo_label=None):
        batch_size, time_steps = z1.size(0), z1.size(1)
        if pseudo_label is None:
            pseudo_label = torch.full((batch_size,), -1, dtype=torch.int64, device=z1.device)

        if batch_size == 1:
            return z1.sum() * 0.0

        pseudo_label = pseudo_label.to(z1.device)
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        hard_w = self.hard_w

        uni_z1 = hard_w * z1[pos_indices, :, :] + (1 - hard_w) * z1[neg_indices, :, :].view(z1.size())
        pos_indices, neg_indices = generate_pos_neg_index(pseudo_label)
        uni_z2 = hard_w * z2[pos_indices, :, :] + (1 - hard_w) * z2[neg_indices, :, :].view(z2.size())

        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=0).transpose(0, 1)
        sim = torch.matmul(z[:, : 2 * batch_size, :], z.transpose(1, 2))

        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).to(z1.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False

        mask_eye = torch.eye(batch_size, dtype=torch.float32, device=z1.device)
        mask &= ~(mask_eye.bool())
        mask = mask.float().repeat(2, 4)
        mask_eye = mask_eye.repeat(2, 4)

        logits_mask = torch.ones(2 * batch_size, 4 * batch_size, device=z1.device)
        rows = torch.arange(2 * batch_size, device=z1.device).view(-1, 1)
        logits_mask = logits_mask.scatter(1, rows, 0)
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        logits_max = torch.max(sim, dim=-1, keepdim=True)[0]
        logits = sim - logits_max
        neg_exp_logits = torch.exp(logits) * logits_mask
        pos_exp_logits = torch.exp(logits)
        neg_exp_log_sum = neg_exp_logits.sum(-1, keepdim=True)
        prob = pos_exp_logits / (neg_exp_log_sum + 1e-10)
        prob = prob[:, 0:batch_size, batch_size:2 * batch_size]

        mask = mask[:batch_size, :batch_size]
        self_mask = mask_eye[:batch_size, batch_size:2 * batch_size]
        pos_mask = self_mask + mask
        pos_prob_sum = (prob * pos_mask.unsqueeze(0)).sum(-1)
        log_prob = torch.log(pos_prob_sum + 1e-10)
        log_prob = log_prob.sum(dim=0) / time_steps
        return (-log_prob).mean()

    def temporal_contrastive_loss_mixup(self, z1, z2, temp=1.0):
        batch_size, time_steps = z1.size(0), z1.size(1)
        if time_steps == 1:
            return z1.sum() * 0.0

        alpha = 0.2
        beta = 0.2
        uni_z1 = alpha * z1 + (1 - alpha) * z1[:, torch.randperm(z1.shape[1]), :].view(z1.size())
        uni_z2 = beta * z2 + (1 - beta) * z2[:, torch.randperm(z2.shape[1]), :].view(z2.size())

        z = torch.cat([z1, z2, uni_z1, uni_z2], dim=1)
        sim = torch.matmul(z[:, : 2 * time_steps, :], z.transpose(1, 2)) / temp
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]

        if time_steps > 1500:
            z = z.cpu()
            sim = sim.cpu()
            torch.cuda.empty_cache()

        logits = -F.log_softmax(logits, dim=-1)
        logits = logits[:, :2 * time_steps, :(2 * time_steps - 1)]
        t = torch.arange(time_steps, device=z1.device)
        return (logits[:, t, time_steps + t - 1].mean() + logits[:, time_steps + t, t].mean()) / 2

    def final_layer_pyramid_contrastive_loss(self, z1, z2, alpha=0.5, temporal_unit=0):
        loss = z1.sum() * 0.0
        depth = 0

        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.mask_instance_loss_with_mixup(z1, z2)
            if depth >= temporal_unit and (1 - alpha) != 0:
                loss += (1 - alpha) * self.temporal_contrastive_loss_mixup(z1, z2)

            depth += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.mask_instance_loss_with_mixup(z1, z2)
            depth += 1

        return loss / max(1, depth)

    def multilayer_contrastive_loss(self, outputs_view1, outputs_view2):
        weights = self._layer_loss_weights()
        reference = outputs_view1[0]
        layer_stats = []
        shallow_temp_losses = []
        deep_temp_losses = []
        deep_temp_weights = []
        deep_inst_losses = []
        deep_inst_weights = []

        for layer_idx, (z1, z2) in enumerate(zip(outputs_view1, outputs_view2)):
            temp_weight, inst_weight = weights[layer_idx]
            if temp_weight > 0:
                temp_loss = self.temporal_contrastive_loss_mixup(z1, z2)
            else:
                temp_loss = z1.sum() * 0.0
            inst_loss = self.mask_instance_loss_with_mixup(z1, z2) if inst_weight > 0 else z1.sum() * 0.0

            if layer_idx < self.instance_start_layer:
                shallow_temp_losses.append(temp_loss)
            else:
                if temp_weight > 0:
                    deep_temp_losses.append(temp_loss)
                    deep_temp_weights.append(temp_weight)
                if inst_weight > 0:
                    deep_inst_losses.append(inst_loss)
                    deep_inst_weights.append(inst_weight)

            layer_stats.append(
                {
                    'layer': layer_idx,
                    'temp_weight': temp_weight,
                    'inst_weight': inst_weight,
                    'temp_loss': float(temp_loss.detach().cpu().item()),
                    'inst_loss': float(inst_loss.detach().cpu().item()),
                    'loss': float((temp_weight * temp_loss + inst_weight * inst_loss).detach().cpu().item()),
                }
            )

        shallow_temp_avg = self._mean_loss(shallow_temp_losses, reference)
        deep_temp_avg = self._weighted_mean_loss(deep_temp_losses, deep_temp_weights, reference)
        deep_inst_avg = self._weighted_mean_loss(deep_inst_losses, deep_inst_weights, reference)
        final_pyramid_loss = self.final_layer_pyramid_contrastive_loss(outputs_view1[-1], outputs_view2[-1])

        budgets = self.group_budgets
        total_loss = (
            budgets['shallow_temp'] * shallow_temp_avg
            + budgets['deep_temp'] * deep_temp_avg
            + budgets['deep_inst'] * deep_inst_avg
            + budgets['final_pyramid'] * final_pyramid_loss
        )

        layer_stats.append(
            {
                'shallow_temp_avg': float(shallow_temp_avg.detach().cpu().item()),
                'deep_temp_avg': float(deep_temp_avg.detach().cpu().item()),
                'deep_inst_avg': float(deep_inst_avg.detach().cpu().item()),
                'final_pyramid_loss': float(final_pyramid_loss.detach().cpu().item()),
                'total_loss': float(total_loss.detach().cpu().item()),
            }
        )
        return total_loss, layer_stats

    def crops_and_extract(self, x, scaling_rate):
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2, high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        batch_index = torch.arange(x.size(0), device=x.device)[:, None]

        if np.random.rand() < 0.5:
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

            overlap_start = torch.as_tensor(crop_offset + crop_left, device=x.device)[:, None]
            overlap_index = overlap_start + torch.arange(crop_l, device=x.device)[None, :]
            view1 = x[batch_index, overlap_index]

            light_sigma = max(0.05, scaling_rate * 0.25)
            if np.random.rand() < 0.5:
                context_len = crop_right - crop_eleft
                context_start = torch.as_tensor(crop_offset + crop_eleft, device=x.device)[:, None]
                context_index = context_start + torch.arange(context_len, device=x.device)[None, :]
                view2 = x[batch_index, context_index]
                view2 = DataTransform(view2, light_sigma).to(self.device).float()
                view2 = view2[:, -crop_l:]
            else:
                context_len = crop_eright - crop_left
                context_start = torch.as_tensor(crop_offset + crop_left, device=x.device)[:, None]
                context_index = context_start + torch.arange(context_len, device=x.device)[None, :]
                view2 = x[batch_index, context_index]
                view2 = DataTransform(view2, light_sigma).to(self.device).float()
                view2 = view2[:, :crop_l]
        else:
            crop_offset = np.random.randint(low=-crop_left, high=ts_l - crop_right + 1, size=x.size(0))
            overlap_start = torch.as_tensor(crop_offset + crop_left, device=x.device)[:, None]
            overlap_index = overlap_start + torch.arange(crop_l, device=x.device)[None, :]
            view1 = x[batch_index, overlap_index]
            view2 = DataTransform(view1, scaling_rate).to(self.device).float()

        return view1.float(), view2.float()

    def evaluate_clustering(self):
        metrics = self._evaluate_final_layer(self.eval_loader, self.net)
        print("-------FK1_Evaluate---------")
        print(f"[Final-layer pooled] acc={metrics['ACC']}  nmi={metrics['NMI']}")
        return metrics

