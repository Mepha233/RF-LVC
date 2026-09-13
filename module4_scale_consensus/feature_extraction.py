"""Frozen-encoder feature extraction (global embeddings and level maps)."""

import os

import numpy as np
import torch

import datautils


def build_eval_loader(data, labels, indices, batch_size):
    return datautils.create_data_loader(data, labels, indices, batch_size, shuffle=False)


def _unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def _normalize_state_dict_keys(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith('module.') for key in state_dict.keys()):
        return {key[len('module.'):]: value for key, value in state_dict.items()}
    return state_dict


def load_encoder_checkpoint(model, checkpoint_path, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = _normalize_state_dict_keys(state_dict)
    elif hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    model.encoder.load_state_dict(state_dict, strict=True)
    model.net = torch.optim.swa_utils.AveragedModel(model.encoder)
    model.net.update_parameters(model.encoder)
    return model


def extract_global_embeddings(model, data_loader):
    backbone = _unwrap_model(model.net)
    previous_mode = backbone.training
    backbone.eval()

    embeddings = np.zeros((model.dataset_size, model.layer_output_dims[-1]), dtype=np.float32)
    labels = np.zeros(model.dataset_size, dtype=np.int64)
    sample_ids = np.zeros(model.dataset_size, dtype=np.int64)

    with torch.no_grad():
        for x, target, index in data_loader:
            x = x.to(model.device)
            final_output = backbone(x, return_all_layers=True)[-1]
            pooled = torch.max(final_output, dim=1).values

            idx_np = index.numpy()
            embeddings[idx_np] = pooled.detach().cpu().numpy()
            labels[idx_np] = target.numpy()
            sample_ids[idx_np] = idx_np

    backbone.train(previous_mode)
    return {
        'embeddings': embeddings,
        'labels': labels,
        'sample_ids': sample_ids,
    }


def extract_layer_feature_maps(model, data_loader, layer_ids):
    backbone = _unwrap_model(model.net)
    previous_mode = backbone.training
    backbone.eval()

    layer_ids = sorted(set(layer_ids))
    label_true = np.zeros(model.dataset_size, dtype=np.int64)
    feature_maps = {layer_id: None for layer_id in layer_ids}

    with torch.no_grad():
        for x, target, index in data_loader:
            x = x.to(model.device)
            all_outputs = backbone(x, return_features=True)
            idx_np = index.numpy()

            for layer_id in layer_ids:
                current = all_outputs[layer_id].detach().cpu().numpy()
                if feature_maps[layer_id] is None:
                    feature_maps[layer_id] = np.zeros(
                        (model.dataset_size, current.shape[1], current.shape[2]),
                        dtype=np.float32,
                    )
                feature_maps[layer_id][idx_np] = current

            label_true[idx_np] = target.numpy()

    backbone.train(previous_mode)
    return {
        'feature_maps': feature_maps,
        'labels': label_true,
    }


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

