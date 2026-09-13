"""Clustering metrics used for local views and for the consensus partition."""

import json

from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import fowlkes_mallows_score as fmi

from module2_encoder.Metrics import acc, nmi, rand_index_score


def compute_metrics(label_true, label_pred, n_clusters):
    return {
        'ACC': float(acc(label_true, label_pred, n_clusters)),
        'NMI': float(nmi(label_true, label_pred)),
        'ARI': float(ari(label_true, label_pred)),
        'RI': float(rand_index_score(label_pred, label_true)),
        'FMI': float(fmi(label_true, label_pred)),
    }


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
