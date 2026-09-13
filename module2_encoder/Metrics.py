import numpy as np
from scipy.special import comb
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score


def _relabel_to_contiguous(values):
    values = np.asarray(values)
    uniques = np.unique(values)
    mapping = {label: idx for idx, label in enumerate(uniques)}
    return np.vectorize(mapping.get)(values)


def rand_index_score(clusters, classes):
    clusters = _relabel_to_contiguous(clusters).astype(int)
    classes = _relabel_to_contiguous(classes).astype(int)
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    assignments = np.c_[clusters, classes]
    tp = sum(comb(np.bincount(assignments[assignments[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(assignments), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def acc(y_true, y_pred, num_cluster):
    y_true = _relabel_to_contiguous(y_true).astype(np.int64)
    y_pred = _relabel_to_contiguous(y_pred).astype(np.int64)
    assert y_pred.size == y_true.size

    size = max(num_cluster, int(max(y_pred.max(), y_true.max())) + 1)
    contingency = np.zeros((size, size))
    for i in range(y_pred.size):
        contingency[y_pred[i], y_true[i]] += 1

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    matched = 0.0
    for row, col in zip(row_ind, col_ind):
        matched += contingency[row, col]
    return matched / y_pred.size
