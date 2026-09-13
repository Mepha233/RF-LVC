"""View-specific clustering and co-association consensus clustering."""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler


def cluster_baseline(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.astype(np.int64)


def cluster_features(features, n_clusters):
    if features.size == 0 or features.shape[1] == 0:
        labels = np.zeros(features.shape[0], dtype=np.int64)
        return labels, features.astype(np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(scaled)
    return labels.astype(np.int64), scaled.astype(np.float32)


def build_consensus_matrix(view_labels):
    if len(view_labels) == 1:
        labels = view_labels[0]
        return (labels[:, None] == labels[None, :]).astype(np.float32)

    num_samples = view_labels[0].shape[0]
    consensus = np.zeros((num_samples, num_samples), dtype=np.float32)
    for labels in view_labels:
        consensus += (labels[:, None] == labels[None, :]).astype(np.float32)
    consensus /= float(len(view_labels))
    np.fill_diagonal(consensus, 1.0)
    return consensus


def consensus_cluster(view_labels, n_clusters):
    if len(view_labels) == 1:
        labels = view_labels[0].astype(np.int64)
        return labels, build_consensus_matrix(view_labels)

    consensus = build_consensus_matrix(view_labels)
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=0,
        assign_labels='kmeans',
    ).fit_predict(consensus)
    return labels.astype(np.int64), consensus

