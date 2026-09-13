"""Local temporal pattern construction: segment a level feature map."""

import numpy as np


def _window_starts(length, window_size, stride):
    if window_size >= length:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last_start = length - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_segments(feature_map, labels, layer_id, window_size, stride=None):
    n_samples, time_steps, _ = feature_map.shape
    window_size = min(window_size, time_steps)
    stride = stride or max(1, window_size // 2)

    segment_ids = []
    sample_ids = []
    sample_labels = []
    starts = []
    ends = []
    raw_segments = []
    segment_embeddings = []

    segment_counter = 0
    for sample_id in range(n_samples):
        for start in _window_starts(time_steps, window_size, stride):
            end = start + window_size
            segment = feature_map[sample_id, start:end, :]
            mean_pool = segment.mean(axis=0)
            max_pool = segment.max(axis=0)
            segment_ids.append(segment_counter)
            sample_ids.append(sample_id)
            sample_labels.append(labels[sample_id])
            starts.append(start)
            ends.append(end)
            raw_segments.append(segment.astype(np.float32))
            segment_embeddings.append(np.concatenate([mean_pool, max_pool], axis=0).astype(np.float32))
            segment_counter += 1

    return {
        'segment_id': np.asarray(segment_ids, dtype=np.int64),
        'sample_id': np.asarray(sample_ids, dtype=np.int64),
        'sample_label': np.asarray(sample_labels, dtype=np.int64),
        'layer_id': np.full(segment_counter, layer_id, dtype=np.int64),
        'window_size': np.full(segment_counter, window_size, dtype=np.int64),
        'start': np.asarray(starts, dtype=np.int64),
        'end': np.asarray(ends, dtype=np.int64),
        'segment_embedding': np.asarray(segment_embeddings, dtype=np.float32),
        'raw_segment': np.asarray(raw_segments, dtype=np.float32),
    }
