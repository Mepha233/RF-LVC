"""Module 4 - candidate temporal scales, local views and consensus clustering."""

from .evaluate import compute_metrics, save_json
from .feature_extraction import (
    build_eval_loader,
    ensure_dir,
    extract_global_embeddings,
    extract_layer_feature_maps,
    load_encoder_checkpoint,
)
from .segments import build_segments
from .view_clustering import (
    build_consensus_matrix,
    cluster_baseline,
    cluster_features,
    consensus_cluster,
)
