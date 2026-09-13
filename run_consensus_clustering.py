"""RF-LVC stage 3 entry point: candidate scales, local views and consensus.

Freezes the encoder trained by ``train.py`` (or trains one if no checkpoint is
given), builds every local temporal pattern defined by a candidate encoder
level and a candidate window length, clusters each pattern independently into
a local view, and fuses the views through a co-association matrix followed by
spectral clustering.

Example
-------
python run_consensus_clustering.py \
    --dataset_dir ./datasets/UCRArchive_2018_csv \
    --dataset_name ECGFiveDays \
    --results_dir ./results_consensus \
    --encoder_ckpt ./results_encoder/encoder_ECGFiveDays_....pt

The scale settings are optional.  By default they follow the rule of the
method -- the deepest ``K`` eligible encoder levels and ``M`` windows over
``[max(5, ceil(0.05 T)), min(0.65 T, 900)]`` -- so a plain invocation needs no
external file (see ``resolve_scale_settings``).  Single fields can be
overridden, and the per-dataset settings behind the tables of the paper can be
replayed by pointing ``--scales_csv`` at the table that records them:

python ... --rec_layers 5,6 --rec_min_length 32 --rec_rate_max_length 0.6
python ... --scales_csv <table>    # replay recorded per-dataset settings
python ... --scales rule                                  # ignore a passed table
"""

import argparse
import datetime
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch

import datautils
from module3_contrastive_training.fk1_model import FK1Model
from module4_scale_consensus import (
    build_eval_loader,
    build_segments,
    cluster_baseline,
    cluster_features,
    compute_metrics,
    consensus_cluster,
    ensure_dir,
    extract_global_embeddings,
    extract_layer_feature_maps,
    load_encoder_checkpoint,
    save_json,
)


def prepare_combined_data(data_path, data_name):
    train_data, train_labels, train_index = datautils.load_csv_data(data_path, data_name, split='TRAIN')
    test_data, test_labels, test_index = datautils.load_csv_data(data_path, data_name, split='TEST')

    max_train_index = np.max(train_index)
    adjusted_test_index = test_index + max_train_index + 1

    combined_data = np.concatenate((train_data, test_data), axis=0)
    combined_labels = np.concatenate((train_labels, test_labels), axis=0)
    combined_index = np.concatenate((train_index, adjusted_test_index), axis=0)
    return combined_data, combined_labels, combined_index


def _auto_window_lengths(seq_len, min_length, rate_max_length, n_lengths):
    max_length = max(min_length, int(seq_len * rate_max_length))
    lengths = {
        int(value)
        for value in np.linspace(min_length, max_length, n_lengths)
    }
    return sorted(length for length in lengths if length > 0)


# --------------------------------------------------------------------------
# Candidate-scale defaults -- these ARE the defaults of the released code, so a
# plain invocation runs the published method without any external file.
#
# Paper, "Candidate encoder levels": the candidate levels are the DEEPEST K
# levels among those whose receptive field stays below rho_max * T (see
# _pre_global_layers below).  Paper, "Candidate window lengths": the window set
# contains M windows spread over [w_min, r_max * T].
#
# K and M are the two cost knobs of the pipeline: the number of local views is
# K * M and every view is more expensive on longer sequences.  We therefore
# keep K at the smallest value that still compares two receptive-field scales
# (K = 2, or K = 3 for small datasets, where the extra scale is cheap) and use
# MIN_WINDOW_COUNT windows as the floor.  A larger M refines the coverage of
# temporal scales and further stabilises the consensus, at a proportionally
# higher cost, so it should only be raised when the budget allows.
# --------------------------------------------------------------------------
MIN_WINDOW_COUNT = 5


def default_window_range(seq_len):
    """Window range ``[w_min, max_window]`` of the workload-aware default.

    ``w_min`` has an absolute floor of 5 samples and grows with the sequence
    length (5% of ``T``), so short series still aggregate over a meaningful
    span while long series spend their resolution on informative segments.
    The largest window is bounded both relatively (0.65 of ``T``) and by an
    absolute ceiling of 900 samples: a window covering most of a long
    sequence would approach global aggregation, which is exactly what the
    local-view construction avoids.  Note that the ceiling does *not* reduce
    the runtime -- large windows produce the fewest segments -- it only
    limits the aggregation span; the cost is dominated by ``w_min``.
    """
    w_min = max(5, int(math.ceil(0.05 * seq_len)))
    max_window = min(int(round(0.65 * seq_len)), 900)
    if max_window <= w_min:
        max_window = w_min + 1
    return w_min, max_window


def default_scale_budget(n_series):
    """Default number of candidate levels ``K`` and windows ``M``.

    ``K`` counts the deepest eligible levels (see the module comment above):
    ``K = 3`` for small datasets, where one more scale is inexpensive, and
    ``K = 2`` otherwise.  ``M`` is kept at ``MIN_WINDOW_COUNT`` by default and
    should be increased only when the computational budget allows, since the
    number of local views is ``K * M``.
    """
    return (3 if n_series <= 300 else 2), MIN_WINDOW_COUNT


def _csv_field(record, key, cast):
    """Read one optional field of a CSV record, mapping blanks to ``None``."""
    value = record.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    return cast(value)


def load_recommended_scales(csv_path):
    """Load a per-dataset scale table from ``csv_path`` as a dict.

    No such table is distributed with this code: it is the record of the
    per-dataset settings used for the experiment tables of the paper, and it is
    read only when the caller asks for it with ``--scales_csv``.  Every entry
    carries the explicit encoder levels (``rec_layers``, ``|`` separated), the
    smallest
    window (``min_length``), the largest window as a fraction of the sequence
    length (``rate_max_length``) and the number of windows (``n_lengths``).
    A missing or unreadable file yields an empty mapping; the caller decides
    whether that is an error (it is, when the path was given explicitly).
    """
    path = csv_path
    if not os.path.isfile(path):
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # an unreadable table must not break a run
        print('[scales] cannot read %s (%s)' % (path, exc))
        return {}

    table = {}
    for record in frame.to_dict('records'):
        name = _csv_field(record, 'dataset_name', str)
        if not name or not name.strip():
            continue
        table[name.strip()] = {
            'rec_layers': (_csv_field(record, 'rec_layers', str) or '').strip(),
            'rec_min_length': _csv_field(record, 'min_length', int),
            'rec_rate_max_length': _csv_field(record, 'rate_max_length', float),
            'n_lengths': _csv_field(record, 'n_lengths', int),
        }
    return table


def resolve_scale_settings(
    dataset_name,
    seq_len,
    n_series,
    rec_layers=None,
    rec_min_length=None,
    rec_rate_max_length=None,
    min_length=None,
    rate_max_length=None,
    n_lengths=None,
    auto_layer_count=None,
    auto_max_rf_ratio=1.2,
    scales='auto',
    csv_path=None,
):
    """Resolve the candidate scales from the CLI, an optional table and the rule.

    The arguments of the run are all optional.  The encoder levels and the
    window range are resolved independently, each by the first source that
    provides a value:

    1. an explicit command-line value (``--rec_layers`` for the levels;
       ``--rec_min_length`` / ``--rec_rate_max_length`` / ``--min_length`` /
       ``--rate_max_length`` / ``--n_lengths`` for the windows);
    2. the per-dataset table given by ``--scales_csv``, which is how the
       settings behind the experiment tables of the paper are replayed;
    3. the default rule of the method, which is what a plain invocation uses --
       the deepest ``K`` eligible levels (``K = 3`` for at most 300 series and
       2 otherwise) and ``M = MIN_WINDOW_COUNT`` windows over
       ``[max(5, ceil(0.05 T)), min(0.65 T, 900)]``.

    The default path therefore depends on no external file at all.  ``scales``
    is ``'auto'`` (consult a table only when one was passed) or ``'rule'``
    (ignore any table).  A dataset that is absent from a readable table is not
    an error -- the rule is used and reported -- but an unreadable table that
    was requested explicitly is, since quietly running another configuration
    would misreport the run.

    Returns a dict with the resolved ``rec_layers``, ``min_length``,
    ``rate_max_length``, ``n_lengths`` and ``auto_layer_count``, the labels
    ``level_source`` / ``window_source`` (``'explicit'``, ``'table'`` or
    ``'rule'``) and the overall ``source`` used for logging.
    """
    if scales not in ('auto', 'rule'):
        raise ValueError("scales must be 'auto' or 'rule'")

    table_entry = None
    table_note = None
    table_used = None
    if scales == 'rule':
        if csv_path:
            table_note = '--scales rule: ignoring the per-dataset table %s' % csv_path
    elif csv_path:
        table = load_recommended_scales(csv_path)
        if not table:
            # An explicitly requested table must not silently fall back: that
            # would change the configuration without changing the command.
            raise ValueError(
                'no usable per-dataset table found at %s; drop --scales_csv or '
                'pass a valid table' % csv_path
            )
        table_used = os.path.relpath(csv_path, os.path.dirname(os.path.abspath(__file__)))
        table_entry = table.get(dataset_name)
        if table_entry is None:
            available = ', '.join(sorted(table)[:5])
            table_note = (
                'dataset %r is not in %s (e.g. %s, ...); using the default rule'
                % (dataset_name, table_used, available)
            )

    rule_w_min, rule_max_window = default_window_range(seq_len)
    rule_layer_count, rule_n_lengths = default_scale_budget(n_series)

    # --- encoder levels ---------------------------------------------------
    if rec_layers is not None and str(rec_layers).strip():
        resolved_layers, level_source = str(rec_layers).strip(), 'explicit'
    elif table_entry is not None and table_entry['rec_layers']:
        resolved_layers, level_source = table_entry['rec_layers'], 'table'
    else:
        resolved_layers, level_source = None, 'rule'
    resolved_layer_count = (
        int(auto_layer_count) if auto_layer_count is not None else rule_layer_count
    )

    # --- window range -----------------------------------------------------
    explicit_min = rec_min_length if rec_min_length is not None else min_length
    if explicit_min is not None:
        resolved_min, min_source = int(explicit_min), 'explicit'
    elif table_entry is not None and table_entry['rec_min_length'] is not None:
        resolved_min, min_source = table_entry['rec_min_length'], 'table'
    else:
        resolved_min, min_source = rule_w_min, 'rule'

    explicit_rate = rec_rate_max_length if rec_rate_max_length is not None else rate_max_length
    if explicit_rate is not None:
        resolved_rate, rate_source = float(explicit_rate), 'explicit'
    elif table_entry is not None and table_entry['rec_rate_max_length'] is not None:
        resolved_rate, rate_source = table_entry['rec_rate_max_length'], 'table'
    else:
        resolved_rate, rate_source = (rule_max_window + 1e-6) / float(seq_len), 'rule'

    if n_lengths is not None:
        resolved_n, count_source = int(n_lengths), 'explicit'
    elif table_entry is not None and table_entry['n_lengths'] is not None:
        resolved_n, count_source = table_entry['n_lengths'], 'table'
    else:
        resolved_n, count_source = rule_n_lengths, 'rule'

    window_source = min_source if min_source == rate_source == count_source else 'mixed'
    if level_source == window_source:
        source = level_source
    elif 'explicit' in (level_source, window_source):
        source = 'explicit + %s' % (window_source if level_source == 'explicit' else level_source)
    else:
        source = 'mixed'

    return {
        'source': source,
        'level_source': level_source,
        'window_source': window_source,
        'table': table_used,
        'table_entry_used': table_entry is not None,
        'note': table_note,
        'rec_layers': resolved_layers,
        'rec_min_length': resolved_min,
        'rec_rate_max_length': resolved_rate,
        'n_lengths': resolved_n,
        'auto_layer_count': resolved_layer_count,
        'auto_max_rf_ratio': float(auto_max_rf_ratio),
        'rule_w_min': rule_w_min,
        'rule_max_window': rule_max_window,
        'rule_layer_count': rule_layer_count,
    }


def _pre_global_layers(receptive_fields, seq_len, layer_count, max_rf_ratio):
    eligible = [
        idx for idx, receptive_field in enumerate(receptive_fields)
        if receptive_field <= seq_len * max_rf_ratio
    ]
    if not eligible:
        return [0]
    deepest = eligible[-1]
    start = max(0, deepest - layer_count + 1)
    return list(range(start, deepest + 1))


def _recommended_layers(rec_layers, receptive_fields):
    """Parse an explicit level list such as ``'5,6'`` or ``'5|6'``.

    Both separators are accepted because the recorded per-dataset tables use
    ``|`` and the command line uses ``,``.
    """
    if rec_layers is None or not str(rec_layers).strip():
        return None
    layers = []
    for value in re.split(r'[,\s|]+', str(rec_layers).strip()):
        if not value:
            continue
        layer_id = int(value)
        if layer_id < 0 or layer_id >= len(receptive_fields):
            raise ValueError(
                'rec_layers contains an invalid level %d; valid range is 0..%d'
                % (layer_id, len(receptive_fields) - 1)
            )
        layers.append(layer_id)
    return sorted(set(layers))


def resolve_view_specs(
    receptive_fields,
    seq_len,
    min_length,
    rate_max_length,
    n_lengths,
    auto_layer_count,
    auto_max_rf_ratio,
    rec_layers=None,
    rec_min_length=None,
    rec_rate_max_length=None,
):
    """Enumerate the (encoder level, window length) candidates.

    Candidate scales can be chosen in two label-free ways:

    * automatic -- the deepest ``auto_layer_count`` levels whose receptive field
      does not exceed ``auto_max_rf_ratio * seq_len``, with windows spaced over
      ``[min_length, rate_max_length * seq_len]``;
    * explicit -- ``rec_layers`` (e.g. ``5,6``) together with
      ``rec_min_length`` / ``rec_rate_max_length``.

    The recorded per-dataset settings of the paper are of the explicit form and
    are filled in by ``resolve_scale_settings`` when ``--scales_csv`` points at
    the table that holds them.
    """
    layers = _recommended_layers(rec_layers, receptive_fields)
    if layers is None:
        layers = _pre_global_layers(receptive_fields, seq_len, auto_layer_count, auto_max_rf_ratio)
        length_min, length_rate = min_length, rate_max_length
    else:
        length_min = rec_min_length if rec_min_length is not None else min_length
        length_rate = rec_rate_max_length if rec_rate_max_length is not None else rate_max_length
    lengths = _auto_window_lengths(seq_len, length_min, length_rate, n_lengths)

    specs = []
    for layer_id in layers:
        receptive_field = int(receptive_fields[layer_id])
        for window_size in lengths:
            specs.append({
                'name': f'rf_l{layer_id}_w{int(window_size)}',
                'layer_id': layer_id,
                'window_size': int(window_size),
                'receptive_field': receptive_field,
                'effective_span': int(receptive_field + int(window_size) - 1),
            })
    return specs


def save_npz(path, **kwargs):
    np.savez_compressed(path, **kwargs)


def build_sample_embedding_features(segments, num_samples):
    segment_embeddings = segments['segment_embedding']
    feature_dim = segment_embeddings.shape[1]
    features = np.zeros((num_samples, feature_dim * 2), dtype=np.float32)

    for sample_id in range(num_samples):
        member_idx = np.where(segments['sample_id'] == sample_id)[0]
        if member_idx.size == 0:
            continue
        sample_embeddings = segment_embeddings[member_idx]
        mean_pool = sample_embeddings.mean(axis=0)
        max_pool = sample_embeddings.max(axis=0)
        features[sample_id] = np.concatenate([mean_pool, max_pool], axis=0).astype(np.float32)

    columns = (
        [f'emb_mean_{idx}' for idx in range(feature_dim)]
        + [f'emb_max_{idx}' for idx in range(feature_dim)]
    )
    feature_df = pd.DataFrame(features, columns=columns)
    feature_df.insert(0, 'sample_id', np.arange(num_samples))
    return {
        'features': features,
        'dataframe': feature_df,
    }


def build_local_view(view_spec, feature_map, labels, n_clusters, args, output_dir):
    """One local temporal pattern: segments -> sample features -> one local view."""
    view_dir = ensure_dir(os.path.join(output_dir, view_spec['name']))
    stride = args.segment_stride if args.segment_stride > 0 else None

    segments = build_segments(
        feature_map=feature_map,
        labels=labels,
        layer_id=view_spec['layer_id'],
        window_size=view_spec['window_size'],
        stride=stride,
    )

    sample_features = build_sample_embedding_features(segments, num_samples=labels.shape[0])
    sample_features['dataframe'].to_csv(
        os.path.join(view_dir, 'sample_embedding_features.csv'), index=False
    )

    view_labels, view_features = cluster_features(sample_features['features'], n_clusters=n_clusters)
    metrics = compute_metrics(labels, view_labels, n_clusters)
    summary = {
        'view_name': view_spec['name'],
        'layer_id': int(view_spec['layer_id']),
        'window_size': int(view_spec['window_size']),
        'receptive_field': int(view_spec.get('receptive_field', -1)),
        'effective_span': int(view_spec.get('effective_span', -1)),
        'feature_dims': int(view_features.shape[1]),
        **metrics,
    }
    save_json(summary, os.path.join(view_dir, 'view_summary.json'))
    save_npz(
        os.path.join(view_dir, 'local_view.npz'),
        sample_features=view_features,
        view_labels=view_labels,
    )
    return view_labels, summary


def main():
    parser = argparse.ArgumentParser(
        description='RF-LVC stage 3: multi-scale local views and consensus clustering.'
    )
    parser.add_argument('--dataset_dir', default='./datasets/UCRArchive_2018_csv/', type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--results_dir', default='./results_consensus/', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1127)
    parser.add_argument('--device', type=str, default='cuda',
                        help="Torch device, e.g. 'cuda' or 'cpu'")

    # Encoder training; only used when --encoder_ckpt is not provided.
    parser.add_argument('--repr_dims', type=int, default=64)
    parser.add_argument('--hidden_dims', type=int, default=64)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--pretraining_epoch', type=int, default=100)
    parser.add_argument('--hard_w', type=float, default=0.2)
    parser.add_argument('--instance_activation_ratio', type=float, default=0.8)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--encoder_ckpt', type=str, default='')

    # Label-free candidate temporal scales.  All of these are optional and
    # default to the rule of the method, so a plain invocation needs no scale
    # argument; see resolve_scale_settings.
    parser.add_argument('--min_length', type=int, default=None,
                        help='Smallest candidate window length; default: max(5, ceil(0.05*T))')
    parser.add_argument('--rate_max_length', type=float, default=None,
                        help='Largest candidate window length as a fraction of the sequence '
                             'length; default: min(0.65*T, 900)/T')
    parser.add_argument('--n_lengths', type=int, default=None,
                        help='Number of candidate window lengths (linearly spaced); default: 5')
    parser.add_argument('--auto_layer_count', type=int, default=None,
                        help='Number of encoder levels used as candidates; default: 3 for at '
                             'most 300 series and 2 otherwise')
    parser.add_argument('--auto_max_rf_ratio', type=float, default=1.2,
                        help='A level is eligible if its receptive field <= this ratio * sequence length')
    parser.add_argument('--rec_layers', type=str, default=None,
                        help="Explicit encoder levels such as '5,6' (or '5|6'), overriding the "
                             'rule')
    parser.add_argument('--rec_min_length', type=int, default=None,
                        help='Smallest window length of an explicit scale setting; overrides '
                             '--min_length')
    parser.add_argument('--rec_rate_max_length', type=float, default=None,
                        help='Largest window length of an explicit setting, as a fraction of '
                             'the sequence length; overrides --rate_max_length')
    parser.add_argument('--scales_csv', type=str, default=None,
                        help='Optional CSV table of per-dataset scale settings to replay, for '
                             'reproducing recorded experiment configurations. It is not needed '
                             'by the method and is never read unless it is passed here')
    parser.add_argument('--scales', choices=('auto', 'rule'), default='auto',
                        help='"auto" (default) replays a per-dataset setting when --scales_csv '
                             'lists the dataset, and otherwise uses the rule of the paper; '
                             '"rule" ignores --scales_csv')
    parser.add_argument('--segment_stride', type=int, default=0,
                        help='Stride of the sliding window; 0 means window_size // 2')
    parser.add_argument('--workload_scales', action='store_true',
                        help='Deprecated alias of --scales rule: use the label-free rule even '
                             'when a per-dataset table is passed')
    args = parser.parse_args()

    if args.workload_scales:
        if args.scales != 'auto':
            parser.error('--workload_scales conflicts with --scales %s' % args.scales)
        args.scales = 'rule'

    datautils.set_seed(args.seed)
    data_path = os.path.join(args.dataset_dir, args.dataset_name)
    combined_data, combined_labels, combined_index = prepare_combined_data(data_path, args.dataset_name)

    train_loader = datautils.create_data_loader(
        combined_data, combined_labels, combined_index, args.batch_size, shuffle=True
    )
    eval_loader = build_eval_loader(combined_data, combined_labels, combined_index, args.batch_size)

    n_cluster = len(np.unique(combined_labels))

    seq_len = combined_data.shape[1]
    n_series = combined_data.shape[0]
    scale_settings = resolve_scale_settings(
        dataset_name=args.dataset_name,
        seq_len=seq_len,
        n_series=n_series,
        rec_layers=args.rec_layers,
        rec_min_length=args.rec_min_length,
        rec_rate_max_length=args.rec_rate_max_length,
        min_length=args.min_length,
        rate_max_length=args.rate_max_length,
        n_lengths=args.n_lengths,
        auto_layer_count=args.auto_layer_count,
        auto_max_rf_ratio=args.auto_max_rf_ratio,
        scales=args.scales,
        csv_path=args.scales_csv,
    )
    if scale_settings['note']:
        print('[scales] %s' % scale_settings['note'])

    # Resolved values feed both the explicit and the automatic path below, so
    # the run is fully described by what is reported here.
    args.rec_layers = scale_settings['rec_layers']
    args.rec_min_length = scale_settings['rec_min_length']
    args.rec_rate_max_length = scale_settings['rec_rate_max_length']
    args.min_length = scale_settings['rec_min_length']
    args.rate_max_length = scale_settings['rec_rate_max_length']
    args.n_lengths = scale_settings['n_lengths']
    args.auto_layer_count = scale_settings['auto_layer_count']
    print(
        '[scales] source=%s (levels=%s, windows=%s) T=%d N=%d layers=%s '
        'w_min=%d w_max=%d M=%d'
        % (
            scale_settings['source'],
            scale_settings['level_source'],
            scale_settings['window_source'],
            seq_len,
            n_series,
            args.rec_layers if args.rec_layers else 'rule (K=%d)' % args.auto_layer_count,
            args.min_length,
            int(seq_len * args.rate_max_length),
            args.n_lengths,
        )
    )

    config = {
        'batch_size': args.batch_size,
        'output_dims': args.repr_dims,
        'hidden_dims': args.hidden_dims,
        'depth': args.depth,
        'lr': args.lr,
        'pretraining_epoch': args.pretraining_epoch,
        'hard_w': args.hard_w,
        'instance_activation_ratio': args.instance_activation_ratio,
        'device': args.device,
        'log_dir': args.log_dir,
        'dataset_name': args.dataset_name,
        'dataset_size': combined_data.shape[0],
        'timesteps_len': combined_data.shape[1],
        'input_dims': combined_data.shape[2],
        'n_cluster': n_cluster,
    }
    print(f'config: {config}')

    model = FK1Model(train_loader, eval_loader=eval_loader, **config)
    run_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = ensure_dir(os.path.join(args.results_dir, args.dataset_name, run_tag))

    t0 = time.time()
    if args.encoder_ckpt:
        model = load_encoder_checkpoint(model, args.encoder_ckpt, map_location=args.device)
        print(f'Loaded encoder checkpoint: {args.encoder_ckpt}')
    elif args.pretraining_epoch > 0:
        model.Pretraining()
        encoder_ckpt = os.path.join(output_dir, f'{args.dataset_name}_encoder.pt')
        torch.save(model.get_inference_state_dict(), encoder_ckpt)
        print(f'Saved trained inference encoder to: {encoder_ckpt}')
    else:
        raise ValueError('Provide --encoder_ckpt or set --pretraining_epoch > 0.')
    pretrain_time = time.time() - t0

    global_state = extract_global_embeddings(model, eval_loader)
    baseline_labels = cluster_baseline(global_state['embeddings'], n_cluster)
    baseline_metrics = compute_metrics(global_state['labels'], baseline_labels, n_cluster)
    save_npz(
        os.path.join(output_dir, 'baseline_embeddings.npz'),
        embeddings=global_state['embeddings'],
        labels=global_state['labels'],
        baseline_labels=baseline_labels,
    )
    save_json(baseline_metrics, os.path.join(output_dir, 'baseline_metrics.json'))

    backbone = model.net.module if hasattr(model.net, 'module') else model.net
    receptive_fields = getattr(backbone, 'layer_receptive_fields')
    view_specs = resolve_view_specs(
        receptive_fields,
        seq_len=combined_data.shape[1],
        min_length=args.min_length,
        rate_max_length=args.rate_max_length,
        n_lengths=args.n_lengths,
        auto_layer_count=args.auto_layer_count,
        auto_max_rf_ratio=args.auto_max_rf_ratio,
        rec_layers=args.rec_layers,
        rec_min_length=args.rec_min_length,
        rec_rate_max_length=args.rec_rate_max_length,
    )
    save_json(
        {
            'receptive_fields': receptive_fields,
            'candidate_config': {
                'min_length': args.min_length,
                'rate_max_length': args.rate_max_length,
                'n_lengths': args.n_lengths,
                'auto_layer_count': args.auto_layer_count,
                'auto_max_rf_ratio': args.auto_max_rf_ratio,
            },
            'explicit_scale_config': {
                'rec_layers': args.rec_layers,
                'rec_min_length': args.rec_min_length,
                'rec_rate_max_length': args.rec_rate_max_length,
                'used': args.rec_layers is not None,
            },
            'scale_settings': {
                'source': scale_settings['source'],
                'level_source': scale_settings['level_source'],
                'window_source': scale_settings['window_source'],
                'requested_scales': args.scales,
                'table': scale_settings['table'],
                'table_entry_used': scale_settings['table_entry_used'],
                'note': scale_settings['note'],
                'rule': {
                    'w_min': scale_settings['rule_w_min'],
                    'max_window': scale_settings['rule_max_window'],
                    'K': scale_settings['rule_layer_count'],
                    'M': MIN_WINDOW_COUNT,
                },
            },
            'views': view_specs,
        },
        os.path.join(output_dir, 'analysis_config.json'),
    )
    print(
        f"[Candidate scales] layers={sorted({spec['layer_id'] for spec in view_specs})} "
        f"windows={sorted({spec['window_size'] for spec in view_specs})} "
        f"views={len(view_specs)} "
        f"source={scale_settings['level_source']}/{scale_settings['window_source']}"
    )

    unique_layers = sorted({spec['layer_id'] for spec in view_specs})
    feature_state = extract_layer_feature_maps(model, eval_loader, unique_layers)

    all_view_labels = []
    view_summaries = []
    for view_spec in view_specs:
        view_labels, summary = build_local_view(
            view_spec=view_spec,
            feature_map=feature_state['feature_maps'][view_spec['layer_id']],
            labels=global_state['labels'],
            n_clusters=n_cluster,
            args=args,
            output_dir=output_dir,
        )
        all_view_labels.append(view_labels)
        view_summaries.append(summary)

    consensus_labels, consensus_matrix = consensus_cluster(all_view_labels, n_cluster)
    consensus_metrics = compute_metrics(global_state['labels'], consensus_labels, n_cluster)

    save_npz(
        os.path.join(output_dir, 'consensus_results.npz'),
        consensus_labels=consensus_labels,
        consensus_matrix=consensus_matrix,
    )
    save_json(consensus_metrics, os.path.join(output_dir, 'consensus_metrics.json'))

    comparison_rows = [{'view': 'baseline', **baseline_metrics}]
    comparison_rows.extend(view_summaries)
    comparison_rows.append({'view': 'consensus', **consensus_metrics})
    pd.DataFrame(comparison_rows).to_csv(os.path.join(output_dir, 'comparison.csv'), index=False)
    save_json(
        {
            'pretrain_or_load_time_sec': round(pretrain_time, 6),
            'total_runtime_sec': round(time.time() - t0, 6),
        },
        os.path.join(output_dir, 'timing.json'),
    )

    print('-------RF-LVC local views and consensus---------')
    print(
        f"[Baseline] ACC={baseline_metrics['ACC']:.6f} "
        f"NMI={baseline_metrics['NMI']:.6f} "
        f"ARI={baseline_metrics['ARI']:.6f} "
        f"RI={baseline_metrics['RI']:.6f}"
    )
    for summary in view_summaries:
        print(
            f"[View {summary['view_name']}] "
            f"ACC={summary['ACC']:.6f} NMI={summary['NMI']:.6f} "
            f"ARI={summary['ARI']:.6f} RI={summary['RI']:.6f} "
            f"dims={summary['feature_dims']}"
        )
    print(
        f"[Consensus] ACC={consensus_metrics['ACC']:.6f} "
        f"NMI={consensus_metrics['NMI']:.6f} "
        f"ARI={consensus_metrics['ARI']:.6f} "
        f"RI={consensus_metrics['RI']:.6f}"
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == '__main__':
    main()

