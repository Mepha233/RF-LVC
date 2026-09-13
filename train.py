"""RF-LVC stage 1-2 entry point: encoder pretraining.

Trains the multi-level dilated-convolution encoder with the layer-wise
contrastive objective (module 3) on the concatenation of the TRAIN and TEST
splits of one UCR dataset, then reports the final-layer pooled K-means
baseline and saves the inference (parameter-averaged) encoder state dict that
stage 3 consumes.

Example
-------
python train.py --dataset_dir ./datasets/UCRArchive_2018_csv \
    --dataset_name ECGFiveDays --results_dir ./results_encoder \
    --pretraining_epoch 100 --batch_size 8
"""

import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd
import torch

import datautils
from module3_contrastive_training.fk1_model import FK1Model


def train_and_evaluate_single(data_name, data_path, config):
    print(f"Processing dataset: {data_name}")

    train_data, train_labels, train_index = datautils.load_csv_data(data_path, data_name, split='TRAIN')
    test_data, test_labels, test_index = datautils.load_csv_data(data_path, data_name, split='TEST')

    max_train_index = train_index.max()
    adjusted_test_index = test_index + max_train_index + 1

    combined_data = np.concatenate((train_data, test_data), axis=0)
    combined_labels = np.concatenate((train_labels, test_labels), axis=0)
    combined_index = np.concatenate((train_index, adjusted_test_index), axis=0)

    all_data_loader = datautils.create_data_loader(
        combined_data,
        combined_labels,
        combined_index,
        config['batch_size'],
        shuffle=True,
    )
    all_eval_loader = datautils.create_data_loader(
        combined_data,
        combined_labels,
        combined_index,
        config['batch_size'],
        shuffle=False,
    )

    config['dataset_name'] = data_name
    config['dataset_size'] = combined_data.shape[0]
    config['timesteps_len'] = combined_data.shape[1]
    config['input_dims'] = combined_data.shape[2]
    config['n_cluster'] = len(np.unique(combined_labels))
    print(f"config: {config}")

    model = FK1Model(all_data_loader, eval_loader=all_eval_loader, **config)

    train_start = time.time()
    model.Pretraining()
    pretraining_time = time.time() - train_start
    print(f"[Timing] Pretraining time: {datetime.timedelta(seconds=pretraining_time)}")

    eval_start = time.time()
    evaluation = model.evaluate_clustering()
    eval_time = time.time() - eval_start
    total_train_time = time.time() - train_start
    print(f"[Timing] Evaluation time: {datetime.timedelta(seconds=eval_time)}")
    print(f"[Timing] Total training time: {datetime.timedelta(seconds=total_train_time)}")

    encoder_state = model.get_inference_state_dict()
    return evaluation, pretraining_time, total_train_time, eval_time, encoder_state


def main():
    parser = argparse.ArgumentParser(description='RF-LVC encoder pretraining (stages 1-2).')
    parser.add_argument('--dataset_dir', default='./datasets/UCRArchive_2018_csv/', type=str,
                        help='The dataset directory (root of UCRArchive_2018_csv)')
    parser.add_argument('--dataset_name', required=True, type=str,
                        help='Single dataset name under dataset_dir (e.g. "ECGFiveDays")')
    parser.add_argument('--results_dir', default='./results_encoder/', type=str,
                        help='The results directory')
    parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
    parser.add_argument('--repr_dims', type=int, default=64,
                        help='The representation dimension of the final convolution layer')
    parser.add_argument('--hidden_dims', type=int, default=64,
                        help='Hidden channel size of intermediate convolution layers')
    parser.add_argument('--depth', type=int, default=10,
                        help='Number of dilated convolution blocks in the encoder')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of the pre-training phase')
    parser.add_argument('--pretraining_epoch', type=int, default=100, help='Epoch count of the pre-training phase')
    parser.add_argument('--seed', type=int, default=1127, help='Random seed')
    parser.add_argument('--hard_w', type=float, default=0.2, help='Hard negative hardness')
    parser.add_argument('--instance_activation_ratio', type=float, default=0.8,
                        help='eta: the instance-level loss is activated from the first level whose '
                             'receptive field reaches eta times the sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                        help="Torch device of the encoder, e.g. 'cuda' or 'cpu'")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory of the per-epoch pretraining log (loss / ACC / NMI)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    datautils.set_seed(args.seed)

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
    }

    now = datetime.datetime.now()
    time_tag = f"{now.year}.{now.month}.{now.day}_{now.hour:02d}.{now.minute:02d}"
    stem = (
        f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr}_ep{args.pretraining_epoch}_"
        f"seed{args.seed}_hd{args.hidden_dims}_rd{args.repr_dims}_{time_tag}"
    )

    summary_csv = os.path.join(args.results_dir, f"summary_{stem}.csv")
    timing_csv = os.path.join(args.results_dir, f"timing_{stem}.csv")
    encoder_ckpt = os.path.join(args.results_dir, f"encoder_{stem}.pt")
    data_path = os.path.join(args.dataset_dir, args.dataset_name)

    total_run_start = time.time()
    evaluation, pre_t, train_total_t, eval_t, encoder_state = train_and_evaluate_single(
        args.dataset_name,
        data_path,
        config,
    )
    total_run_time = time.time() - total_run_start

    summary_row = {
        'Dataset': args.dataset_name,
        'Final_ACC': evaluation['ACC'],
        'Final_NMI': evaluation['NMI'],
    }
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)

    timing_row = {
        'Dataset': args.dataset_name,
        'PretrainingTime_sec': round(pre_t, 6),
        'TrainTotalTime_sec': round(train_total_t, 6),
        'EvalTime_sec': round(eval_t, 6),
        'ScriptTotalRunTime_sec': round(total_run_time, 6),
    }
    pd.DataFrame([timing_row]).to_csv(timing_csv, index=False)

    torch.save(encoder_state, encoder_ckpt)
    print(f"Encoder saved to: {encoder_ckpt}")
    print(f"Summary saved to: {summary_csv}")
    print(f"Timing saved to: {timing_csv}")

    print("\n===== Final Results =====")
    print(f"Dataset: {args.dataset_name}")
    print(f"Final-layer pooled: acc={evaluation['ACC']}  nmi={evaluation['NMI']}")
    print(f"Pretraining time: {datetime.timedelta(seconds=pre_t)}")
    print(f"Total training time: {datetime.timedelta(seconds=train_total_t)}")
    print(f"Evaluation time: {datetime.timedelta(seconds=eval_t)}")
    print(f"Script total run time: {datetime.timedelta(seconds=total_run_time)}")


if __name__ == "__main__":
    start = time.time()
    main()
    duration = time.time() - start
    print(f"\nrunning time: {datetime.timedelta(seconds=duration)}\n")
