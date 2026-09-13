# RF-LVC

Project code of the paper *RF-LVC: Receptive-Field-Aware Local-View Consensus
for Time Series Clustering*.

* `train.py` — trains the encoder.
* `run_consensus_clustering.py` — builds the local views and the consensus partition.

## Environment

* Python 3.8.10
* PyTorch 1.10.0+cu113 (CUDA 11.3); a single GPU, or CPU with `--device cpu`
* NumPy 1.24.4, pandas 2.0.3, SciPy 1.10.1, scikit-learn 1.3.2

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data layout

Under `--dataset_dir`, one directory per dataset:

```
datasets/UCRArchive_2018_csv/<DatasetName>/
├── TRAIN.csv          # (n_train, length), one series per row
├── TRAIN_label.csv    # (n_train, 1)
├── TEST.csv           # (n_test, length)
└── TEST_label.csv     # (n_test, 1)
```

## Usage

Train the encoder:

```bash
python train.py \
  --dataset_dir ./datasets/UCRArchive_2018_csv \
  --dataset_name Mallat \
  --results_dir ./results_encoder
```

Local views and consensus clustering with the trained encoder:

```bash
python run_consensus_clustering.py \
  --dataset_dir ./datasets/UCRArchive_2018_csv \
  --dataset_name Mallat \
  --results_dir ./results_consensus \
  --encoder_ckpt ./results_encoder/encoder_Mallat_....pt \
  --device cuda
```

Without `--encoder_ckpt`, the second command trains an encoder itself, so it can
also be run on its own.
