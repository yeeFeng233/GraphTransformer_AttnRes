<div align="center">

# Depth-Wise Attention Residuals for Long-Range Propagation in Graph Transformers

Official implementation of **AttnRes-GT**, including **GPS+AttnRes** and
**GRIT+AttnRes**, evaluated on the full **ECHO benchmark**.

[ECHO paper](https://openreview.net/forum?id=DgkWFPZMPp) |
[ECHO dataset](https://huggingface.co/datasets/lucamiglior/echo-benchmark)

</div>

## Overview

Graph Transformers improve spatial communication across nodes, but information
acquired at an early depth must still remain accessible through subsequent
computation. AttnRes-GT complements spatial graph propagation with depth-wise
residual routing over the causal contribution history of each node.

Within every Graph Transformer layer, AttnRes-GT routes the available history
before graph propagation, appends the resulting graph-propagation contribution,
routes the updated history before the FFN, and then appends the FFN contribution.
A final AttnRes operator routes the complete history to the unchanged task
readout. The graph-propagation operators of GPS and GRIT remain unchanged.

`attnres_block_size=1` selects Full AttnRes. Larger values sum consecutive
sublayer contributions into blocks before depth-wise routing.

## Getting Started

### Environment

The experiments use Python 3.11, PyTorch 2.6.0, CUDA 11.8, PyTorch Geometric,
Lightning, Ray Tune, and Optuna. We recommend using a clean environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The binary package sources in `requirements.txt` match the PyTorch and CUDA
versions used in our experiments. Verify the installation before training:

```bash
python tests/test_model_source_contract.py
python tests/test_attnres_history.py
python tests/smoke_models.py
```

### Download Dataset

Download the ECHO benchmark from its official Hugging Face repository:

```bash
python scripts/download-all.py
```

The evaluated tasks are `diam`, `ecc`, and `sssp` from ECHO-Synth, and `energy`
and `charge` from ECHO-Chem. Raw datasets are not stored in this repository.

### Best Hyperparameters

The machine-readable configurations are provided in
[`configs/best_attnres.csv`](configs/best_attnres.csv). All configurations were
selected by validation MAE. GPS uses multi-head global attention, GRIT uses
ReLU, and `b` denotes the AttnRes block size.

| Model | Task | Layers | Hidden | Batch | b | LR | Weight decay | Dropout | Heads | Attn. dropout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPS+AttnRes | diam | 29 | 96 | 256 | 2 | 7.841e-5 | 7.879e-5 | 0.00 | 4 | 0.08 |
| GPS+AttnRes | ecc | 19 | 162 | 256 | 2 | 2.859e-4 | 1.275e-4 | 0.00 | 2 | 0.05 |
| GPS+AttnRes | sssp | 28 | 56 | 256 | 4 | 1.307e-4 | 1.518e-4 | 0.00 | 2 | 0.03 |
| GPS+AttnRes | energy | 26 | 192 | 256 | 2 | 5.860e-5 | 3.196e-4 | 0.00 | 4 | 0.08 |
| GPS+AttnRes | charge | 24 | 160 | 256 | 2 | 5.640e-5 | 3.930e-5 | 0.00 | 8 | 0.15 |
| GRIT+AttnRes | diam | 40 | 256 | 128 | 8 | 4.780e-4 | 1.254e-4 | 0.05 | 2 | 0.20 |
| GRIT+AttnRes | ecc | 40 | 256 | 128 | 8 | 2.364e-4 | 1.040e-6 | 0.03 | 2 | 0.02 |
| GRIT+AttnRes | sssp | 48 | 96 | 160 | 4 | 2.800e-4 | 1.500e-4 | 0.05 | 8 | 0.00 |
| GRIT+AttnRes | energy | 12 | 64 | 256 | 2 | 7.180e-4 | 9.400e-4 | 0.00 | 4 | 0.17 |
| GRIT+AttnRes | charge | 40 | 128 | 256 | 8 | 3.400e-4 | 3.300e-4 | 0.05 | 2 | 0.20 |

### Train

To reproduce the selected configurations with seeds `1, 2, 3, 4` on four
GPUs, run:

```bash
DETACH=1 \
BEST_CONFIGS="configs/best_attnres.csv" \
SEEDS="1 2 3 4" \
GPU_IDS="0 1 2 3" \
bash scripts/run_attnres_multiseed.sh
```

The summary is written to `results/multiseed/summary.csv`. Existing completed
runs are skipped, and raw logs remain under `results/` without being tracked by
Git.

To repeat hyperparameter search and automatically evaluate each selected
configuration with the same four seeds, run:

```bash
TASKS="diam ecc sssp charge energy" \
MODELS="gps_attnres grit_attnres" \
NUM_GPUS=4 \
N_SAMPLES=24 \
AUTO_MULTI_SEED=1 \
FINAL_SEEDS="1 2 3 4" \
FINAL_GPU_IDS="0 1 2 3" \
DETACH=1 \
bash scripts/run_attnres_search.sh
```

Search, early stopping, and checkpoint selection use validation MAE only; the
test split is evaluated after configuration selection. Final results report the
mean and population standard deviation of test MAE over the four seeds.

## Citation

The AttnRes-GT citation will be added after publication. This implementation
uses the ECHO benchmark datasets and evaluation protocol; please also cite:

```bibtex
@inproceedings{echobenchmark,
  title     = {Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation},
  author    = {Luca Miglior and Matteo Tolloso and Alessio Gravina and Davide Bacciu},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=DgkWFPZMPp}
}
```
