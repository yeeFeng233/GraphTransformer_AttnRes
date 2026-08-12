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

For every Graph Transformer layer, the implementation:

1. routes the available history before graph propagation;
2. appends the graph-propagation contribution;
3. routes the updated history before the FFN;
4. appends the FFN contribution.

A final AttnRes operator routes the complete history to the unchanged task
readout. The GPS and GRIT graph-propagation operators themselves are retained.
`attnres_block_size=1` selects Full AttnRes; larger values sum consecutive
sublayer contributions into blocks before depth-wise routing.

## Repository Scope

This public release contains the code needed for the paper's main experiments:

```text
models/                 AttnRes history, GPS+AttnRes, and GRIT+AttnRes
scripts/                training, search, selection, and four-seed evaluation
search-space/           GPS+AttnRes and GRIT+AttnRes search spaces for 5 tasks
configs/                validation-selected configurations used in the paper
tests/                  recurrence, source-contract, and forward smoke tests
utils/                  ECHO dataset and Lightning utilities
```

Ablation studies, independently rerun backbones, raw datasets, checkpoints, and
experiment logs are intentionally excluded from this repository.

## Environment

The experiments use Python 3.11, PyTorch, PyTorch Geometric, Lightning, Ray
Tune, and Optuna. We recommend creating a clean environment and installing the
server-exported dependency lock:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch Geometric binary packages must match the installed PyTorch and CUDA
versions. The wheel sources at the top of `requirements.txt` match the locked
PyTorch 2.6.0 and CUDA 11.8 environment used for the experiments.

To regenerate `requirements.txt` from the exact server environment before a
release:

```bash
source /path/to/environment/bin/activate
bash scripts/export_requirements.sh
```

## Data

ECHO data are downloaded from the official Hugging Face repository and are not
stored in Git:

```bash
python scripts/download-all.py
```

The five evaluated tasks are:

- ECHO-Synth: `diam`, `ecc`, and `sssp`;
- ECHO-Chem: `energy` and `charge`.

## Verification

Run these checks before starting an experiment:

```bash
python tests/test_model_source_contract.py
python tests/test_attnres_history.py
python tests/smoke_models.py
```

The latter two tests require PyTorch and PyTorch Geometric.

## Reproduce Selected Configurations

The exact validation-selected configurations are stored in
`configs/best_attnres.csv`. The following command launches seeds 1--4, assigns
one seed to each listed GPU, skips complete existing results, and writes the
summary to `results/multiseed/summary.csv`:

```bash
DETACH=1 \
BEST_CONFIGS="configs/best_attnres.csv" \
SEEDS="1 2 3 4" \
GPU_IDS="0 1 2 3" \
bash scripts/run_attnres_multiseed.sh
```

To run only selected model/task rows, create a filtered copy of the CSV and
pass it through `BEST_CONFIGS`.

## Hyperparameter Search

Search evaluates validation data only. Test metrics are produced after
configuration selection by the fixed four-seed stage.

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

The launcher writes search CSVs and logs under `results/search/`, selects each
configuration by validation MAE, and then runs the four fixed seeds. Ray uses
the short temporary path `/tmp/ar_$UID` by default to avoid Unix socket path
limits.

## Selected Hyperparameters

All models use the ECHO splits and validation-MAE checkpoint selection. GPS
uses multi-head global attention; GRIT uses ReLU. `b` denotes AttnRes block
size.

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

## Result Safety

- Hyperparameters and checkpoints are selected using validation MAE only.
- The test split is excluded from search, early stopping, and checkpoint
  selection.
- Final results use seeds `1, 2, 3, 4` and report test MAE mean and population
  standard deviation.
- Raw logs are written below `results/` and are never tracked by Git.

## Acknowledgment

This repository builds on the official ECHO benchmark implementation. Please
cite ECHO when using its datasets or evaluation protocol:

```bibtex
@inproceedings{echobenchmark,
  title     = {Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation},
  author    = {Luca Miglior and Matteo Tolloso and Alessio Gravina and Davide Bacciu},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=DgkWFPZMPp}
}
```

The AttnRes-GT paper citation will be added after publication.
