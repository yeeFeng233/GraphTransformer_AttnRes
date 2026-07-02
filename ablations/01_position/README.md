# 01 Position Ablation

Goal: test which AttnRes insertion point matters most.

Variants:

| Variant | Pre-GT AttnRes | Pre-FFN AttnRes |
|---|---:|---:|
| `none` | no | no |
| `pre_gt` | yes | no |
| `pre_ffn` | no | yes |
| `both` | yes | yes |

Primary task:

- `gps_ecc`

Auxiliary tasks:

- `gps_diam`
- `grit_sssp`

Seed policy:

- Smoke test: `SEEDS="1"`.
- Paper-facing run: `SEEDS="1 2 3 4"`.

Example smoke test:

```bash
RUNS="gps_ecc" VARIANTS="none pre_gt pre_ffn both" SEEDS="1" GPU_IDS="0" \
bash ablations/01_position/scripts/run_multiseed.sh
```

Example four-seed run:

```bash
RUNS="gps_ecc gps_diam grit_sssp" \
VARIANTS="none pre_gt pre_ffn both" \
SEEDS="1 2 3 4" GPU_IDS="0 1 2 3" SKIP_EXISTING=1 \
nohup bash ablations/01_position/scripts/run_multiseed.sh \
> ablations/01_position/logs/run_position.log 2>&1 &
```

Outputs:

- Seed-level JSON: `ablations/01_position/results/<run>/<variant>/seed_<seed>.json`
- Logs: `ablations/01_position/logs/<run>/<variant>/seed_<seed>.log`
- Checkpoints: `ablations/01_position/checkpoints/<run>/<variant>/seed_<seed>/`
- Summary CSV: `ablations/01_position/results/summary.csv`

