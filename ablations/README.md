# Ablations

This directory contains paper ablation experiments for AttnRes.

Rules:

- Keep main experiment code unchanged.
- Put ablation-specific model variants, training entry points, configs, run
  scripts, logs, checkpoints, results, and figures inside numbered ablation
  folders.
- Do not overwrite raw logs or seed-level JSON files.
- Use validation-selected checkpoints and report test metrics from those
  checkpoints.
- For paper-facing ablation tables, run seed 1 as a smoke test first; if the
  trend is meaningful, run seeds `1 2 3 4`.

Folders:

- `01_position/`: insertion-position ablation for AttnRes.

