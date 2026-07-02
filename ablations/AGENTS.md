# Ablation Rules

- Do not modify the main `models/`, `scripts/train.py`, or search scripts for
  ablation variants unless explicitly requested.
- Keep each ablation study in its own numbered folder.
- Heavy outputs should remain local:
  - `ablations/**/results/`
  - `ablations/**/logs/`
  - `ablations/**/checkpoints/`
  - generated figures
- Copy only final lightweight tables/figures into `paper_bigdata2026/` after
  results are verified.
- Do not invent numbers. Missing metrics remain `TODO:`.

