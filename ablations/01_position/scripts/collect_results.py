from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect position ablation seed JSON files.")
    parser.add_argument("--input_dir", type=Path, default=Path("ablations/01_position/results"))
    parser.add_argument("--output_csv", type=Path, default=Path("ablations/01_position/results/summary.csv"))
    return parser.parse_args()


def as_float(value):
    if value is None:
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    rows = []

    for run_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        for variant_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            seed_rows = []
            for json_path in sorted(variant_dir.glob("seed_*.json")):
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                seed_rows.append(
                    {
                        "seed": payload.get("seed"),
                        "test_mae": as_float(metrics.get("test_mae")),
                        "val_mae": as_float(metrics.get("val_mae")),
                        "test_loss": as_float(metrics.get("test_loss")),
                        "val_loss": as_float(metrics.get("val_loss")),
                        "best_epoch": metrics.get("best_epoch"),
                    }
                )

            if not seed_rows:
                continue

            test_maes = [row["test_mae"] for row in seed_rows if row["test_mae"] is not None]
            val_maes = [row["val_mae"] for row in seed_rows if row["val_mae"] is not None]
            best_epochs = [float(row["best_epoch"]) for row in seed_rows if row["best_epoch"] is not None]

            rows.append(
                {
                    "run": run_dir.name,
                    "variant": variant_dir.name,
                    "n_seeds": len(seed_rows),
                    "seeds": " ".join(str(row["seed"]) for row in seed_rows),
                    "test_mae_mean": mean(test_maes) if test_maes else "",
                    "test_mae_std": stdev(test_maes) if len(test_maes) > 1 else 0.0,
                    "val_mae_mean": mean(val_maes) if val_maes else "",
                    "val_mae_std": stdev(val_maes) if len(val_maes) > 1 else 0.0,
                    "best_epoch_mean": mean(best_epochs) if best_epochs else "",
                    "best_epoch_std": stdev(best_epochs) if len(best_epochs) > 1 else 0.0,
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "variant",
        "n_seeds",
        "seeds",
        "test_mae_mean",
        "test_mae_std",
        "val_mae_mean",
        "val_mae_std",
        "best_epoch_mean",
        "best_epoch_std",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()

