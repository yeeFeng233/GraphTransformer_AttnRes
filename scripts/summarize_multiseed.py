import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("results/multiseed"))
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/multiseed/summary.csv"),
    )
    return parser.parse_args()


def mean_std(values):
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    args = parse_args()
    groups = defaultdict(list)

    for path in args.input_dir.rglob("seed_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        config = payload["config"]
        metrics = payload["metrics"]
        groups[(config["conv_layer"], config["task"])].append(
            {
                "seed": payload["seed"],
                "val_mae": float(metrics["val_mae"]),
                "test_mae": float(metrics["test_mae"]),
                "best_epoch": int(metrics["best_epoch"]),
            }
        )

    rows = []
    for (model, task), runs in sorted(groups.items()):
        runs.sort(key=lambda item: item["seed"])
        val_mean, val_std = mean_std([run["val_mae"] for run in runs])
        test_mean, test_std = mean_std([run["test_mae"] for run in runs])
        epoch_mean, epoch_std = mean_std([run["best_epoch"] for run in runs])
        rows.append(
            {
                "model": model,
                "task": task,
                "n_seeds": len(runs),
                "seeds": " ".join(str(run["seed"]) for run in runs),
                "val_mae_mean": val_mean,
                "val_mae_std": val_std,
                "test_mae_mean": test_mean,
                "test_mae_std": test_std,
                "best_epoch_mean": epoch_mean,
                "best_epoch_std": epoch_std,
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "task",
        "n_seeds",
        "seeds",
        "val_mae_mean",
        "val_mae_std",
        "test_mae_mean",
        "test_mae_std",
        "best_epoch_mean",
        "best_epoch_std",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['model']}/{row['task']}: "
            f"test MAE {row['test_mae_mean']:.6f} +/- {row['test_mae_std']:.6f} "
            f"({row['n_seeds']} seeds)"
        )


if __name__ == "__main__":
    main()
