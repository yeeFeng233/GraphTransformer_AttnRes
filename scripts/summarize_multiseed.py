from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="results/multiseed",
    )
    parser.add_argument(
        "--output_csv",
        default="results/multiseed/summary.csv",
    )
    parser.add_argument(
        "--expected_num_seeds",
        type=int,
        default=4,
        help="Required unique seeds per model/task; use 0 to allow partial groups.",
    )
    args = parser.parse_args()

    groups = {}
    for path in Path(args.input_root).glob("*/*/seed_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        model = payload["config"]["conv_layer"]
        task = payload["config"]["task"]
        groups.setdefault((model, task), []).append(payload)

    rows = []
    for (model, task), payloads in sorted(groups.items()):
        payloads.sort(key=lambda item: int(item["seed"]))
        seeds = [int(item["seed"]) for item in payloads]
        if len(seeds) != len(set(seeds)):
            raise SystemExit(f"Duplicate seeds for {model}/{task}: {seeds}")
        if args.expected_num_seeds and len(seeds) != args.expected_num_seeds:
            raise SystemExit(
                f"Incomplete {model}/{task}: expected "
                f"{args.expected_num_seeds} seeds, found {seeds}"
            )
        test_values = [
            float(item["metrics"]["test_mae"])
            for item in payloads
        ]
        val_values = [
            float(item["metrics"]["val_mae"])
            for item in payloads
        ]
        rows.append(
            {
                "model": model,
                "task": task,
                "num_seeds": len(payloads),
                "seeds": " ".join(str(seed) for seed in seeds),
                "test_mae_mean": statistics.fmean(test_values),
                "test_mae_std": statistics.pstdev(test_values),
                "val_mae_mean": statistics.fmean(val_values),
                "val_mae_std": statistics.pstdev(val_values),
            }
        )

    if not rows:
        raise SystemExit(f"No seed JSON files under {args.input_root}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} summaries to {output_path}")


if __name__ == "__main__":
    main()
