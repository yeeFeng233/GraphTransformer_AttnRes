from __future__ import annotations

import argparse
import csv
from pathlib import Path


CONFIG_FIELDS = [
    "task",
    "conv_layer",
    "num_layers",
    "hidden_dim",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout_prob",
    "attnres_block_size",
    "gps_num_heads",
    "gps_attn_dropout",
    "gps_attn_type",
    "grit_num_heads",
    "grit_attn_dropout",
    "grit_act",
    "selection_metric",
    "selection_value",
    "source_csv",
    "trial_id",
]
INTEGER_FIELDS = {
    "num_layers",
    "hidden_dim",
    "batch_size",
    "attnres_block_size",
    "gps_num_heads",
    "grit_num_heads",
}


def normalize_value(field: str, value: str) -> str:
    if value in {"", None}:
        return ""
    if field in INTEGER_FIELDS:
        return str(int(float(value)))
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="results/search/csv",
    )
    parser.add_argument(
        "--output_csv",
        default="configs/best_attnres.csv",
    )
    parser.add_argument(
        "--metric",
        default="val_mae",
        choices=["val_mae", "val_loss"],
    )
    parser.add_argument(
        "--require_tasks",
        nargs="+",
        default=None,
        help="Restrict selection to these tasks and fail if any are missing.",
    )
    parser.add_argument(
        "--require_models",
        nargs="+",
        default=None,
        choices=["GPSAttnRes", "GRITAttnRes"],
        help="Restrict selection to these models and fail if any are missing.",
    )
    args = parser.parse_args()

    best = {}
    for path in sorted(Path(args.input_dir).glob("search_*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                model = row.get("conv_layer")
                task = row.get("task")
                if model not in {"GPSAttnRes", "GRITAttnRes"} or not task:
                    continue
                if args.require_tasks and task not in args.require_tasks:
                    continue
                if args.require_models and model not in args.require_models:
                    continue
                try:
                    value = float(row[args.metric])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (task, model)
                if key not in best or value < best[key][0]:
                    best[key] = (value, row, path)

    if not best:
        raise SystemExit(
            f"No valid AttnRes trials with {args.metric} under {args.input_dir}"
        )
    if args.require_tasks or args.require_models:
        required_tasks = args.require_tasks or sorted(
            {task for task, _model in best}
        )
        required_models = args.require_models or sorted(
            {model for _task, model in best}
        )
        missing = [
            (task, model)
            for task in required_tasks
            for model in required_models
            if (task, model) not in best
        ]
        if missing:
            formatted = ", ".join(
                f"{task}/{model}" for task, model in missing
            )
            raise SystemExit(
                "Refusing to launch final runs; missing validation-selected "
                f"configurations: {formatted}"
            )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONFIG_FIELDS)
        writer.writeheader()
        for (task, model), (value, row, path) in sorted(best.items()):
            selected = {
                field: normalize_value(field, row.get(field, ""))
                for field in CONFIG_FIELDS
            }
            selected["task"] = task
            selected["conv_layer"] = model
            selected["selection_metric"] = args.metric
            selected["selection_value"] = value
            selected["source_csv"] = str(path)
            writer.writerow(selected)

    print(f"Wrote {len(best)} validation-selected configs to {output_path}")


if __name__ == "__main__":
    main()
