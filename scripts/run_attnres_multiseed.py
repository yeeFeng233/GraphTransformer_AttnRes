from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


COMMON_FIELDS = [
    "num_layers",
    "hidden_dim",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout_prob",
    "attnres_block_size",
]
SUPPORTED_MODELS = {"GPSAttnRes", "GRITAttnRes"}


def add_option(command: list[str], name: str, value: str) -> None:
    if value not in {"", None}:
        command.extend([f"--{name}", str(value)])


def is_complete_result(
    path: Path,
    row: dict[str, str],
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        config = payload["config"]
        metrics = payload["metrics"]
        return (
            int(payload["seed"]) == seed
            and config["task"] == row["task"]
            and config["conv_layer"] == row["conv_layer"]
            and metrics.get("test_mae") is not None
            and metrics.get("val_mae") is not None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def command_for(
    row: dict[str, str],
    seed: int,
    result_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        args.python,
        "scripts/train.py",
        "--task",
        row["task"],
        "--conv_layer",
        row["conv_layer"],
        "--seed",
        str(seed),
        "--max_epochs",
        str(args.max_epochs),
        "--early_stopping_patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--monitor_metric",
        args.monitor_metric,
        "--result_json",
        str(result_path),
        "--quiet",
    ]
    for field in COMMON_FIELDS:
        add_option(command, field, row.get(field, ""))

    if row["conv_layer"] == "GPSAttnRes":
        for field in (
            "gps_num_heads",
            "gps_attn_dropout",
            "gps_attn_type",
        ):
            add_option(command, field, row.get(field, ""))
    else:
        for field in (
            "grit_num_heads",
            "grit_attn_dropout",
            "grit_act",
        ):
            add_option(command, field, row.get(field, ""))
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--best_configs",
        default="configs/best_attnres.csv",
    )
    parser.add_argument(
        "--output_root",
        default="results/multiseed",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--monitor_metric",
        default="val_mae",
        choices=["val_mae", "val_loss"],
    )
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if len(args.seeds) > len(args.gpus):
        raise SystemExit("Provide at least one GPU ID per concurrent seed.")

    with Path(args.best_configs).open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        configs = list(csv.DictReader(handle))
    if not configs:
        raise SystemExit(f"No configs in {args.best_configs}")

    output_root = Path(args.output_root)
    for row in configs:
        task = row["task"]
        model = row["conv_layer"]
        if model not in SUPPORTED_MODELS:
            raise SystemExit(
                f"Unsupported model in {args.best_configs}: {model!r}"
            )
        run_dir = output_root / model / task
        run_dir.mkdir(parents=True, exist_ok=True)
        processes = []

        for seed, gpu in zip(args.seeds, args.gpus):
            result_path = run_dir / f"seed_{seed}.json"
            log_path = run_dir / f"seed_{seed}.log"
            if (
                args.skip_existing
                and is_complete_result(result_path, row, seed)
            ):
                print(f"Skip existing {model}/{task}/seed_{seed}")
                continue

            command = command_for(row, seed, result_path, args)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(
                "\n[launch] " + " ".join(command) + "\n"
            )
            log_handle.flush()
            print(f"Start {model}/{task}/seed_{seed} on GPU {gpu}")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            processes.append((process, log_handle, seed))

        failures = []
        for process, log_handle, seed in processes:
            return_code = process.wait()
            log_handle.close()
            if return_code:
                failures.append(seed)
        if failures:
            raise SystemExit(f"{model}/{task} failed seeds: {failures}")


if __name__ == "__main__":
    main()
