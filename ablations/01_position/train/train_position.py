from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ABLATION_DIR = CURRENT_DIR.parent
PROJECT_ROOT = ABLATION_DIR.parents[1]
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(ABLATION_DIR / "models"))
sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from lit_position import LitPositionGNN
from utils import KHopTransform, get_dataset


torch.set_float32_matmul_precision("high")


def get_epoch(path: str) -> int | None:
    match = re.search(r"epoch=(\d+)", path or "")
    return int(match.group(1)) if match else None


class EpochSummaryCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        parts = [f"epoch={trainer.current_epoch}"]
        for name in ("train_loss", "train_mae", "val_loss", "val_mae"):
            value = metrics.get(name)
            if value is not None:
                value = value.detach().cpu().item() if hasattr(value, "detach") else value
                parts.append(f"{name}={value:.6g}")
        print("[epoch_summary] " + " ".join(parts), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train insertion-position ablation variants.")
    parser.add_argument("--task", type=str, required=True, choices=["sssp", "ecc", "diam", "charge", "energy"])
    parser.add_argument("--base_model", type=str, required=True, choices=["GPS", "GRIT"])
    parser.add_argument("--position_variant", type=str, required=True, choices=["none", "pre_gt", "pre_ffn", "both"])
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--monitor_metric", type=str, default="val_mae", choices=["val_loss", "val_mae"])
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=80)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--result_json", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--default_root_dir", type=str, default=str(ABLATION_DIR))
    parser.add_argument("--data_root", type=str, default=str(PROJECT_ROOT / "data"))

    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--activ_fun", type=str, default="relu")
    parser.add_argument("--khop", type=int)
    parser.add_argument("--constant_feature", type=float)

    parser.add_argument("--gps_num_heads", type=int)
    parser.add_argument("--gps_attn_dropout", type=float)
    parser.add_argument("--gps_attn_type", type=str, choices=["multihead", "performer"], default="multihead")
    parser.add_argument("--attnres_history_stride", type=int)

    parser.add_argument("--grit_num_heads", type=int)
    parser.add_argument("--grit_attn_dropout", type=float)
    parser.add_argument("--grit_act", type=str, default="relu")
    return parser.parse_args()


def train(seed: int, config: argparse.Namespace) -> dict:
    L.seed_everything(seed, workers=True)

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root=config.data_root,
        task=config.task,
        pre_transform=KHopTransform(k=config.khop) if getattr(config, "khop", None) else None,
        constant_feature=config.constant_feature,
    )

    scaling_factor = data_train.scaling_factor[config.task]
    if scaling_factor is None and config.task in ["charge", "energy"]:
        scaling_factor = 1.0

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(data_train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(data_val, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(data_test, shuffle=False, **loader_kwargs)

    hp_conf = vars(config).copy()
    model = LitPositionGNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=config.task not in ["diam", "energy"],
        scaling_factor=scaling_factor,
        edge_dim=2,
        **hp_conf,
    )

    checkpoint_dir = config.checkpoint_dir
    callbacks = [
        EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            mode="min",
        ),
        ModelCheckpoint(
            monitor=config.monitor_metric,
            save_top_k=1,
            mode="min",
            dirpath=checkpoint_dir,
            filename="{epoch:04d}-{" + config.monitor_metric + ":.6f}",
        ),
    ]
    if config.quiet:
        callbacks.append(EpochSummaryCallback())

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.device,
        callbacks=callbacks,
        deterministic=config.deterministic,
        enable_progress_bar=not config.quiet,
        enable_model_summary=not config.quiet,
        log_every_n_steps=200,
        num_sanity_val_steps=0 if config.quiet else 2,
        default_root_dir=config.default_root_dir,
    )

    trainer.fit(model, train_loader, val_loader)
    fit_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path

    val_results = trainer.validate(model, val_loader, ckpt_path="best", verbose=not config.quiet)
    test_results = trainer.test(model, test_loader, ckpt_path="best", verbose=not config.quiet)

    return {
        "train_loss": fit_metrics.get("train_loss"),
        "val_loss": val_results[0].get("val_loss") if val_results else None,
        "val_mse": val_results[0].get("val_mse") if val_results else None,
        "val_mae": val_results[0].get("val_mae") if val_results else None,
        "test_loss": test_results[0].get("test_loss") if test_results else None,
        "test_mse": test_results[0].get("test_mse") if test_results else None,
        "test_mae": test_results[0].get("test_mae") if test_results else None,
        "best_epoch": get_epoch(best_checkpoint_path),
        "best_checkpoint_path": best_checkpoint_path,
    }


def main() -> None:
    args = parse_args()
    metrics = train(seed=args.seed, config=args)
    print("Metrics:", metrics)
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "task": args.task,
            "base_model": args.base_model,
            "position_variant": args.position_variant,
            "config": vars(args),
            "metrics": metrics,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
