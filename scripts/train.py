## add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from utils import get_dataset
from utils.litmodels import LitGraphNN

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    required=True,
    choices=["sssp", "ecc", "diam", "charge", "energy"],
)
parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")
parser.add_argument("--seed", type=int, default=5, help="Random seed")
parser.add_argument(
    "--monitor_metric",
    type=str,
    default="val_mae",
    choices=["val_loss", "val_mae"],
    help="Metric used for checkpointing and early stopping",
)
parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of training epochs")
parser.add_argument("--early_stopping_patience", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--result_json", type=str, help="Optional path for metrics and config JSON")

# general gnn parameters
parser.add_argument(
    "--conv_layer",
    required=True,
    choices=["GPSAttnRes", "GRITAttnRes"],
)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--hidden_dim", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
parser.add_argument("--gnn_type", type=str, default="GNN", choices=["GNN"])
parser.add_argument("--dropout_prob", type=float, default=0.0, help="Dropout probability for the backbone")
parser.add_argument("--quiet", action="store_true", help="Disable batch progress output and print epoch summaries")
parser.add_argument(
    "--attnres_block_size",
    type=int,
    default=1,
    help="1 selects Full AttnRes; values >1 select Block AttnRes.",
)

# GRIT-specific
parser.add_argument("--grit_num_heads", type=int, default=4)
parser.add_argument("--grit_attn_dropout", type=float, default=0.0)
parser.add_argument("--grit_act", type=str, default="relu", help="Activation used by GRIT layers")

# GPS-specific
parser.add_argument("--gps_num_heads", type=int, default=4)
parser.add_argument("--gps_attn_dropout", type=float, default=0.0)
parser.add_argument(
    "--gps_attn_type",
    type=str,
    default="multihead",
    choices=["multihead", "performer"],
    help="Attention type for GPS-based models",
)


torch.set_float32_matmul_precision("high")
get_epoch = lambda path: int(re.findall(r"epoch=(\d+)", path)[0])


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


def train(seed, config):
    """Train and validate the model."""
    task = config.task

    L.seed_everything(seed, workers=True)
    batch_size = config.batch_size

    print("Current directory: ", os.getcwd())

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root="./data/",
        task=task,
        pre_transform=None,
        constant_feature=None,
    )

    scaling_factor = data_train.scaling_factor[task]
    if scaling_factor is None and task in ["charge", "energy"]:
        scaling_factor = 1.0

    print(f"Scaling factor for {task}: {scaling_factor}")

    train_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print("Data loaded")

    hp_conf = vars(config)

    model = LitGraphNN(
        input_dim=num_feat,
        output_dim=num_class,
        node_level_task=False if task in ["diam", "energy"] else True,
        scaling_factor=scaling_factor,
        edge_dim=2,
        **hp_conf,
    )

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
            filename="{epoch:04d}-{" + config.monitor_metric + ":.6f}",
        ),
    ]
    if config.quiet:
        callbacks.append(EpochSummaryCallback())

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.device,
        devices=1,
        strategy="auto",
        default_root_dir=(
            str(Path(config.result_json).resolve().parent)
            if config.result_json
            else "results/train"
        ),
        callbacks=callbacks,
        deterministic=config.deterministic,
        enable_progress_bar=not config.quiet,
        enable_model_summary=not config.quiet,
        log_every_n_steps=200,
        num_sanity_val_steps=0 if config.quiet else 2,
    )

    trainer.fit(model, train_loader, val_loader)
    print("after fit:", trainer.callback_metrics.keys())
    fit_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
    best_epoch = get_epoch(trainer.checkpoint_callback.best_model_path)

    val_results = trainer.validate(model, val_loader, ckpt_path="best")
    print("after validate:", trainer.callback_metrics.keys())
    test_results = trainer.test(model, test_loader, ckpt_path="best")
    print("after test:", trainer.callback_metrics.keys())

    metrics = {
        "train_loss": fit_metrics.get("train_loss"),
        "val_loss": val_results[0].get("val_loss") if val_results else None,
        "val_mse": val_results[0].get("val_mse") if val_results else None,
        "val_mae": val_results[0].get("val_mae") if val_results else None,
        "test_loss": test_results[0].get("test_loss") if test_results else None,
        "test_mse": test_results[0].get("test_mse") if test_results else None,
        "test_mae": test_results[0].get("test_mae") if test_results else None,
        "test_acc": test_results[0].get("test_acc") if test_results else None,
        "best_epoch": best_epoch,
        "best_checkpoint_path": trainer.checkpoint_callback.best_model_path,
    }

    return metrics


if __name__ == "__main__":
    args = parser.parse_args()
    metrics = train(
        seed=args.seed,
        config=args,
    )

    print("Metrics: ", metrics)
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed": args.seed,
            "config": vars(args),
            "metrics": metrics,
        }
        temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(result_path)
