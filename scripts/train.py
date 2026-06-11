## add parent directory to path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from utils import KHopTransform, get_dataset
from utils.litmodels import LitGraphNN

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, help="Task to run: [sssp, ecc, diam, charge, energy]")
parser.add_argument("--device", type=str, default="gpu", help="Device to use for training")

# general gnn parameters
parser.add_argument("--conv_layer", type=str)
parser.add_argument("--num_layers", type=int, help="Number of layers in the GNN")
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of the GNN")
parser.add_argument("--lr", type=float, help="Learning rate for the optimizer")
parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for the DataLoader")
parser.add_argument("--gnn_type", type=str)
parser.add_argument("--dropout_prob", type=float, default=0.0, help="Dropout probability for the backbone")
parser.add_argument("--quiet", action="store_true", help="Disable batch progress output and print epoch summaries")

# GRIT-specific
parser.add_argument("--grit_num_heads", type=int, help="Number of heads in the GRIT attention")
parser.add_argument("--grit_attn_dropout", type=float, help="Dropout ratio for the GRIT attention layer")

# GPS-specific
parser.add_argument("--gps_num_heads", type=int, help="Number of heads in GPS global attention")
parser.add_argument("--gps_attn_dropout", type=float, help="Dropout ratio for GPS global attention")
parser.add_argument(
    "--gps_attn_type",
    type=str,
    choices=["multihead", "performer"],
    help="Attention type for GPS-based models",
)
parser.add_argument(
    "--attnres_history_stride",
    type=int,
    help="History stride for GPSAttnRes depth residuals",
)

# adgn, swan specific params
parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for the ADGN model")
parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the ADGN model")
parser.add_argument("--activ_fun", type=str, default="tanh", help="Activation function for the ADGN model")
parser.add_argument("--graph_conv", type=str, default="GCNConv", help="Graph convolution layer for the ADGN model")
parser.add_argument("--bias", type=bool, help="Use bias in the ADGN model")
parser.add_argument("--train_weights", type=bool)
parser.add_argument("--weight_sharing", type=bool, help="Use weight sharing in the ADGN model")

# drew specific parameters
parser.add_argument("--khop", type=int)
parser.add_argument("--delay", type=bool)
parser.add_argument("--constant_feature", type=float, help="Constant feature")

# gcn2 params
parser.add_argument("--alpha", type=float, help="Alpha for the GCN2 model")

# phdgn specific parameters
parser.add_argument("--beta", type=float, help="Beta parameter for the PHDGN model")
parser.add_argument("--p_conv_mode", type=str, choices=["naive", "gcn"], help="P convolution mode for the PhDGN model")
parser.add_argument("--q_conv_mode", type=str, choices=["naive", "gcn"], help="Q convolution mode for the PhDGN model")
parser.add_argument("--doubled_dim", type=bool, choices=[True, False], help="Whether to double the dimension in the PhDGN model")
parser.add_argument("--final_state", type=str, choices=["p", "q", "pq"], help="Final state mode for the PhDGN model")
parser.add_argument("--dampening_mode", type=str, choices=["param", "param+", "MLP4ReLU", "DGNReLU", "none"], help="Dampening mode for the PhDGN model")
parser.add_argument("--external_mode", type=str, choices=["MLP4Sin", "DGNtanh", "none"], help="External mode for the PhDGN model")


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
    config = parser.parse_args()
    task = config.task

    L.seed_everything(seed)
    batch_size = config.batch_size

    print("Current directory: ", os.getcwd())

    data_train, data_val, data_test, num_feat, num_class = get_dataset(
        root="./data/",
        task=task,
        pre_transform=(
            KHopTransform(k=config.khop) if config.gnn_type == "DRew_GCN" else None
        ),
        constant_feature=config.constant_feature,
    )

    scaling_factor = data_train.scaling_factor[task]
    if scaling_factor is None and task in ["charge", "energy"]:
        scaling_factor = 1.0

    print(f"Scaling factor for {task}: {scaling_factor}")

    train_loader = DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        data_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
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
        EarlyStopping(monitor="val_loss", patience=100),
        ModelCheckpoint(monitor="val_loss", save_top_k=1),
    ]
    if config.quiet:
        callbacks.append(EpochSummaryCallback())

    trainer = L.Trainer(
        max_epochs=1000,
        accelerator=config.device,
        strategy="ddp_find_unused_parameters_true" if config.device == "gpu" else "auto",
        callbacks=callbacks,
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
        seed=5,
        config=args,
    )

    print("Metrics: ", metrics)
