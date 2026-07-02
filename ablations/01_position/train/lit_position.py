from __future__ import annotations

import csv
import pathlib
import time
from typing import Optional

import lightning as L
import torch

from position_gnn import PositionGNN


class LitPositionGNN(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        node_level_task: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        scaling_factor: float = 1.0,
        enable_timing: bool = False,
        timing_csv_base_path: str = "training_timings",
        task: str = "sssp",
        **kwargs,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.task = task
        self.scaling_factor = scaling_factor
        self.enable_timing = enable_timing
        self._epoch_start_time = None
        self.timing_csv_file = None

        self.model = PositionGNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            node_level_task=node_level_task,
            **kwargs,
        )
        self.criterion = torch.nn.MSELoss()
        self.save_hyperparameters()

        if self.enable_timing:
            base = pathlib.Path(timing_csv_base_path)
            base.mkdir(parents=True, exist_ok=True)
            parts = [
                str(kwargs.get("base_model", "model")),
                str(kwargs.get("position_variant", "variant")),
                str(self.task),
                "timing.csv",
            ]
            self.timing_csv_file = base / "_".join(parts)
            if not self.timing_csv_file.exists():
                with self.timing_csv_file.open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(["epoch", "training_time_seconds"])

    def on_train_epoch_start(self):
        if self.enable_timing:
            self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if self.enable_timing and self._epoch_start_time is not None and self.timing_csv_file:
            with self.timing_csv_file.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([self.current_epoch, time.time() - self._epoch_start_time])
            self._epoch_start_time = None

    def forward(self, data):
        return self.model(data)

    def _shared_step(self, batch, metric_prefix: str):
        if batch.x.dtype == torch.float64:
            batch.x = batch.x.float()
        if getattr(batch, "edge_attr", None) is not None and batch.edge_attr.dtype == torch.float64:
            batch.edge_attr = batch.edge_attr.float()
        if batch.y.dtype == torch.float64:
            batch.y = batch.y.float()

        out = self.model(batch).squeeze(-1)
        loss = torch.log10(self.criterion(out, batch.y))

        if self.task == "energy":
            out = 10**out.detach()
            batch.y = 10**batch.y

        mae = torch.nn.functional.l1_loss(
            out.detach() * self.scaling_factor,
            batch.y * self.scaling_factor,
        )
        mse = torch.nn.functional.mse_loss(
            out.detach() * self.scaling_factor,
            batch.y * self.scaling_factor,
        )

        self.log(f"{metric_prefix}_loss", loss, sync_dist=True, batch_size=batch.y.size(0))
        self.log(f"{metric_prefix}_mae", mae, sync_dist=True, prog_bar=metric_prefix == "val", batch_size=batch.y.size(0))
        self.log(f"{metric_prefix}_mse", mse, sync_dist=True, batch_size=batch.y.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

