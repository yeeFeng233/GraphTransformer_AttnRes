from __future__ import annotations

from typing import Optional

import torch
import torch_geometric.nn as pyg_nn
from torch import nn
from torch.nn import LeakyReLU, Linear, Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from yacs.config import CfgNode as CN

from gps_position import GPSPositionConv
from grit_position import GritPositionTransformerLayer
from models.grit_layer import GritTransformerLayer


def variant_flags(position_variant: str) -> tuple[bool, bool]:
    if position_variant == "none":
        return False, False
    if position_variant == "pre_gt":
        return True, False
    if position_variant == "pre_ffn":
        return False, True
    if position_variant == "both":
        return True, True
    raise ValueError(f"Unknown position variant: {position_variant}")


class PositionGNN(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        node_level_task: bool = False,
        base_model: str = "GPS",
        position_variant: str = "both",
        dropout_prob: float = 0.0,
        edge_dim: Optional[int] = None,
        activ_fun: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.base_model = base_model
        self.position_variant = position_variant
        self.dropout = nn.Dropout(p=dropout_prob)
        self.edge_dim = edge_dim
        self.activation = getattr(torch, activ_fun, torch.relu)

        enable_pre_gt, enable_pre_ffn = variant_flags(position_variant)

        self.emb = Linear(self.input_dim, self.hidden_dim)
        self.edge_emb = None
        if base_model == "GRIT" and self.edge_dim is not None:
            self.edge_emb = Linear(self.edge_dim, self.hidden_dim)

        self.conv = ModuleList()
        for lidx in range(num_layers):
            if base_model == "GPS":
                gps_num_heads = kwargs.get("gps_num_heads", 2)
                gps_attn_dropout = kwargs.get("gps_attn_dropout", 0.0)
                gps_attn_type = kwargs.get("gps_attn_type", "multihead")
                attnres_history_stride = kwargs.get("attnres_history_stride", 2)
                local_conv = pyg_nn.GCNConv(self.hidden_dim, self.hidden_dim)
                self.conv.append(
                    GPSPositionConv(
                        channels=self.hidden_dim,
                        conv=local_conv,
                        heads=gps_num_heads,
                        dropout=dropout_prob,
                        act="relu",
                        norm="layer",
                        attn_type=gps_attn_type,
                        attn_kwargs={"dropout": gps_attn_dropout},
                        lidx=lidx,
                        attnres_history_stride=attnres_history_stride,
                        enable_pre_gt=enable_pre_gt,
                        enable_pre_ffn=enable_pre_ffn,
                    )
                )
            elif base_model == "GRIT":
                grit_num_heads = kwargs.get("grit_num_heads", 4)
                grit_attn_dropout = kwargs.get("grit_attn_dropout", 0.0)
                grit_act = kwargs.get("grit_act", "relu")

                cfg = CN()
                cfg.attn = CN()
                cfg.attn.clamp = 5.0
                cfg.attn.act = grit_act
                cfg.attn.full_attn = True
                cfg.attn.edge_enhance = True
                cfg.attn.O_e = True
                cfg.attn.norm_e = True
                cfg.attn.fwl = False
                cfg.bn_momentum = 0.1
                cfg.bn_no_runner = False
                cfg.dropout = dropout_prob
                cfg.num_heads = grit_num_heads
                cfg.lidx = lidx

                layer_cls = GritTransformerLayer if position_variant == "none" else GritPositionTransformerLayer
                kwargs_extra = {}
                if position_variant != "none":
                    kwargs_extra = {
                        "enable_pre_gt": enable_pre_gt,
                        "enable_pre_ffn": enable_pre_ffn,
                    }
                self.conv.append(
                    layer_cls(
                        in_dim=self.hidden_dim,
                        out_dim=self.hidden_dim,
                        num_heads=grit_num_heads,
                        dropout=dropout_prob,
                        attn_dropout=grit_attn_dropout,
                        act=grit_act,
                        cfg=cfg,
                        **kwargs_extra,
                    )
                )
            else:
                raise ValueError(f"Unsupported base_model: {base_model}")

        self.node_level_task = node_level_task
        if self.node_level_task:
            self.readout = Sequential(
                Linear(self.hidden_dim, self.hidden_dim // 2),
                LeakyReLU(),
                Linear(self.hidden_dim // 2, self.output_dim),
            )
        else:
            self.readout = Sequential(
                Linear(self.hidden_dim * 3, (self.hidden_dim * 3) // 2),
                LeakyReLU(),
                Linear((self.hidden_dim * 3) // 2, self.output_dim),
            )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.emb(x)
        node_history = []

        if self.base_model == "GRIT":
            data_cloned = data.clone()
            data_cloned.x = x
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                data_cloned.edge_attr = self.edge_emb(data.edge_attr) if self.edge_emb is not None else data.edge_attr
            else:
                data_cloned.edge_attr = None

        for i, conv in enumerate(self.conv):
            if self.base_model == "GPS":
                x = conv(x, edge_index, batch=batch, history=node_history)
                x = self.activation(x)
                x = self.dropout(x)
                history_stride = getattr(conv, "attnres_history_stride", 1)
                if i % history_stride == 0:
                    node_history.append(x)
            else:
                if self.position_variant == "none":
                    data_cloned = conv(data_cloned)  # type: ignore[has-type]
                else:
                    data_cloned = conv(data_cloned, history=node_history)  # type: ignore[has-type]
                    node_history.append(data_cloned.x)

        if self.base_model == "GRIT":
            x = data_cloned.x  # type: ignore[has-type]

        if not self.node_level_task:
            x = torch.cat(
                [
                    global_add_pool(x, batch),
                    global_max_pool(x, batch),
                    global_mean_pool(x, batch),
                ],
                dim=1,
            )

        return self.readout(x)

