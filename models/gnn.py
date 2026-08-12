from __future__ import annotations

from typing import Optional

import torch
import torch_geometric.nn as pyg_nn
from torch import Tensor
from torch.nn import LeakyReLU, Linear, Module, ModuleList, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from yacs.config import CfgNode as CN

from .depth_attnres import AttnResHistory, DepthAttnRes
from .gps_layer_attnres import GPSAttnResConv
from .grit_layer_attnres import GritAttnResTransformerLayer


class GNN(Module):
    """Corrected GPS+AttnRes and GRIT+AttnRes model."""

    SUPPORTED_MODELS = {"GPSAttnRes", "GRITAttnRes"}

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        node_level_task: bool = False,
        conv_layer: str = "GPSAttnRes",
        dropout_prob: float = 0.0,
        edge_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if conv_layer not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Corrected overlay supports only {sorted(self.SUPPORTED_MODELS)}; "
                f"got {conv_layer!r}."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.conv_name = conv_layer
        self.node_level_task = bool(node_level_task)
        self.attnres_block_size = int(kwargs.get("attnres_block_size", 1))
        if self.attnres_block_size < 1:
            raise ValueError("attnres_block_size must be at least 1.")

        self.emb = Linear(input_dim, hidden_dim)
        self.edge_emb = (
            Linear(edge_dim, hidden_dim)
            if conv_layer == "GRITAttnRes" and edge_dim is not None
            else None
        )

        self.layers = ModuleList()
        for lidx in range(num_layers):
            if conv_layer == "GPSAttnRes":
                num_heads = int(kwargs.get("gps_num_heads", 4))
                if hidden_dim % num_heads != 0:
                    raise ValueError(
                        "GPS hidden_dim must be divisible by gps_num_heads: "
                        f"{hidden_dim} % {num_heads} != 0."
                    )
                local_conv = pyg_nn.GCNConv(hidden_dim, hidden_dim)
                self.layers.append(
                    GPSAttnResConv(
                        channels=hidden_dim,
                        conv=local_conv,
                        heads=num_heads,
                        dropout=dropout_prob,
                        act="relu",
                        norm="layer",
                        attn_type=kwargs.get("gps_attn_type", "multihead"),
                        attn_kwargs={
                            "dropout": float(kwargs.get("gps_attn_dropout", 0.0))
                        },
                        lidx=lidx,
                    )
                )
            else:
                cfg = CN()
                cfg.attn = CN()
                cfg.attn.clamp = 5.0
                cfg.attn.act = kwargs.get("grit_act", "relu")
                cfg.attn.full_attn = True
                cfg.attn.edge_enhance = True
                cfg.attn.O_e = True
                cfg.attn.norm_e = True
                cfg.attn.fwl = False
                cfg.bn_momentum = 0.1
                cfg.bn_no_runner = False
                cfg.dropout = dropout_prob
                cfg.num_heads = int(kwargs.get("grit_num_heads", 4))
                if hidden_dim % cfg.num_heads != 0:
                    raise ValueError(
                        "GRIT hidden_dim must be divisible by grit_num_heads: "
                        f"{hidden_dim} % {cfg.num_heads} != 0."
                    )
                cfg.lidx = lidx

                self.layers.append(
                    GritAttnResTransformerLayer(
                        in_dim=hidden_dim,
                        out_dim=hidden_dim,
                        num_heads=cfg.num_heads,
                        dropout=dropout_prob,
                        attn_dropout=float(
                            kwargs.get("grit_attn_dropout", 0.0)
                        ),
                        act=kwargs.get("grit_act", "relu"),
                        cfg=cfg,
                    )
                )

        self.output_attnres = DepthAttnRes(hidden_dim)
        if node_level_task:
            self.readout = Sequential(
                Linear(hidden_dim, hidden_dim // 2),
                LeakyReLU(),
                Linear(hidden_dim // 2, output_dim),
            )
        else:
            self.readout = Sequential(
                Linear(hidden_dim * 3, (hidden_dim * 3) // 2),
                LeakyReLU(),
                Linear((hidden_dim * 3) // 2, output_dim),
            )

    def forward(self, data: Data) -> Tensor:
        x0 = self.emb(data.x)
        history = AttnResHistory(x0, block_size=self.attnres_block_size)

        if self.conv_name == "GPSAttnRes":
            contribution = x0
            for layer in self.layers:
                contribution = layer(
                    contribution,
                    data.edge_index,
                    batch=data.batch,
                    history=history,
                )
        else:
            graph = data.clone()
            graph.x = x0
            if getattr(data, "edge_attr", None) is not None:
                if self.edge_emb is None:
                    raise ValueError(
                        "GRIT received edge_attr, but edge_dim was not "
                        "configured for the input projection."
                    )
                graph.edge_attr = self.edge_emb(data.edge_attr)
            else:
                graph.edge_attr = None

            for layer in self.layers:
                graph = layer(graph, history=history)

        x = self.output_attnres(history.candidates())
        if not self.node_level_task:
            x = torch.cat(
                [
                    global_add_pool(x, data.batch),
                    global_max_pool(x, data.batch),
                    global_mean_pool(x, data.batch),
                ],
                dim=1,
            )
        return self.readout(x)
