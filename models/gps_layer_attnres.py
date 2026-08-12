from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GPSConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch

from .depth_attnres import AttnResHistory, DepthAttnRes


class GPSAttnResConv(GPSConv):
    """GPS propagation with Full/Block AttnRes replacing node residual sums."""

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "layer",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        lidx: int = 0,
    ) -> None:
        super().__init__(
            channels=channels,
            conv=conv,
            heads=heads,
            dropout=dropout,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
        )
        self.lidx = int(lidx)
        self.pre_gt_attnres = DepthAttnRes(channels)
        self.pre_ffn_attnres = DepthAttnRes(channels)

    def _pre_norm(self, norm, x: Tensor, batch: Optional[Tensor]) -> Tensor:
        if norm is None:
            return x
        if self.norm_with_batch:
            return norm(x, batch=batch)
        return norm(x)

    def forward(
        self,
        x: Tensor,
        edge_index,
        batch: Optional[Tensor],
        history: AttnResHistory,
        **kwargs,
    ) -> Tensor:
        del x  # The residual stream is represented explicitly by history.

        routed_gt = self.pre_gt_attnres(history.candidates())
        gt_contributions = []

        if self.conv is not None:
            local_in = self._pre_norm(self.norm1, routed_gt, batch)
            local = self.conv(local_in, edge_index, **kwargs)
            local = F.dropout(local, p=self.dropout, training=self.training)
            gt_contributions.append(local)

        global_in = self._pre_norm(self.norm2, routed_gt, batch)
        dense, mask = to_dense_batch(global_in, batch)
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            global_out, _ = self.attn(
                dense,
                dense,
                dense,
                key_padding_mask=~mask,
                need_weights=False,
            )
        else:
            global_out = self.attn(dense, mask=mask)
        global_out = global_out[mask]
        global_out = F.dropout(
            global_out,
            p=self.dropout,
            training=self.training,
        )
        gt_contributions.append(global_out)

        gt_contribution = sum(gt_contributions)
        history.append(gt_contribution, f"gt_{self.lidx}")

        routed_ffn = self.pre_ffn_attnres(history.candidates())
        ffn_in = self._pre_norm(self.norm3, routed_ffn, batch)
        ffn_contribution = self.mlp(ffn_in)
        history.append(ffn_contribution, f"ffn_{self.lidx}")
        return ffn_contribution

