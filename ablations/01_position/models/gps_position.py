from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GPSConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch


class DepthAttnRes(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        self.key_norm = nn.RMSNorm(dim)

    def forward(
        self,
        history: Optional[List[torch.Tensor]],
        partial_block: torch.Tensor,
    ) -> torch.Tensor:
        history = history or []
        local_history = history + [partial_block]
        scores = torch.stack(
            [(self.key_norm(h) * self.q).sum(dim=-1) for h in local_history],
            dim=0,
        )
        alpha = scores.softmax(dim=0)
        return sum(alpha[i].unsqueeze(-1) * local_history[i] for i in range(len(local_history)))


class GPSPositionConv(GPSConv):
    """GPSConv with independently switchable AttnRes insertion positions."""

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
        lidx: int = 0,
        attnres_history_stride: int = 2,
        enable_pre_gt: bool = True,
        enable_pre_ffn: bool = True,
    ):
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
        self.lidx = lidx
        self.attnres_history_stride = max(1, attnres_history_stride)
        self.enable_pre_gt = enable_pre_gt
        self.enable_pre_ffn = enable_pre_ffn
        self.pre_res_attn = DepthAttnRes(channels) if enable_pre_gt else None
        self.ffn_res_attn = DepthAttnRes(channels) if enable_pre_ffn else None

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        batch: Optional[torch.Tensor] = None,
        history: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        history = history or []
        x_in = self.pre_res_attn(history, x) if self.pre_res_attn is not None else x

        hs = []
        if self.conv is not None:
            h = self.conv(x_in, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x_in
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        h, mask = to_dense_batch(x_in, batch)
        if isinstance(self.attn, torch.nn.MultiheadAttention):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        else:
            h = self.attn(h, mask=mask)

        h = h[mask]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x_in
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs) if hs else x_in
        ffn_in = self.ffn_res_attn(history, out) if self.ffn_res_attn is not None else out
        out = ffn_in + self.mlp(ffn_in)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

