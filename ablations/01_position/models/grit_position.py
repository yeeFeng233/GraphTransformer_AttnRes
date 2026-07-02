from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.register import *

from models.grit_layer_attnres import MultiHeadAttentionLayerGritSparse, get_log_deg


class DepthAttnRes(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(dim))
        self.key_norm = nn.RMSNorm(dim)

    def forward(self, history, partial_block):
        history = history or []
        local_history = history + [partial_block]
        scores = torch.stack(
            [(self.key_norm(h) * self.q).sum(dim=-1) for h in local_history],
            dim=0,
        )
        alpha = scores.softmax(dim=0)
        return sum(alpha[i].unsqueeze(-1) * local_history[i] for i in range(len(local_history)))


class GritPositionTransformerLayer(nn.Module):
    """GRIT+AttnRes layer with independently switchable AttnRes positions."""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        act="relu",
        norm_e=True,
        O_e=True,
        cfg=dict(),
        enable_pre_gt=True,
        enable_pre_ffn=True,
        **kwargs,
    ):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.enable_pre_gt = enable_pre_gt
        self.enable_pre_ffn = enable_pre_ffn

        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = None
        self.bn_no_runner = None
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.0),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn=cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim) if O_e else nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=cfg.bn_momentum,
            )
            self.batch_norm1_e = (
                nn.BatchNorm1d(
                    out_dim,
                    track_running_stats=not self.bn_no_runner,
                    eps=1e-5,
                    momentum=cfg.bn_momentum,
                )
                if norm_e
                else nn.Identity()
            )

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=cfg.bn_momentum,
            )

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

        self.lidx = cfg.get("lidx", None)
        self.block_size = 8
        self.node_res_attn = DepthAttnRes(out_dim) if enable_pre_gt else None
        self.node_res_ffn = DepthAttnRes(out_dim) if enable_pre_ffn else None

    def forward(self, batch, history=None):
        history = history or []
        partial_block = batch.x
        h_attn_in = self.node_res_attn(history, partial_block) if self.node_res_attn is not None else partial_block

        if self.lidx % (self.block_size // 2) == 0:
            history.append(partial_block)
            partial_block = None

        batch.x = h_attn_in
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h_attn_in
        e_in1 = batch.get("edge_attr", None)
        e = None

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero:
                h = h * self.alpha1_h
            h = h_in1 + h
            if e is not None:
                if self.rezero:
                    e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        partial_block = partial_block + h if partial_block is not None else h

        h_ffn_in = self.node_res_ffn(history, partial_block) if self.node_res_ffn is not None else partial_block
        h_in2 = h_ffn_in
        h = self.FFN_h_layer1(h_ffn_in)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero:
                h = h * self.alpha2_h
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h + partial_block
        batch.edge_attr = e if self.update_e else e_in1
        return batch

