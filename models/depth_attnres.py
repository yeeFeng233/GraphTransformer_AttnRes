from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn


class DepthAttnRes(nn.Module):
    """Softmax attention over the depth/history dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))
        self.key_norm = nn.RMSNorm(dim)

    def forward(
        self,
        candidates: Sequence[Tensor],
        *,
        return_weights: bool = False,
    ):
        if not candidates:
            raise ValueError("AttnRes requires at least one history candidate.")

        values = torch.stack(tuple(candidates), dim=0)
        keys = self.key_norm(values)
        scores = (keys * self.query).sum(dim=-1)
        weights = scores.softmax(dim=0)
        routed = (weights.unsqueeze(-1) * values).sum(dim=0)

        if return_weights:
            return routed, weights
        return routed


@dataclass(frozen=True)
class HistorySnapshot:
    sources: Tuple[str, ...]
    completed_count: int
    partial_size: int


class AttnResHistory:
    """Full/Block AttnRes state for one model forward pass."""

    def __init__(self, initial_state: Tensor, block_size: int = 1) -> None:
        if block_size < 1:
            raise ValueError("attnres_block_size must be at least 1.")

        self.block_size = int(block_size)
        self._completed: List[Tensor] = [initial_state]
        self._completed_sources: List[str] = ["x0"]
        self._partial: Tensor | None = None
        self._partial_sources: List[str] = []

    def candidates(self) -> Tuple[Tensor, ...]:
        if self._partial is None:
            return tuple(self._completed)
        return tuple(self._completed) + (self._partial,)

    def snapshot(self) -> HistorySnapshot:
        sources = list(self._completed_sources)
        if self._partial_sources:
            sources.append("partial(" + "+".join(self._partial_sources) + ")")
        return HistorySnapshot(
            sources=tuple(sources),
            completed_count=len(self._completed),
            partial_size=len(self._partial_sources),
        )

    def append(self, contribution: Tensor, source: str) -> None:
        if contribution.shape != self._completed[0].shape:
            raise ValueError(
                "History contribution shape mismatch: "
                f"expected {tuple(self._completed[0].shape)}, "
                f"got {tuple(contribution.shape)} from {source}."
            )

        if self.block_size == 1:
            self._completed.append(contribution)
            self._completed_sources.append(source)
            return

        if self._partial is None:
            self._partial = contribution
        else:
            self._partial = self._partial + contribution
        self._partial_sources.append(source)

        if len(self._partial_sources) == self.block_size:
            self._completed.append(self._partial)
            self._completed_sources.append(
                "block(" + "+".join(self._partial_sources) + ")"
            )
            self._partial = None
            self._partial_sources = []

