from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.depth_attnres import AttnResHistory, DepthAttnRes


def test_full_history() -> None:
    x0 = torch.randn(5, 8)
    history = AttnResHistory(x0, block_size=1)
    history.append(torch.randn_like(x0), "gt_0")
    assert history.snapshot().sources == ("x0", "gt_0")
    history.append(torch.randn_like(x0), "ffn_0")
    assert history.snapshot().sources == ("x0", "gt_0", "ffn_0")
    assert len(history.candidates()) == 3


def test_block_history() -> None:
    x0 = torch.zeros(5, 8)
    history = AttnResHistory(x0, block_size=4)
    contributions = [
        torch.full_like(x0, 1.0),
        torch.full_like(x0, 2.0),
        torch.full_like(x0, 3.0),
        torch.full_like(x0, 4.0),
    ]
    names = ["gt_0", "ffn_0", "gt_1", "ffn_1"]

    for index, (value, name) in enumerate(zip(contributions, names), start=1):
        history.append(value, name)
        snapshot = history.snapshot()
        if index < 4:
            assert snapshot.completed_count == 1
            assert snapshot.partial_size == index

    assert history.snapshot().partial_size == 0
    assert history.snapshot().completed_count == 2
    assert torch.equal(history.candidates()[1], sum(contributions))


def test_depth_attention() -> None:
    router = DepthAttnRes(8)
    candidates = tuple(torch.randn(5, 8, requires_grad=True) for _ in range(3))
    routed, weights = router(candidates, return_weights=True)
    assert routed.shape == (5, 8)
    assert weights.shape == (3, 5)
    assert torch.allclose(weights.sum(dim=0), torch.ones(5))
    routed.square().mean().backward()
    assert router.query.grad is not None


if __name__ == "__main__":
    test_full_history()
    test_block_history()
    test_depth_attention()
    print("PASS: Full/Block history and depth attention")
