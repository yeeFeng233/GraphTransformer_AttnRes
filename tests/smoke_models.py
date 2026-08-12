from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn import GNN


def make_graph() -> Data:
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 1, 3], [1, 2, 3, 0, 0, 2]],
        dtype=torch.long,
    )
    return Data(
        x=torch.randn(4, 3),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 2),
        batch=torch.zeros(4, dtype=torch.long),
    )


def check_model(conv_layer: str, block_size: int) -> None:
    model = GNN(
        input_dim=3,
        output_dim=1,
        hidden_dim=16,
        num_layers=3,
        node_level_task=True,
        conv_layer=conv_layer,
        edge_dim=2,
        gps_num_heads=2,
        grit_num_heads=2,
        attnres_block_size=block_size,
    )
    output = model(make_graph())
    assert output.shape == (4, 1)
    output.square().mean().backward()

    router_queries = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith(".query")
    ]
    assert len(router_queries) == 2 * 3 + 1
    assert all(parameter.grad is not None for parameter in router_queries)
    print(
        f"PASS: {conv_layer} block_size={block_size}, "
        f"routers={len(router_queries)}"
    )


if __name__ == "__main__":
    for model_name in ("GPSAttnRes", "GRITAttnRes"):
        for size in (1, 4):
            check_model(model_name, size)
