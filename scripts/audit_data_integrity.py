from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Batch


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = (
    SCRIPT_ROOT
    if (SCRIPT_ROOT / "utils").is_dir()
    else SCRIPT_ROOT.parent
)
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.gnn import GNN  # noqa: E402
from utils import get_dataset  # noqa: E402


SYNTH_TASKS = {"diam", "ecc", "sssp"}
ALL_TASKS = ["diam", "ecc", "sssp", "charge", "energy"]
SPLITS = ("train", "val", "test")


def tensor_bytes(value: torch.Tensor | None) -> bytes:
    if value is None:
        return b"<none>"
    tensor = value.detach().cpu().contiguous()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return header + tensor.numpy().tobytes()


def graph_fingerprint(data) -> str:
    digest = hashlib.sha256()
    for field in ("x", "edge_index", "edge_attr"):
        digest.update(field.encode())
        digest.update(tensor_bytes(getattr(data, field, None)))
    return digest.hexdigest()


def dataset_fingerprints(dataset, limit: int) -> set[str]:
    count = len(dataset) if limit <= 0 else min(limit, len(dataset))
    return {graph_fingerprint(dataset[index]) for index in range(count)}


def finite_tensor(name: str, value: torch.Tensor | None, issues: list[str]) -> None:
    if value is None or not torch.is_floating_point(value):
        return
    if not torch.isfinite(value).all():
        issues.append(f"non-finite values in {name}")


def feature_target_checks(dataset, limit: int) -> dict:
    exact_matches: list[str] = []
    correlations: dict[str, float] = {}
    node_x: list[torch.Tensor] = []
    node_y: list[torch.Tensor] = []
    count = min(limit, len(dataset))

    for index in range(count):
        data = dataset[index]
        x = data.x.detach().cpu().float()
        y = data.y.detach().cpu().float().reshape(-1)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if y.numel() == x.shape[0]:
            node_x.append(x)
            node_y.append(y)

    if not node_x:
        return {"exact_matches": exact_matches, "correlations": correlations}

    x_all = torch.cat(node_x)
    y_all = torch.cat(node_y)
    for column in range(x_all.shape[1]):
        feature = x_all[:, column]
        if torch.allclose(feature, y_all, rtol=1e-6, atol=1e-7):
            exact_matches.append(f"x[:, {column}] == y")
        if feature.numel() > 1 and feature.std() > 0 and y_all.std() > 0:
            corr = torch.corrcoef(torch.stack([feature, y_all]))[0, 1]
            correlations[f"x_{column}"] = float(corr)
    return {"exact_matches": exact_matches, "correlations": correlations}


def dense_distances(data) -> np.ndarray:
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    adjacency = csr_matrix(
        (
            np.ones(edge_index.shape[1], dtype=np.float64),
            (edge_index[0], edge_index[1]),
        ),
        shape=(num_nodes, num_nodes),
    )
    return shortest_path(adjacency, directed=False, unweighted=True)


def find_source_node(x: torch.Tensor) -> int | None:
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    for column in range(x.shape[1]):
        values = x[:, column]
        binary = torch.all((values == 0) | (values == 1))
        indices = torch.nonzero(values == 1, as_tuple=False).flatten()
        if binary and indices.numel() == 1:
            return int(indices.item())
    return None


def recompute_synth_targets(task: str, dataset, limit: int) -> dict:
    factor = float(dataset.scaling_factor[task])
    errors: list[float] = []
    source_failures = 0
    disconnected = 0
    count = min(limit, len(dataset))

    for index in range(count):
        data = dataset[index]
        distances = dense_distances(data)
        if not np.isfinite(distances).all():
            disconnected += 1
            continue

        if task == "sssp":
            source = find_source_node(data.x.detach().cpu())
            if source is None:
                source_failures += 1
                continue
            expected = distances[source]
            observed = data.y.detach().cpu().numpy().reshape(-1) * factor
        elif task == "ecc":
            expected = distances.max(axis=1)
            observed = data.y.detach().cpu().numpy().reshape(-1) * factor
        else:
            expected = np.asarray([distances.max()])
            observed = data.y.detach().cpu().numpy().reshape(-1) * factor

        if expected.shape != observed.shape:
            errors.append(float("inf"))
        else:
            errors.append(float(np.max(np.abs(expected - observed))))

    return {
        "checked": len(errors),
        "max_abs_error": max(errors, default=None),
        "source_failures": source_failures,
        "disconnected_graphs": disconnected,
    }


@torch.no_grad()
def model_isolation_checks(task: str, dataset) -> dict:
    examples = [dataset[index].clone() for index in range(min(2, len(dataset)))]
    if len(examples) < 2:
        return {"skipped": "fewer than two graphs"}

    input_dim = int(examples[0].x.shape[-1])
    edge_attr = getattr(examples[0], "edge_attr", None)
    edge_dim = int(edge_attr.shape[-1]) if edge_attr is not None else None
    node_level = task not in {"diam", "energy"}
    results = {}

    for model_name in ("GPSAttnRes", "GRITAttnRes"):
        torch.manual_seed(1234)
        model = GNN(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=16,
            num_layers=2,
            node_level_task=node_level,
            conv_layer=model_name,
            dropout_prob=0.0,
            edge_dim=edge_dim,
            gps_num_heads=4,
            grit_num_heads=4,
            attnres_block_size=1,
        ).eval()

        single_outputs = []
        for example in examples:
            single_outputs.append(model(Batch.from_data_list([example.clone()])))
        batched = Batch.from_data_list([example.clone() for example in examples])
        batched_output = model(batched)
        expected = torch.cat(single_outputs, dim=0)
        batch_delta = float((expected - batched_output).abs().max())

        changed = Batch.from_data_list([example.clone() for example in examples])
        original_output = model(changed)
        changed.y = torch.randn_like(changed.y) * 1_000_000
        changed_output = model(changed)
        label_delta = float((original_output - changed_output).abs().max())

        results[model_name] = {
            "single_vs_batch_max_abs_delta": batch_delta,
            "label_mutation_max_abs_delta": label_delta,
        }
    return results


def split_metadata(dataset, task: str) -> dict:
    return {
        "size": len(dataset),
        "processed_path": str(Path(dataset.processed_paths[0]).resolve()),
        "scaling_factor": dataset.scaling_factor[task],
        "max_ecc": dataset.max_ecc,
        "max_diam": dataset.max_diam,
        "max_sssp": dataset.max_sssp,
        "max_charge": dataset.max_charge,
    }


def audit_task(
    task: str,
    data_root: Path,
    fingerprint_limit: int,
    sample_limit: int,
) -> dict:
    train, val, test, _num_features, _num_classes = get_dataset(
        root=str(data_root),
        task=task,
        pre_transform=None,
        constant_feature=None,
    )
    datasets = {"train": train, "val": val, "test": test}
    issues: list[str] = []
    report = {
        "splits": {
            split: split_metadata(dataset, task)
            for split, dataset in datasets.items()
        }
    }

    processed_paths = [
        report["splits"][split]["processed_path"] for split in SPLITS
    ]
    if len(set(processed_paths)) != len(processed_paths):
        issues.append("processed cache path collision across splits")

    for split, dataset in datasets.items():
        sample = dataset[0]
        finite_tensor(f"{split}.x", sample.x, issues)
        finite_tensor(f"{split}.y", sample.y, issues)
        finite_tensor(f"{split}.edge_attr", getattr(sample, "edge_attr", None), issues)

    factors = {
        split: report["splits"][split]["scaling_factor"]
        for split in SPLITS
    }
    if task in SYNTH_TASKS and len(set(factors.values())) != 1:
        issues.append(
            "split-specific target scaling detected: "
            + ", ".join(f"{split}={value}" for split, value in factors.items())
        )

    fingerprints = {
        split: dataset_fingerprints(dataset, fingerprint_limit)
        for split, dataset in datasets.items()
    }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = fingerprints[left] & fingerprints[right]
        overlaps[f"{left}_{right}"] = len(overlap)
        if overlap:
            issues.append(
                f"{len(overlap)} exact x/edge/edge_attr fingerprints overlap "
                f"between {left} and {right}"
            )
    report["exact_split_overlap"] = overlaps

    report["feature_target_checks"] = {
        split: feature_target_checks(dataset, sample_limit)
        for split, dataset in datasets.items()
    }
    for split, check in report["feature_target_checks"].items():
        if check["exact_matches"]:
            issues.append(
                f"{split} has direct feature-target equality: "
                + ", ".join(check["exact_matches"])
            )

    if task in SYNTH_TASKS:
        report["target_recomputation"] = {
            split: recompute_synth_targets(task, dataset, sample_limit)
            for split, dataset in datasets.items()
        }
        for split, check in report["target_recomputation"].items():
            error = check["max_abs_error"]
            if error is None or not math.isfinite(error) or error > 1e-4:
                issues.append(
                    f"{split} graph-algorithm target recomputation failed: {check}"
                )

    report["model_isolation"] = model_isolation_checks(task, train)
    for model_name, check in report["model_isolation"].items():
        if "skipped" in check:
            continue
        if check["single_vs_batch_max_abs_delta"] > 1e-5:
            issues.append(
                f"{model_name} mixes graphs in a batch "
                f"(delta={check['single_vs_batch_max_abs_delta']})"
            )
        if check["label_mutation_max_abs_delta"] > 0:
            issues.append(
                f"{model_name} output changes when only y changes "
                f"(delta={check['label_mutation_max_abs_delta']})"
            )

    report["issues"] = issues
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    parser.add_argument(
        "--fingerprint_limit",
        type=int,
        default=0,
        help="Graphs hashed per split; 0 checks every graph.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=64,
        help="Graphs used for feature/target and algorithmic checks.",
    )
    parser.add_argument(
        "--output_json",
        default="results/integrity_audit.json",
    )
    args = parser.parse_args()

    report = {
        "data_root": str(Path(args.data_root).resolve()),
        "tasks": {},
        "static_warning": (
            "utils.get_dataset does not pass train.max_charge into val/test. "
            "With the current OR condition in ECHO_Dataset.normalize, Synth "
            "val/test can recompute normalization maxima from their own labels."
        ),
    }
    for task in args.tasks:
        print(f"[audit] {task}", flush=True)
        report["tasks"][task] = audit_task(
            task,
            Path(args.data_root),
            args.fingerprint_limit,
            args.sample_limit,
        )
        for issue in report["tasks"][task]["issues"]:
            print(f"  ISSUE: {issue}", flush=True)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    issue_count = sum(len(item["issues"]) for item in report["tasks"].values())
    print(f"[audit] wrote {output_path}; dynamic issues={issue_count}")
    if issue_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
