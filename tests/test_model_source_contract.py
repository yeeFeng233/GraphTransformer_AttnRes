from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse(relative_path: str) -> ast.AST:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8-sig"))


def called_method_on_name(tree: ast.AST, owner: str, method: str) -> int:
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == owner
        ):
            count += 1
    return count


def test_layer_history_writes() -> None:
    gps = parse("models/gps_layer_attnres.py")
    grit = parse("models/grit_layer_attnres.py")
    gnn = parse("models/gnn.py")
    assert called_method_on_name(gps, "history", "append") == 2
    assert called_method_on_name(grit, "history", "append") == 2
    assert called_method_on_name(gnn, "history", "append") == 0


def test_only_attnres_models() -> None:
    source = (ROOT / "models/gnn.py").read_text(encoding="utf-8-sig")
    assert '{"GPSAttnRes", "GRITAttnRes"}' in source
    assert "attnres_history_stride" not in source
    assert "output_attnres" in source
    assert "hidden_dim must be divisible" in source
    assert "GRIT received edge_attr" in source


def test_no_node_residual_additions_in_grit() -> None:
    source = (ROOT / "models/grit_layer_attnres.py").read_text(
        encoding="utf-8-sig"
    )
    assert "h_in1 + h" not in source
    assert "h_in2 + h" not in source
    assert "e + e_in1" in source


def test_search_space_contract() -> None:
    files = sorted((ROOT / "search-space").glob("*/*_attnres.yaml"))
    assert len(files) == 10
    for path in files:
        source = path.read_text(encoding="utf-8-sig")
        assert "attnres_block_size:" in source
        assert "attnres_history_stride" not in source
        assert ("GPSAttnRes" in source) ^ ("GRITAttnRes" in source)


def test_search_does_not_evaluate_test_split() -> None:
    source = (ROOT / "scripts/search.py").read_text(encoding="utf-8-sig")
    assert "trainer.test(" not in source
    assert '"test_mae"' not in source
    assert '"test_loss"' not in source
    assert "from ray.air import session" not in source
    assert "tune.report(" in source


def test_multiseed_results_are_restart_safe() -> None:
    runner = (ROOT / "scripts/run_attnres_multiseed.py").read_text(
        encoding="utf-8-sig"
    )
    trainer = (ROOT / "scripts/train.py").read_text(encoding="utf-8-sig")
    summary = (
        ROOT / "scripts/summarize_multiseed.py"
    ).read_text(encoding="utf-8-sig")
    assert "is_complete_result" in runner
    assert "temporary_path.replace(result_path)" in trainer
    assert "--expected_num_seeds" in summary


def test_runtime_tests_add_project_root_to_import_path() -> None:
    for relative_path in (
        "tests/test_attnres_history.py",
        "tests/smoke_models.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
        assert "Path(__file__).resolve().parents[1]" in source
        assert "sys.path.insert(0, str(PROJECT_ROOT))" in source


def test_ray_socket_path_is_short() -> None:
    source = (ROOT / "scripts/run_attnres_search.sh").read_text(
        encoding="utf-8-sig"
    )
    assert 'RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ar_${UID:-user}}"' in source
    assert '--ray_temp_dir "${RAY_TEMP_DIR}"' in source
    assert '$(pwd)/${OUTPUT_ROOT}/ray_tmp' not in source


def test_asha_grace_is_not_aggressive() -> None:
    search = (ROOT / "scripts/search.py").read_text(encoding="utf-8-sig")
    launcher = (ROOT / "scripts/run_attnres_search.sh").read_text(
        encoding="utf-8-sig"
    )
    assert "default_grace_epochs = min(200, args.max_epochs)" in search
    assert 'ASHA_GRACE_EPOCHS="${ASHA_GRACE_EPOCHS:-200}"' in launcher
    assert '--asha_grace_period "${ASHA_GRACE_REPORTS}"' in launcher


def test_search_can_chain_validation_selection_and_four_seeds() -> None:
    launcher = (ROOT / "scripts/run_attnres_search.sh").read_text(
        encoding="utf-8-sig"
    )
    selector = (ROOT / "scripts/select_best_configs.py").read_text(
        encoding="utf-8-sig"
    )
    assert 'AUTO_MULTI_SEED="${AUTO_MULTI_SEED:-0}"' in launcher
    assert "scripts/select_best_configs.py" in launcher
    assert "scripts/run_attnres_multiseed.sh" in launcher
    assert "AUTO_MULTI_SEED requires MONITOR_METRIC=val_mae." in launcher
    assert "requires exactly four FINAL_SEEDS" in launcher
    assert "--require_tasks" in selector
    assert "--require_models" in selector
    assert "missing validation-selected" in selector


if __name__ == "__main__":
    test_layer_history_writes()
    test_only_attnres_models()
    test_no_node_residual_additions_in_grit()
    test_search_space_contract()
    test_search_does_not_evaluate_test_split()
    test_multiseed_results_are_restart_safe()
    test_runtime_tests_add_project_root_to_import_path()
    test_ray_socket_path_is_short()
    test_asha_grace_is_not_aggressive()
    test_search_can_chain_validation_selection_and_four_seeds()
    print("PASS: AttnRes-GT model source contract")
