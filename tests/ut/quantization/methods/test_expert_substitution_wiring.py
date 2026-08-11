import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
QUANT_METHODS = (
    "vllm_ascend/quantization/methods/w4a8.py",
    "vllm_ascend/quantization/methods/w4a8_mxfp4.py",
    "vllm_ascend/quantization/methods/w8a8_dynamic.py",
)
REQUIRED_SUBSTITUTION_KWARGS = {
    "router_logits",
    "scoring_func",
    "e_score_correction_bias",
    "is_hash_routed",
}


def _manager_update_calls(relative_path: str) -> dict[str, list[ast.Call]]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    calls = {"update_weights": [], "update_weights_multi_card": []}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Attribute) and function.attr in calls:
            calls[function.attr].append(node)
    return calls


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


@pytest.mark.parametrize("relative_path", QUANT_METHODS)
def test_quant_method_wires_substitution_for_single_and_multi_card(
    relative_path,
):
    calls = _manager_update_calls(relative_path)

    for method_name, method_calls in calls.items():
        assert method_calls, f"{relative_path} does not call {method_name}"
        for call in method_calls:
            # layer, topk_ids, log2phy, topk_weights stay positional so the
            # original router weights flow unchanged into the manager/GMM path.
            assert len(call.args) >= 4
            assert isinstance(call.args[3], ast.Name)
            assert call.args[3].id == "topk_weights"
            keyword_names = {keyword.arg for keyword in call.keywords}
            assert keyword_names >= REQUIRED_SUBSTITUTION_KWARGS


def test_single_and_multi_card_managers_never_write_back_topk_weights():
    manager_path = (
        REPO_ROOT
        / "vllm_ascend/expert_offload/expert_offload_manager.py"
    )
    tree = ast.parse(manager_path.read_text())
    wrappers = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"update_weights", "update_weights_multi_card"}
    }

    assert wrappers.keys() == {"update_weights", "update_weights_multi_card"}
    for function in wrappers.values():
        for call in ast.walk(function):
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr == "copy_":
                assert _root_name(call.func.value) != "topk_weights"
