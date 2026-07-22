from pathlib import Path
import tomllib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from rdna3_fastmm import inductor
from rdna3_fastmm.external_mm import (
    is_rank49_dynamic_eligible,
    rdna3_rank49_dynamic_v1_out,
)
from rdna3_fastmm.linear import RDNA3_LINEAR_OP, rewrite_eligible_linears


def test_external_mm_falls_back_for_ineligible_cpu_input() -> None:
    left = torch.randn((7, 11), dtype=torch.float16)
    right = torch.randn((11, 5), dtype=torch.float16)
    out = torch.empty((7, 5), dtype=torch.float16)
    reference = left @ right

    with torch.inference_mode():
        assert not is_rank49_dynamic_eligible(left, right, out)
        rdna3_rank49_dynamic_v1_out(left, right, out=out)

    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_backend_adds_external_choice_and_disables_cudagraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    marker_choice = object()
    graph_module = torch.fx.symbolic_trace(lambda value: value.sin())
    inputs = [torch.randn(4)]

    def fake_compile(
        graph: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        options: dict[str, object],
    ) -> object:
        captured.update(graph=graph, inputs=example_inputs, options=options)
        return object()

    monkeypatch.setattr(inductor, "_enable_external_matmul", lambda: True)
    monkeypatch.setattr(inductor, "rewrite_eligible_linears", lambda graph, inputs: 0)
    monkeypatch.setattr(torch._inductor, "compile", fake_compile)
    inductor.compile_backend(
        graph_module,
        inputs,
        mode="reduce-overhead",
        options={"external_matmul": [marker_choice]},
    )

    options = captured["options"]
    assert isinstance(options, dict)
    assert options["external_matmul"] == [
        marker_choice,
        rdna3_rank49_dynamic_v1_out,
    ]
    assert options["max_autotune_gemm"] is True
    assert options["triton.cudagraphs"] is False


def test_backend_preserves_standard_inductor_when_runtime_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    graph_module = torch.fx.symbolic_trace(lambda value: value.cos())

    def fake_compile(
        graph: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        options: dict[str, object],
    ) -> object:
        captured["options"] = options
        return object()

    monkeypatch.setattr(inductor, "_enable_external_matmul", lambda: False)
    monkeypatch.setattr(torch._inductor, "compile", fake_compile)
    inductor.compile_backend(graph_module, [torch.randn(4)], mode="reduce-overhead")

    assert captured["options"] == {"triton.cudagraphs": True}


def test_pyproject_registers_named_dynamo_backend() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["entry-points"]["torch_dynamo_backends"] == {
        "rdna3-fastmm": "rdna3_fastmm.inductor:compile_backend"
    }


@pytest.mark.parametrize(
    "operator_name",
    ("rdna3_fastmm::linear", "rdna3_fastmm::linear_rank7"),
)
def test_triton_linear_ops_register_generated_kernels(operator_name: str) -> None:
    kernels = torch._library.triton.get_triton_kernels_for_op(operator_name)

    assert [kernel.fn.__name__ for kernel in kernels] == [
        "left_transform_kernel",
        "right_transform_weight_kernel",
        "output_transform_kernel",
        "output_transform_bias_kernel",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_rewrite_replaces_eligible_bfloat16_linear() -> None:
    class Linear(torch.nn.Module):
        def forward(
            self,
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            return torch.nn.functional.linear(input_tensor, weight)

    graph_module = torch.fx.symbolic_trace(Linear())
    input_tensor = torch.empty((5120, 4608), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((12288, 4608), device="cuda", dtype=torch.bfloat16)

    rewritten = rewrite_eligible_linears(graph_module, [input_tensor, weight])

    assert rewritten == 1
    targets = {
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    }
    assert RDNA3_LINEAR_OP in targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_rewrite_preserves_unmeasured_linear() -> None:
    class Linear(torch.nn.Module):
        def forward(
            self,
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            return torch.nn.functional.linear(input_tensor, weight)

    graph_module = torch.fx.symbolic_trace(Linear())
    input_tensor = torch.empty((4704, 4608), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((12288, 4608), device="cuda", dtype=torch.bfloat16)

    rewritten = rewrite_eligible_linears(graph_module, [input_tensor, weight])

    assert rewritten == 0
