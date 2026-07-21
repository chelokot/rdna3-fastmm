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
