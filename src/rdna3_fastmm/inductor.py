from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch._inductor
import torch._inductor.config
from torch.fx import GraphModule

from rdna3_fastmm.external_mm import rdna3_rank49_dynamic_v1_out
from rdna3_fastmm.runtime import is_tested_runtime


class CompiledGraph(Protocol):
    def __call__(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


def _enable_external_matmul() -> bool:
    if torch.is_grad_enabled() or torch.version.hip is None:
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    return is_tested_runtime(device)


def compile_backend(
    graph_module: GraphModule,
    example_inputs: Sequence[torch.Tensor],
    *,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
) -> CompiledGraph:
    inductor_options = torch._inductor.list_mode_options(mode)
    inductor_options.update(options or {})
    if _enable_external_matmul():
        configured_choices = inductor_options.get(
            "external_matmul", torch._inductor.config.external_matmul
        )
        external_choices = list(configured_choices)
        if rdna3_rank49_dynamic_v1_out not in external_choices:
            external_choices.append(rdna3_rank49_dynamic_v1_out)
        inductor_options["external_matmul"] = external_choices
        inductor_options["max_autotune_gemm"] = True
        inductor_options["triton.cudagraphs"] = False
    return torch._inductor.compile(
        graph_module,
        list(example_inputs),
        options=inductor_options,
    )
