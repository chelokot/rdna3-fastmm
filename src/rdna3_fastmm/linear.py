from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.library import triton_op, wrap_triton
import triton

from rdna3_fastmm.generated import rank7_2x2x2, rank49_4x4x4
from rdna3_fastmm.runtime import Rank49Plan, WeightTransformConfig


LINEAR_TARGETS = frozenset({torch._C._nn.linear, torch.ops.aten.linear.default})


@dataclass(frozen=True)
class _LinearKernelConfig:
    scheme_size: int
    rank: int
    transform_elements: int
    transform_warps: int
    weight: WeightTransformConfig


_RANK7_DEFAULT_CONFIG = _LinearKernelConfig(
    scheme_size=2,
    rank=7,
    transform_elements=256,
    transform_warps=2,
    weight=WeightTransformConfig(8, 512, 8),
)
_RANK7_CONFIGS = {
    (3_072, 12_288): _LinearKernelConfig(
        scheme_size=2,
        rank=7,
        transform_elements=256,
        transform_warps=2,
        weight=WeightTransformConfig(8, 256, 4),
    ),
    (4_096, 16_384): _LinearKernelConfig(
        scheme_size=2,
        rank=7,
        transform_elements=256,
        transform_warps=2,
        weight=WeightTransformConfig(8, 256, 4),
    ),
    (4_608, 12_288): _LinearKernelConfig(
        scheme_size=2,
        rank=7,
        transform_elements=512,
        transform_warps=4,
        weight=WeightTransformConfig(8, 512, 8),
    ),
    (12_288, 4_608): _LinearKernelConfig(
        scheme_size=2,
        rank=7,
        transform_elements=1_024,
        transform_warps=4,
        weight=WeightTransformConfig(8, 256, 8),
    ),
}


def _rank7_kernel_config(inner: int, columns: int) -> _LinearKernelConfig:
    return _RANK7_CONFIGS.get((inner, columns), _RANK7_DEFAULT_CONFIG)


def _rank49_kernel_config(inner: int, columns: int) -> _LinearKernelConfig:
    return _LinearKernelConfig(
        scheme_size=4,
        rank=49,
        transform_elements=256,
        transform_warps=2,
        weight=Rank49Plan.weight_transform_config(inner, columns),
    )


def _create_linear_operator(
    name: str,
    generated: ModuleType,
    resolve_config: Callable[[int, int], _LinearKernelConfig],
) -> Callable[..., torch.Tensor]:
    left_transform_kernel = generated.left_transform_kernel
    right_transform_weight_kernel = generated.right_transform_weight_kernel
    output_transform_kernel = generated.output_transform_kernel
    output_transform_bias_kernel = generated.output_transform_bias_kernel

    @triton_op(name, mutates_args={})
    def operator(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inner = input_tensor.shape[-1]
        columns = weight.shape[0]
        config = resolve_config(inner, columns)
        flattened_input = input_tensor.reshape(-1, inner)
        rows = flattened_input.shape[0]
        block_rows = triton.cdiv(rows, config.scheme_size)
        block_inner = triton.cdiv(inner, config.scheme_size)
        block_columns = triton.cdiv(columns, config.scheme_size)
        left_transformed = torch.empty(
            (config.rank, block_rows, block_inner),
            device=input_tensor.device,
            dtype=torch.float16,
        )
        right_transformed = torch.empty(
            (config.rank, block_inner, block_columns),
            device=input_tensor.device,
            dtype=torch.float16,
        )
        transform_grid = (
            triton.cdiv(block_rows * block_inner, config.transform_elements),
        )
        wrap_triton(left_transform_kernel)[transform_grid](
            flattened_input,
            left_transformed,
            rows,
            inner,
            block_rows,
            block_inner,
            block_rows,
            block_inner,
            config.transform_elements,
            num_warps=config.transform_warps,
            num_stages=1,
        )
        weight_grid = (
            triton.cdiv(block_inner, config.weight.block_rows),
            triton.cdiv(block_columns, config.weight.block_columns),
        )
        wrap_triton(right_transform_weight_kernel)[weight_grid](
            weight,
            right_transformed,
            columns,
            inner,
            block_inner,
            block_columns,
            config.weight.block_rows,
            config.weight.block_columns,
            num_warps=config.weight.warps,
            num_stages=1,
        )
        products = torch.bmm(left_transformed, right_transformed)
        flattened_output = torch.empty(
            (rows, columns), device=input_tensor.device, dtype=input_tensor.dtype
        )
        output_grid = (
            triton.cdiv(block_rows * block_columns, config.transform_elements),
        )
        if bias is None:
            wrap_triton(output_transform_kernel)[output_grid](
                products,
                flattened_output,
                block_rows,
                block_columns,
                rows,
                columns,
                block_rows,
                block_columns,
                config.transform_elements,
                num_warps=config.transform_warps,
                num_stages=1,
            )
        else:
            wrap_triton(output_transform_bias_kernel)[output_grid](
                products,
                flattened_output,
                bias,
                block_rows,
                block_columns,
                rows,
                columns,
                block_rows,
                block_columns,
                config.transform_elements,
                num_warps=config.transform_warps,
                num_stages=1,
            )
        return flattened_output.reshape(*input_tensor.shape[:-1], columns)

    return operator


rdna3_linear = _create_linear_operator(
    "rdna3_fastmm::linear", rank49_4x4x4, _rank49_kernel_config
)
rdna3_rank7_linear = _create_linear_operator(
    "rdna3_fastmm::linear_rank7", rank7_2x2x2, _rank7_kernel_config
)


RDNA3_LINEAR_OP = torch.ops.rdna3_fastmm.linear.default
RDNA3_RANK7_LINEAR_OP = torch.ops.rdna3_fastmm.linear_rank7.default


def _node_tensor(node: Node) -> torch.Tensor | None:
    value = node.meta.get("val", node.meta.get("example_value"))
    return value if isinstance(value, torch.Tensor) else None


def _eligible_linear_node(node: Node) -> bool:
    if node.op != "call_function" or node.target not in LINEAR_TARGETS:
        return False
    if node.kwargs or len(node.args) not in {2, 3}:
        return False
    input_node, weight_node = node.args[:2]
    bias_node = node.args[2] if len(node.args) == 3 else None
    if not isinstance(input_node, Node) or not isinstance(weight_node, Node):
        return False
    input_tensor = _node_tensor(input_node)
    weight = _node_tensor(weight_node)
    if input_tensor is None or weight is None:
        return False
    if input_tensor.ndim < 2 or weight.ndim != 2:
        return False
    if input_tensor.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if input_tensor.device.type != "cuda" or weight.device != input_tensor.device:
        return False
    if not input_tensor.is_contiguous() or not weight.is_contiguous():
        return False
    inner = input_tensor.shape[-1]
    columns, weight_inner = weight.shape
    if inner != weight_inner:
        return False
    dimensions = (*input_tensor.shape[:-1], inner, columns)
    if not all(isinstance(dimension, int) for dimension in dimensions):
        return False
    has_bias = bias_node is not None
    if isinstance(bias_node, Node):
        bias = _node_tensor(bias_node)
        if (
            bias is None
            or bias.shape != (columns,)
            or bias.dtype != torch.bfloat16
            or bias.device != input_tensor.device
            or not bias.is_contiguous()
        ):
            return False
    elif bias_node is not None:
        return False
    rows = input_tensor.numel() // inner
    return Rank49Plan.has_measured_linear_win((rows, inner, columns), has_bias=has_bias)


def rewrite_eligible_linears(
    graph_module: GraphModule, example_inputs: list[torch.Tensor]
) -> int:
    linear_nodes = tuple(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target in LINEAR_TARGETS
    )
    if any(
        isinstance(argument, Node) and _node_tensor(argument) is None
        for node in linear_nodes
        for argument in node.args
    ):
        FakeTensorProp(graph_module).propagate(*example_inputs)
    rewritten = 0
    for node in linear_nodes:
        if not _eligible_linear_node(node):
            continue
        input_node, weight_node = node.args[:2]
        bias_node = node.args[2] if len(node.args) == 3 else None
        with graph_module.graph.inserting_before(node):
            replacement = graph_module.graph.call_function(
                RDNA3_LINEAR_OP,
                args=(input_node, weight_node, bias_node),
            )
        replacement.meta = node.meta.copy()
        node.replace_all_uses_with(replacement)
        graph_module.graph.erase_node(node)
        rewritten += 1
    if rewritten:
        graph_module.graph.lint()
        graph_module.recompile()
    return rewritten
