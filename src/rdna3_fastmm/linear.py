import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.library import triton_op, wrap_triton
import triton

from rdna3_fastmm.generated import rank49_4x4x4
from rdna3_fastmm.runtime import Rank49Plan


LINEAR_TARGETS = frozenset({torch._C._nn.linear, torch.ops.aten.linear.default})


@triton_op("rdna3_fastmm::linear", mutates_args={})
def rdna3_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    inner = input_tensor.shape[-1]
    columns = weight.shape[0]
    flattened_input = input_tensor.reshape(-1, inner)
    rows = flattened_input.shape[0]
    block_rows = triton.cdiv(rows, 4)
    block_inner = triton.cdiv(inner, 4)
    block_columns = triton.cdiv(columns, 4)
    left_transformed = torch.empty(
        (49, block_rows, block_inner),
        device=input_tensor.device,
        dtype=torch.float16,
    )
    right_transformed = torch.empty(
        (49, block_inner, block_columns),
        device=input_tensor.device,
        dtype=torch.float16,
    )
    left_block_elements = 256
    left_grid = (triton.cdiv(block_rows * block_inner, left_block_elements),)
    wrap_triton(rank49_4x4x4.left_transform_kernel)[left_grid](
        flattened_input,
        left_transformed,
        rows,
        inner,
        block_rows,
        block_inner,
        block_rows,
        block_inner,
        left_block_elements,
        num_warps=2,
        num_stages=1,
    )
    weight_config = Rank49Plan.weight_transform_config(inner, columns)
    weight_grid = (
        triton.cdiv(block_inner, weight_config.block_rows),
        triton.cdiv(block_columns, weight_config.block_columns),
    )
    wrap_triton(rank49_4x4x4.right_transform_weight_kernel)[weight_grid](
        weight,
        right_transformed,
        columns,
        inner,
        block_inner,
        block_columns,
        weight_config.block_rows,
        weight_config.block_columns,
        num_warps=weight_config.warps,
        num_stages=1,
    )
    products = torch.bmm(left_transformed, right_transformed)
    flattened_output = torch.empty(
        (rows, columns), device=input_tensor.device, dtype=input_tensor.dtype
    )
    output_block_elements = 256
    output_grid = (triton.cdiv(block_rows * block_columns, output_block_elements),)
    output_kernel = (
        rank49_4x4x4.output_transform_kernel
        if bias is None
        else rank49_4x4x4.output_transform_bias_kernel
    )
    output_arguments = [products, flattened_output]
    if bias is not None:
        output_arguments.append(bias)
    wrap_triton(output_kernel)[output_grid](
        *output_arguments,
        block_rows,
        block_columns,
        rows,
        columns,
        block_rows,
        block_columns,
        output_block_elements,
        num_warps=2,
        num_stages=1,
    )
    return flattened_output.reshape(*input_tensor.shape[:-1], columns)


RDNA3_LINEAR_OP = torch.ops.rdna3_fastmm.linear.default


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
