import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import cast, Literal

import torch
import triton

from benchmarks.benchmark import measure_operations, TimingSummary, validate_outputs
from rdna3_fastmm.linear import rdna3_linear
from rdna3_fastmm.runtime import Rank49Plan
from research.prototypes import generated_rank49_output_fusion


FusionStrategy = Literal["serial", "atomic"]


@dataclass(frozen=True)
class KernelConfig:
    block_rows: int
    block_columns: int
    block_inner: int
    warps: int


def parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return cast(tuple[int, int, int], dimensions)


def run_output_fusion(
    plan: Rank49Plan,
    bias: torch.Tensor | None,
    coefficients: torch.Tensor,
    config: KernelConfig,
    left_transformed: torch.Tensor,
    right_transformed: torch.Tensor,
    output: torch.Tensor,
    accumulator: torch.Tensor | None,
    strategy: FusionStrategy,
) -> None:
    shape = plan.shape
    if strategy == "serial":
        serial_grid = (
            triton.cdiv(shape.block_rows, config.block_rows),
            triton.cdiv(shape.block_columns, config.block_columns),
        )
        generated_rank49_output_fusion.fused_product_output_kernel[serial_grid](
            left_transformed,
            right_transformed,
            coefficients,
            output,
            output if bias is None else bias,
            shape.block_rows,
            shape.block_inner,
            shape.block_columns,
            shape.rows,
            shape.columns,
            config.block_rows,
            config.block_columns,
            config.block_inner,
            bias is not None,
            num_warps=config.warps,
            num_stages=1,
        )
        return
    if accumulator is None:
        raise ValueError("atomic fusion requires an accumulator")
    atomic_grid = (
        triton.cdiv(shape.block_rows, config.block_rows),
        triton.cdiv(shape.block_columns, config.block_columns),
        plan.rank,
    )
    generated_rank49_output_fusion.atomic_product_output_kernel[atomic_grid](
        left_transformed,
        right_transformed,
        coefficients,
        accumulator,
        shape.block_rows,
        shape.block_inner,
        shape.block_columns,
        shape.rows,
        shape.columns,
        config.block_rows,
        config.block_columns,
        config.block_inner,
        num_warps=config.warps,
        num_stages=1,
    )
    output_elements = shape.rows * shape.columns
    finalize_block_elements = 256
    finalize_grid = (triton.cdiv(output_elements, finalize_block_elements),)
    generated_rank49_output_fusion.atomic_output_finalize_kernel[finalize_grid](
        accumulator,
        output,
        output if bias is None else bias,
        output_elements,
        shape.columns,
        finalize_block_elements,
        bias is not None,
        True,
        num_warps=2,
        num_stages=1,
    )


def benchmark_stages(
    plan: Rank49Plan,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    coefficients: torch.Tensor,
    config: KernelConfig,
    strategy: FusionStrategy,
    warmups: int,
    rounds: int,
) -> tuple[dict[str, TimingSummary], list[list[str]], int]:
    rows, _, columns = plan.shape.dimensions
    left_transformed = torch.empty(
        (plan.rank, plan.shape.block_rows, plan.shape.block_inner),
        device=plan.device,
        dtype=torch.float16,
    )
    right_transformed = torch.empty(
        (plan.rank, plan.shape.block_inner, plan.shape.block_columns),
        device=plan.device,
        dtype=torch.float16,
    )
    products = torch.empty(
        (plan.rank, plan.shape.block_rows, plan.shape.block_columns),
        device=plan.device,
        dtype=torch.float16,
    )
    current_output = torch.empty(
        (rows, columns), device=plan.device, dtype=torch.bfloat16
    )
    fused_output = torch.empty(
        (rows, columns), device=plan.device, dtype=torch.bfloat16
    )
    accumulator = (
        torch.zeros((rows, columns), device=plan.device, dtype=torch.float32)
        if strategy == "atomic"
        else None
    )
    plan._transform_left(input_tensor, left_transformed)
    plan._transform_weight(weight, right_transformed)

    def run_current_stage() -> None:
        torch.bmm(left_transformed, right_transformed, out=products)
        if bias is None:
            plan._reconstruct(products, current_output)
        else:
            plan._reconstruct_bias(products, bias, current_output)

    def run_candidate_output() -> None:
        run_output_fusion(
            plan,
            bias,
            coefficients,
            config,
            left_transformed,
            right_transformed,
            fused_output,
            accumulator,
            strategy,
        )

    timings, order = measure_operations(
        {
            "bmm_plus_reconstruction": run_current_stage,
            f"{strategy}_product_output": run_candidate_output,
        },
        warmups,
        rounds,
    )
    workspace_bytes = (
        left_transformed.numel() + right_transformed.numel()
    ) * torch.float16.itemsize
    if accumulator is not None:
        workspace_bytes += accumulator.numel() * accumulator.element_size()
    return timings, order, workspace_bytes


def benchmark(
    shape: tuple[int, int, int],
    has_bias: bool,
    config: KernelConfig,
    strategy: FusionStrategy,
    warmups: int,
    rounds: int,
    tile_size: int,
    max_memory_fraction: float,
) -> dict[str, object]:
    device = torch.device("cuda")
    plan = Rank49Plan(
        *shape,
        device=device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    rows, inner, columns = shape
    source_bytes = (
        rows * inner + columns * inner + (columns if has_bias else 0)
    ) * torch.bfloat16.itemsize
    output_bytes = rows * columns * torch.bfloat16.itemsize
    output_buffer_count = 4 if strategy == "atomic" else 3
    required_bytes = (
        source_bytes + plan.workspace_bytes + output_buffer_count * output_bytes
    )
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs {required_bytes / 2**30:.2f} GiB, exceeding the "
            f"{free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = torch.randn((rows, inner), device=device, dtype=torch.bfloat16)
    weight = torch.randn((columns, inner), device=device, dtype=torch.bfloat16)
    bias = (
        torch.randn((columns,), device=device, dtype=torch.bfloat16)
        if has_bias
        else None
    )
    coefficients = torch.tensor(
        generated_rank49_output_fusion.FUSED_OUTPUT_COEFFICIENTS,
        device=device,
        dtype=torch.int8,
    )
    stage_timings, stage_order, candidate_workspace_bytes = benchmark_stages(
        plan,
        input_tensor,
        weight,
        bias,
        coefficients,
        config,
        strategy,
        warmups,
        rounds,
    )
    torch.cuda.empty_cache()
    outputs: dict[str, torch.Tensor] = {}

    def run_torch_linear() -> None:
        outputs["torch_linear"] = torch.nn.functional.linear(input_tensor, weight, bias)

    def run_current_operator() -> None:
        outputs["current_operator"] = rdna3_linear(input_tensor, weight, bias)

    def run_candidate_operator() -> None:
        candidate_left = torch.empty(
            (plan.rank, plan.shape.block_rows, plan.shape.block_inner),
            device=device,
            dtype=torch.float16,
        )
        candidate_right = torch.empty(
            (plan.rank, plan.shape.block_inner, plan.shape.block_columns),
            device=device,
            dtype=torch.float16,
        )
        candidate_output = torch.empty(
            (rows, columns), device=device, dtype=torch.bfloat16
        )
        candidate_accumulator = (
            torch.zeros((rows, columns), device=device, dtype=torch.float32)
            if strategy == "atomic"
            else None
        )
        plan._transform_left(input_tensor, candidate_left)
        plan._transform_weight(weight, candidate_right)
        run_output_fusion(
            plan,
            bias,
            coefficients,
            config,
            candidate_left,
            candidate_right,
            candidate_output,
            candidate_accumulator,
            strategy,
        )
        outputs["candidate_operator"] = candidate_output

    full_timings, full_order = measure_operations(
        {
            "torch_linear": run_torch_linear,
            "current_operator": run_current_operator,
            "candidate_operator": run_candidate_operator,
        },
        warmups,
        rounds,
    )
    correctness = validate_outputs(
        input_tensor,
        weight.T,
        bias,
        outputs,
        "torch_linear",
        tile_size,
        plan.scheme_size,
    )
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    return {
        "shape": list(shape),
        "bias": has_bias,
        "strategy": strategy,
        "config": asdict(config),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "current_workspace_bytes": plan.workspace_bytes,
            "candidate_workspace_bytes": candidate_workspace_bytes,
            "eliminated_product_bytes": (
                plan.rank * plan.shape.block_rows * plan.shape.block_columns
            )
            * torch.float16.itemsize,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "total_bytes": total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "stage_timings": {
            name: asdict(timing) for name, timing in stage_timings.items()
        },
        "stage_order": stage_order,
        "full_timings": {name: asdict(timing) for name, timing in full_timings.items()},
        "full_order": full_order,
        "speedup": {
            "candidate_over_torch": (
                full_timings["torch_linear"].median_ms
                / full_timings["candidate_operator"].median_ms
            ),
            "candidate_over_current": (
                full_timings["current_operator"].median_ms
                / full_timings["candidate_operator"].median_ms
            ),
            "candidate_stage_over_current_stage": (
                stage_timings["bmm_plus_reconstruction"].median_ms
                / stage_timings[f"{strategy}_product_output"].median_ms
            ),
        },
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--strategy", choices=("serial", "atomic"), default="serial")
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--block-columns", type=int, default=16)
    parser.add_argument("--block-inner", type=int, default=32)
    parser.add_argument("--warps", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if arguments.tile_size < 1:
        parser.error("tile size must be positive")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    config = KernelConfig(
        arguments.block_rows,
        arguments.block_columns,
        arguments.block_inner,
        arguments.warps,
    )
    if any(
        value < 16 or value & (value - 1)
        for value in (
            config.block_rows,
            config.block_columns,
            config.block_inner,
        )
    ):
        parser.error("kernel block sizes must be powers of two and at least 16")
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.shape,
        arguments.bias,
        config,
        arguments.strategy,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.max_memory_fraction,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)
        print(arguments.output)


if __name__ == "__main__":
    main()
