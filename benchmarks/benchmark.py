import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import time
from typing import cast

import torch
import triton

from rdna3_fastmm.runtime import (
    PackedRight,
    PackedWeight,
    Rank7Plan,
    Rank48Plan,
    Rank49Plan,
    Rank343Plan,
    Workspace,
)
from rdna3_fastmm.external_mm import rdna3_rank49_dynamic_v1_out
from rdna3_fastmm.linear import rdna3_linear, rdna3_rank7_linear


LinearPlan = Rank7Plan | Rank48Plan | Rank49Plan
Plan = LinearPlan | Rank343Plan
ARTIFACTS = {
    "rank7": (
        Path("certificates/2x2x2_rank7_15add/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank7_2x2x2.py"),
        Rank7Plan,
    ),
    "rank49": (
        Path("certificates/4x4x4_rank49_159add/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank49_4x4x4.py"),
        Rank49Plan,
    ),
    "rank48": (
        Path("certificates/4x4x4_rank48_accurate/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank48_4x4x4.py"),
        Rank48Plan,
    ),
    "rank343": (
        Path("certificates/8x8x8_rank343_1661add/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank343_8x8x8.py"),
        Rank343Plan,
    ),
}
LINEAR_OPERATORS = {
    "rank7": rdna3_rank7_linear,
    "rank49": rdna3_linear,
}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
ENVIRONMENT_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_FILENAME",
    "TORCH_BLAS_PREFER_HIPBLASLT",
)


@dataclass(frozen=True)
class TimingSummary:
    raw_ms: list[float]
    median_ms: float
    first_quartile_ms: float
    third_quartile_ms: float


@dataclass(frozen=True)
class ErrorSummary:
    sample_count: int
    relative_l2_error: float
    maximum_absolute_error: float
    nonfinite_count: int


def parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or any(dimension < 1 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return cast(tuple[int, int, int], dimensions)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def quartile(values: list[float], numerator: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * numerator / 4
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> TimingSummary:
    return TimingSummary(
        raw_ms=values,
        median_ms=median(values),
        first_quartile_ms=quartile(values, 1),
        third_quartile_ms=quartile(values, 3),
    )


def elapsed_ms(operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure_operations(
    operations: dict[str, Callable[[], None]], warmups: int, rounds: int
) -> tuple[dict[str, TimingSummary], list[list[str]]]:
    names = list(operations)
    for warmup in range(warmups):
        offset = warmup % len(names)
        for name in names[offset:] + names[:offset]:
            operations[name]()
    torch.cuda.synchronize()
    raw_timings = {name: [] for name in names}
    order_schedule: list[list[str]] = []
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        order_schedule.append(order)
        for name in order:
            raw_timings[name].append(elapsed_ms(operations[name]))
    return (
        {name: summarize(values) for name, values in raw_timings.items()},
        order_schedule,
    )


def macroblock_tile_starts(
    dimension: int, tile_size: int, scheme_size: int
) -> list[int]:
    block_size = triton.cdiv(dimension, scheme_size)
    starts: list[int] = []
    for block_index in range(scheme_size):
        region_start = block_index * block_size
        region_end = min(region_start + block_size, dimension)
        if region_end - region_start < tile_size:
            raise ValueError("every rank-49 macroblock must contain one sample tile")
        starts.append(region_start + (region_end - region_start - tile_size) // 2)
    return starts


def validate_outputs(
    left: torch.Tensor,
    right: torch.Tensor,
    bias: torch.Tensor | None,
    outputs: dict[str, torch.Tensor],
    baseline_name: str,
    tile_size: int,
    scheme_size: int,
) -> dict[str, object]:
    row_starts = macroblock_tile_starts(left.shape[0], tile_size, scheme_size)
    column_starts = macroblock_tile_starts(right.shape[1], tile_size, scheme_size)
    left_slabs = [
        left.narrow(0, start, tile_size).contiguous().cpu().float()
        for start in row_starts
    ]
    right_slabs = [
        right.narrow(1, start, tile_size).contiguous().cpu().float()
        for start in column_starts
    ]
    squared_difference = {name: 0.0 for name in outputs}
    squared_reference = 0.0
    maximum_absolute_error = {name: 0.0 for name in outputs}
    nonfinite_count = {name: 0 for name in outputs}
    per_tile: list[dict[str, object]] = []
    for row_index, row_start in enumerate(row_starts):
        for column_index, column_start in enumerate(column_starts):
            reference = left_slabs[row_index] @ right_slabs[column_index]
            if bias is not None:
                reference += (
                    bias.narrow(0, column_start, tile_size).contiguous().cpu().float()
                )
            squared_reference += float(torch.sum(reference.double().square()).item())
            tile_algorithms: dict[str, dict[str, float | int]] = {}
            for name, output in outputs.items():
                candidate = (
                    output.narrow(0, row_start, tile_size)
                    .narrow(1, column_start, tile_size)
                    .contiguous()
                    .cpu()
                    .float()
                )
                difference = candidate - reference
                finite = torch.isfinite(candidate)
                tile_nonfinite_count = int((~finite).sum().item())
                safe_difference = torch.where(
                    finite, difference, torch.zeros_like(difference)
                )
                tile_squared_difference = float(
                    torch.sum(safe_difference.double().square()).item()
                )
                tile_maximum_absolute_error = float(safe_difference.abs().max().item())
                squared_difference[name] += tile_squared_difference
                maximum_absolute_error[name] = max(
                    maximum_absolute_error[name], tile_maximum_absolute_error
                )
                nonfinite_count[name] += tile_nonfinite_count
                tile_algorithms[name] = {
                    "relative_l2_error": math.sqrt(
                        tile_squared_difference
                        / float(torch.sum(reference.double().square()).item())
                    ),
                    "maximum_absolute_error": tile_maximum_absolute_error,
                    "nonfinite_count": tile_nonfinite_count,
                }
            per_tile.append(
                {
                    "macroblock": [row_index, column_index],
                    "row_start": row_start,
                    "column_start": column_start,
                    "algorithms": tile_algorithms,
                }
            )
    sample_count = len(row_starts) * len(column_starts) * tile_size**2
    aggregate = {
        name: ErrorSummary(
            sample_count=sample_count,
            relative_l2_error=math.sqrt(value / squared_reference),
            maximum_absolute_error=maximum_absolute_error[name],
            nonfinite_count=nonfinite_count[name],
        )
        for name, value in squared_difference.items()
    }
    baseline_error = aggregate[baseline_name].relative_l2_error
    return {
        "aggregate": {name: asdict(summary) for name, summary in aggregate.items()},
        "candidate_to_baseline_error_ratio": {
            name: summary.relative_l2_error / baseline_error
            for name, summary in aggregate.items()
            if name != baseline_name
        },
        "per_tile": per_tile,
    }


def measure_packing(
    pack: Callable[[], PackedRight | PackedWeight],
) -> tuple[PackedRight | PackedWeight, dict[str, float | int | None]]:
    warmup = pack()
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    packed = pack()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()
    return packed, {
        "device_ms": float(start.elapsed_time(end)),
        "wall_ms": (wall_end - wall_start) * 1000,
        "warmup_calls": 1,
    }


def packing_break_even_reuses(
    packing_ms: float,
    reference_ms: float,
    prepacked_ms: float,
) -> int | None:
    saved_ms = reference_ms - prepacked_ms
    return math.ceil(packing_ms / saved_ms) if saved_ms > 0 else None


def memory_requirement_bytes(
    shape: tuple[int, int, int],
    plan: Plan,
    modes: tuple[str, ...],
    has_bias: bool,
    retains_replaced_output: bool = False,
) -> int:
    rows, inner, columns = shape
    matrix_elements = rows * inner + inner * columns
    matrix_elements += (1 + len(modes)) * rows * columns
    if retains_replaced_output:
        matrix_elements += rows * columns
    workspace_bytes = 0
    if "dynamic" in modes:
        workspace_bytes += plan.workspace_bytes
    if "external" in modes:
        workspace_bytes += plan.workspace_bytes
    if "prepacked" in modes:
        workspace_bytes += plan.prepacked_workspace_bytes + plan.packed_right_bytes
    bias_elements = columns if has_bias else 0
    return (matrix_elements + bias_elements) * plan.dtype.itemsize + workspace_bytes


def algorithm_metrics(
    timings: dict[str, TimingSummary],
    shape: tuple[int, int, int],
    plan: Plan,
    operator: str,
    baseline_name: str,
    linear_implementation: str,
) -> dict[str, object]:
    rows, inner, columns = shape
    classical_flops = 2 * rows * inner * columns
    leaf_flops = (
        2
        * plan.rank
        * plan.shape.block_rows
        * plan.shape.block_inner
        * plan.shape.block_columns
    )
    baseline_ms = timings[baseline_name].median_ms
    metrics: dict[str, object] = {}
    for name, timing in timings.items():
        api = {
            "torch_mm": "torch.mm(out=...)",
            "torch_linear": "torch.nn.functional.linear(input, weight, bias)",
            "candidate_dynamic": f"{type(plan).__name__}.run(..., output=...)",
            "candidate_prepacked": (
                f"{type(plan).__name__}.run_linear_packed(...)"
                if operator == "linear"
                else f"{type(plan).__name__}.run_packed(..., output=...)"
            ),
            "candidate_external": "Inductor external_matmul out-callable",
        }[name]
        if name == "candidate_dynamic" and operator == "linear":
            api = (
                (
                    "rdna3_fastmm::linear_rank7 triton_op"
                    if isinstance(plan, Rank7Plan)
                    else "rdna3_fastmm::linear triton_op"
                )
                if linear_implementation == "triton-op"
                else f"{type(plan).__name__}.run_linear(...)"
            )
        entry: dict[str, object] = {
            "api": api,
            "timing": asdict(timing),
            "effective_classical_tflops": classical_flops / timing.median_ms / 1e9,
        }
        if name != baseline_name:
            entry["speedup"] = baseline_ms / timing.median_ms
            entry["executed_leaf_tflops"] = leaf_flops / timing.median_ms / 1e9
        metrics[name] = entry
    return metrics


def benchmark(
    algorithm: str,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    compute_dtype: torch.dtype,
    modes: tuple[str, ...],
    warmups: int,
    rounds: int,
    tile_size: int,
    seed: int,
    max_memory_fraction: float,
    allow_dirty: bool,
    allow_unrecommended: bool,
    operator: str = "mm",
    linear_bias: bool = True,
    linear_implementation: str = "triton-op",
) -> dict[str, object]:
    dirty = bool(git_output("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("refusing to benchmark a dirty tree without --allow-dirty")
    device = torch.device("cuda", torch.cuda.current_device())
    certificate_path, generated_module_path, plan_type = ARTIFACTS[algorithm]
    plan = plan_type(
        *shape,
        device=device,
        dtype=dtype,
        compute_dtype=compute_dtype,
    )
    if operator not in {"mm", "linear"}:
        raise ValueError("operator must be mm or linear")
    if operator == "linear" and any(
        mode not in {"dynamic", "prepacked"} for mode in modes
    ):
        raise ValueError("linear benchmarking supports dynamic and prepacked modes")
    if linear_implementation not in {"plan", "triton-op"}:
        raise ValueError("linear implementation must be plan or triton-op")
    if (
        operator == "linear"
        and linear_implementation == "triton-op"
        and algorithm not in LINEAR_OPERATORS
    ):
        raise ValueError("rank48 Linear benchmarking requires the plan implementation")
    if (
        operator == "linear"
        and "prepacked" in modes
        and linear_implementation != "plan"
    ):
        raise ValueError(
            "prepacked Linear benchmarking requires the plan implementation"
        )
    if operator == "linear":
        if not isinstance(plan, (Rank7Plan, Rank48Plan, Rank49Plan)):
            raise AssertionError("linear plan was not initialized")
        recommendations = {
            "dynamic": plan.is_linear_recommended(has_bias=linear_bias),
            "external": False,
            "prepacked": plan.is_linear_recommended(
                has_bias=linear_bias,
                prepacked_weight=True,
            ),
        }
    else:
        recommendations = {
            "dynamic": plan.is_recommended(),
            "external": (
                algorithm == "rank49"
                and dtype == torch.float16
                and compute_dtype == torch.float16
                and plan.is_recommended()
            ),
            "prepacked": plan.is_recommended(prepacked_right=True),
        }
    if not allow_unrecommended and any(not recommendations[mode] for mode in modes):
        raise RuntimeError(
            "shape or runtime is outside the measured dispatch whitelist"
        )
    required_bytes = memory_requirement_bytes(
        shape,
        plan,
        modes,
        operator == "linear" and linear_bias,
        operator == "linear",
    )
    free_before, _ = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs {required_bytes / 2**30:.2f} GiB, exceeding the "
            f"{free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    rows, inner, columns = shape
    left = torch.randn((rows, inner), device=device, dtype=dtype)
    if operator == "linear":
        weight = torch.randn((columns, inner), device=device, dtype=dtype)
        right = weight.T
        bias = (
            torch.randn((columns,), device=device, dtype=dtype) if linear_bias else None
        )
        baseline_name = "torch_linear"
    else:
        right = torch.randn((inner, columns), device=device, dtype=dtype)
        weight = None
        bias = None
        baseline_name = "torch_mm"
    outputs = {baseline_name: torch.empty((rows, columns), device=device, dtype=dtype)}
    if operator == "linear":
        if weight is None:
            raise AssertionError("linear tensors were not initialized")

        def run_torch_linear() -> None:
            outputs[baseline_name] = torch.nn.functional.linear(left, weight, bias)

        operations: dict[str, Callable[[], None]] = {baseline_name: run_torch_linear}
    else:
        operations = {
            baseline_name: lambda: torch.mm(left, right, out=outputs[baseline_name])
        }
    workspaces: dict[str, Workspace] = {}
    packing: dict[str, float | int | None] | None = None
    if "dynamic" in modes:
        outputs["candidate_dynamic"] = torch.empty(
            (rows, columns), device=device, dtype=dtype
        )
        if operator != "linear" or linear_implementation == "plan":
            workspaces["dynamic"] = plan.allocate_workspace(
                max_free_memory_fraction=1.0
            )

        def run_dynamic() -> None:
            if operator == "linear":
                if (
                    not isinstance(plan, (Rank7Plan, Rank48Plan, Rank49Plan))
                    or weight is None
                ):
                    raise AssertionError("linear plan was not initialized")
                if linear_implementation == "triton-op":
                    outputs["candidate_dynamic"] = LINEAR_OPERATORS[algorithm](
                        left, weight, bias
                    )
                else:
                    outputs["candidate_dynamic"] = plan.run_linear(
                        left,
                        weight,
                        workspaces["dynamic"],
                        bias,
                    )
            else:
                plan.run(
                    left,
                    right,
                    workspaces["dynamic"],
                    outputs["candidate_dynamic"],
                )

        operations["candidate_dynamic"] = run_dynamic
    if "external" in modes:
        if (
            algorithm != "rank49"
            or dtype != torch.float16
            or compute_dtype != torch.float16
        ):
            raise ValueError("the external_matmul candidate currently uses FP16 rank49")
        outputs["candidate_external"] = torch.empty(
            (rows, columns), device=device, dtype=dtype
        )

        def run_external() -> None:
            rdna3_rank49_dynamic_v1_out(left, right, out=outputs["candidate_external"])

        operations["candidate_external"] = run_external
    if "prepacked" in modes:
        outputs["candidate_prepacked"] = torch.empty(
            (rows, columns), device=device, dtype=dtype
        )
        if operator == "linear":
            if (
                not isinstance(plan, (Rank7Plan, Rank48Plan, Rank49Plan))
                or weight is None
            ):
                raise AssertionError("linear plan was not initialized")
            packed_operand, packing = measure_packing(
                lambda: plan.pack_weight(weight, max_free_memory_fraction=1.0)
            )
        else:
            packed_operand, packing = measure_packing(
                lambda: plan.pack_right(right, max_free_memory_fraction=1.0)
            )
        workspaces["prepacked"] = plan.allocate_workspace(
            max_free_memory_fraction=1.0, prepacked_right=True
        )

        def run_prepacked() -> None:
            if operator == "linear":
                if not isinstance(
                    plan, (Rank7Plan, Rank48Plan, Rank49Plan)
                ) or not isinstance(packed_operand, PackedWeight):
                    raise AssertionError("packed Linear weight was not initialized")
                outputs["candidate_prepacked"] = plan.run_linear_packed(
                    left,
                    packed_operand,
                    workspaces["prepacked"],
                    bias,
                )
            else:
                if not isinstance(packed_operand, PackedRight):
                    raise AssertionError("packed right matrix was not initialized")
                plan.run_packed(
                    left,
                    packed_operand,
                    workspaces["prepacked"],
                    outputs["candidate_prepacked"],
                )

        operations["candidate_prepacked"] = run_prepacked
    timings, order_schedule = measure_operations(operations, warmups, rounds)
    torch.cuda.synchronize()
    correctness = validate_outputs(
        left,
        right,
        bias,
        outputs,
        baseline_name,
        tile_size,
        plan.scheme_size,
    )
    algorithms = algorithm_metrics(
        timings,
        shape,
        plan,
        operator,
        baseline_name,
        linear_implementation,
    )
    if packing is not None:
        dynamic_name = "candidate_dynamic"
        prepacked_name = "candidate_prepacked"
        prepacked_ms = timings[prepacked_name].median_ms
        device_ms = cast(float, packing["device_ms"])
        if dynamic_name in timings:
            dynamic_break_even = packing_break_even_reuses(
                device_ms,
                timings[dynamic_name].median_ms,
                prepacked_ms,
            )
            packing["break_even_reuses"] = dynamic_break_even
            packing["break_even_reuses_vs_dynamic"] = dynamic_break_even
        packing["break_even_reuses_vs_baseline"] = packing_break_even_reuses(
            device_ms,
            timings[baseline_name].median_ms,
            prepacked_ms,
        )
        packing["first_call_ms"] = device_ms + prepacked_ms
    properties = torch.cuda.get_device_properties(device)
    free_after, total_memory = torch.cuda.mem_get_info(device)
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": dirty,
            "certificate_path": str(certificate_path),
            "certificate_sha256": file_sha256(certificate_path),
            "generated_module_path": str(generated_module_path),
            "generated_module_sha256": file_sha256(generated_module_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "algorithm": plan.algorithm,
            "linear_implementation": (
                linear_implementation if operator == "linear" else None
            ),
            "input_output_dtype": str(dtype).removeprefix("torch."),
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
        },
        "device": {
            "name": properties.name,
            "architecture": getattr(properties, "gcnArchName", ""),
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": total_memory,
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "protocol": {
            "seed": seed,
            "input_distribution": (
                "independent standard normal rounded to "
                f"{str(dtype).removeprefix('torch.')}"
            ),
            "input_output_dtype": str(dtype).removeprefix("torch."),
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "timing_method": "one operation per HIP event pair",
            "operator": operator,
            "weight_layout": "out_in" if operator == "linear" else "inner_columns",
            "bias": bias is not None,
            "correctness_reference": (
                "CPU FP32 tile matmul from original inputs"
                + (" with FP32 bias addition" if bias is not None else "")
            ),
            "tile_policy": (
                f"centered {tile_size}x{tile_size} tile in each of "
                f"{plan.scheme_size**2} output macroblocks"
            ),
        },
        "case": {
            "shape": list(shape),
            "operator": operator,
            "recommended": recommendations,
            "memory": {
                "estimated_required_bytes": required_bytes,
                "free_before_bytes": free_before,
                "free_after_bytes": free_after,
                "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "dynamic_workspace_bytes": plan.workspace_bytes,
                "prepacked_workspace_bytes": plan.prepacked_workspace_bytes,
                "packed_right_bytes": plan.packed_right_bytes,
                "packed_weight_bytes": (
                    plan.packed_weight_bytes
                    if isinstance(plan, (Rank7Plan, Rank48Plan, Rank49Plan))
                    else None
                ),
            },
            "algorithms": algorithms,
            "packing": packing,
            "correctness": correctness,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=tuple(ARTIFACTS), required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--operator", choices=("mm", "linear"), default="mm")
    parser.add_argument(
        "--linear-implementation",
        choices=("triton-op", "plan"),
        default="triton-op",
    )
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float16")
    parser.add_argument("--compute-dtype", choices=tuple(DTYPES))
    parser.add_argument(
        "--mode",
        choices=("dynamic", "prepacked", "external", "both"),
        default="both",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-unrecommended", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if arguments.tile_size < 1:
        parser.error("tile size must be positive")
    if arguments.no_bias and arguments.operator != "linear":
        parser.error("--no-bias requires --operator linear")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    modes = ("dynamic", "prepacked") if arguments.mode == "both" else (arguments.mode,)
    dtype = DTYPES[arguments.dtype]
    compute_dtype = (
        dtype if arguments.compute_dtype is None else DTYPES[arguments.compute_dtype]
    )
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.algorithm,
        arguments.shape,
        dtype,
        compute_dtype,
        modes,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.seed,
        arguments.max_memory_fraction,
        arguments.allow_dirty,
        arguments.allow_unrecommended,
        arguments.operator,
        not arguments.no_bias,
        arguments.linear_implementation,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)
        print(arguments.output)


if __name__ == "__main__":
    main()
