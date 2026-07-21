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

from rdna3_fastmm.runtime import PackedRight, Rank49Plan, Rank49Workspace


CERTIFICATE_PATH = Path("certificates/4x4x4_rank49_159add/certificate.json")
GENERATED_MODULE_PATH = Path("src/rdna3_fastmm/generated/rank49_4x4x4.py")
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


def macroblock_tile_starts(dimension: int, tile_size: int) -> list[int]:
    block_size = triton.cdiv(dimension, 4)
    starts: list[int] = []
    for block_index in range(4):
        region_start = block_index * block_size
        region_end = min(region_start + block_size, dimension)
        if region_end - region_start < tile_size:
            raise ValueError("every rank-49 macroblock must contain one sample tile")
        starts.append(region_start + (region_end - region_start - tile_size) // 2)
    return starts


def validate_outputs(
    left: torch.Tensor,
    right: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    tile_size: int,
) -> dict[str, object]:
    row_starts = macroblock_tile_starts(left.shape[0], tile_size)
    column_starts = macroblock_tile_starts(right.shape[1], tile_size)
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
    baseline_error = aggregate["torch_mm"].relative_l2_error
    return {
        "aggregate": {name: asdict(summary) for name, summary in aggregate.items()},
        "candidate_to_baseline_error_ratio": {
            name: summary.relative_l2_error / baseline_error
            for name, summary in aggregate.items()
            if name != "torch_mm"
        },
        "per_tile": per_tile,
    }


def measure_packing(
    plan: Rank49Plan, right: torch.Tensor
) -> tuple[PackedRight, dict[str, float | int | None]]:
    warmup = plan.pack_right(right, max_free_memory_fraction=1.0)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    packed = plan.pack_right(right, max_free_memory_fraction=1.0)
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()
    return packed, {
        "device_ms": float(start.elapsed_time(end)),
        "wall_ms": (wall_end - wall_start) * 1000,
        "warmup_calls": 1,
    }


def memory_requirement_bytes(
    shape: tuple[int, int, int], plan: Rank49Plan, modes: tuple[str, ...]
) -> int:
    rows, inner, columns = shape
    matrix_elements = rows * inner + inner * columns
    matrix_elements += (1 + len(modes)) * rows * columns
    workspace_bytes = 0
    if "dynamic" in modes:
        workspace_bytes += plan.workspace_bytes
    if "prepacked" in modes:
        workspace_bytes += plan.prepacked_workspace_bytes + plan.packed_right_bytes
    return matrix_elements * 2 + workspace_bytes


def algorithm_metrics(
    timings: dict[str, TimingSummary],
    shape: tuple[int, int, int],
    plan: Rank49Plan,
) -> dict[str, object]:
    rows, inner, columns = shape
    classical_flops = 2 * rows * inner * columns
    leaf_flops = (
        2
        * 49
        * plan.shape.block_rows
        * plan.shape.block_inner
        * plan.shape.block_columns
    )
    baseline_ms = timings["torch_mm"].median_ms
    metrics: dict[str, object] = {}
    for name, timing in timings.items():
        api = {
            "torch_mm": "torch.mm(out=...)",
            "rank49_dynamic": "Rank49Plan.run(..., output=...)",
            "rank49_prepacked": "Rank49Plan.run_packed(..., output=...)",
        }[name]
        entry: dict[str, object] = {
            "api": api,
            "timing": asdict(timing),
            "effective_classical_tflops": classical_flops / timing.median_ms / 1e9,
        }
        if name != "torch_mm":
            entry["speedup"] = baseline_ms / timing.median_ms
            entry["executed_leaf_tflops"] = leaf_flops / timing.median_ms / 1e9
        metrics[name] = entry
    return metrics


def benchmark(
    shape: tuple[int, int, int],
    modes: tuple[str, ...],
    warmups: int,
    rounds: int,
    tile_size: int,
    seed: int,
    max_memory_fraction: float,
    allow_dirty: bool,
    allow_unrecommended: bool,
) -> dict[str, object]:
    dirty = bool(git_output("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("refusing to benchmark a dirty tree without --allow-dirty")
    device = torch.device("cuda", torch.cuda.current_device())
    plan = Rank49Plan(*shape, device=device)
    recommendations = {
        "dynamic": plan.is_recommended(),
        "prepacked": plan.is_recommended(prepacked_right=True),
    }
    if not allow_unrecommended and any(not recommendations[mode] for mode in modes):
        raise RuntimeError(
            "shape or runtime is outside the measured dispatch whitelist"
        )
    required_bytes = memory_requirement_bytes(shape, plan, modes)
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
    left = torch.randn((rows, inner), device=device, dtype=torch.float16)
    right = torch.randn((inner, columns), device=device, dtype=torch.float16)
    outputs = {
        "torch_mm": torch.empty((rows, columns), device=device, dtype=torch.float16)
    }
    operations: dict[str, Callable[[], None]] = {
        "torch_mm": lambda: torch.mm(left, right, out=outputs["torch_mm"])
    }
    workspaces: dict[str, Rank49Workspace] = {}
    packing: dict[str, float | int | None] | None = None
    if "dynamic" in modes:
        outputs["rank49_dynamic"] = torch.empty(
            (rows, columns), device=device, dtype=torch.float16
        )
        workspaces["dynamic"] = plan.allocate_workspace(max_free_memory_fraction=1.0)

        def run_dynamic() -> None:
            plan.run(
                left,
                right,
                workspaces["dynamic"],
                outputs["rank49_dynamic"],
            )

        operations["rank49_dynamic"] = run_dynamic
    if "prepacked" in modes:
        outputs["rank49_prepacked"] = torch.empty(
            (rows, columns), device=device, dtype=torch.float16
        )
        packed_right, packing = measure_packing(plan, right)
        workspaces["prepacked"] = plan.allocate_workspace(
            max_free_memory_fraction=1.0, prepacked_right=True
        )

        def run_prepacked() -> None:
            plan.run_packed(
                left,
                packed_right,
                workspaces["prepacked"],
                outputs["rank49_prepacked"],
            )

        operations["rank49_prepacked"] = run_prepacked
    timings, order_schedule = measure_operations(operations, warmups, rounds)
    torch.cuda.synchronize()
    correctness = validate_outputs(left, right, outputs, tile_size)
    algorithms = algorithm_metrics(timings, shape, plan)
    if packing is not None:
        dynamic_name = "rank49_dynamic"
        prepacked_name = "rank49_prepacked"
        if dynamic_name in timings:
            saved_ms = (
                timings[dynamic_name].median_ms - timings[prepacked_name].median_ms
            )
            device_ms = cast(float, packing["device_ms"])
            packing["break_even_reuses"] = (
                math.ceil(device_ms / saved_ms) if saved_ms > 0 else None
            )
        packing["first_call_ms"] = (
            cast(float, packing["device_ms"]) + timings[prepacked_name].median_ms
        )
    properties = torch.cuda.get_device_properties(device)
    free_after, total_memory = torch.cuda.mem_get_info(device)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": dirty,
            "certificate_path": str(CERTIFICATE_PATH),
            "certificate_sha256": file_sha256(CERTIFICATE_PATH),
            "generated_module_path": str(GENERATED_MODULE_PATH),
            "generated_module_sha256": file_sha256(GENERATED_MODULE_PATH),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "kernel": {"block_elements": 256, "num_warps": 2, "num_stages": 1},
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
            "input_distribution": "independent standard normal rounded to FP16",
            "dtype": "float16",
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "timing_method": "one operation per HIP event pair",
            "correctness_reference": "CPU FP32 tile matmul from original FP16 inputs",
            "tile_policy": (
                f"centered {tile_size}x{tile_size} tile in each of 16 rank-49 "
                "output macroblocks"
            ),
        },
        "case": {
            "shape": list(shape),
            "recommended": recommendations,
            "memory": {
                "estimated_required_bytes": required_bytes,
                "free_before_bytes": free_before,
                "free_after_bytes": free_after,
                "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "dynamic_workspace_bytes": plan.workspace_bytes,
                "prepacked_workspace_bytes": plan.prepacked_workspace_bytes,
                "packed_right_bytes": plan.packed_right_bytes,
            },
            "algorithms": algorithms,
            "packing": packing,
            "correctness": correctness,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument(
        "--mode", choices=("dynamic", "prepacked", "both"), default="both"
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
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    modes = ("dynamic", "prepacked") if arguments.mode == "both" else (arguments.mode,)
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.shape,
        modes,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.seed,
        arguments.max_memory_fraction,
        arguments.allow_dirty,
        arguments.allow_unrecommended,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)
        print(arguments.output)


if __name__ == "__main__":
    main()
