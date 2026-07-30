import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
import json
from pathlib import Path
from types import ModuleType

import torch
import triton

from benchmarks.benchmark import git_output, measure_operations, parse_shape
from rdna3_fastmm.generated import rank7_2x2x2
from rdna3_fastmm.runtime import ElementTransformConfig, WeightTransformConfig
from research.prototypes import (
    generated_rank29_3x3x4,
    generated_rank105_4x6x6,
)
from research.prototypes.benchmark_rectangular_products import (
    LIBRARY_RESERVE_BYTES,
    block_shape,
    workspace_elements,
)


@dataclass(frozen=True)
class CandidateSpec:
    generated: ModuleType
    certificate_path: Path
    element_configs: tuple[ElementTransformConfig, ...]
    weight_configs: tuple[WeightTransformConfig, ...]


CANDIDATES = {
    "rank29_3x3x4": CandidateSpec(
        generated_rank29_3x3x4,
        Path("certificates/research/3x3x4_rank29_92add/certificate.json"),
        (
            ElementTransformConfig(256, 2),
            ElementTransformConfig(512, 4),
        ),
        (WeightTransformConfig(8, 512, 8),),
    ),
    "rank105_4x6x6": CandidateSpec(
        generated_rank105_4x6x6,
        Path("certificates/research/4x6x6_rank105_430add/certificate.json"),
        (
            ElementTransformConfig(128, 4),
            ElementTransformConfig(256, 8),
        ),
        (
            WeightTransformConfig(2, 128, 8),
            WeightTransformConfig(4, 64, 8),
        ),
    ),
}
RANK7_ELEMENT_CONFIG = ElementTransformConfig(512, 4)
RANK7_WEIGHT_CONFIG = WeightTransformConfig(8, 512, 8)


def launch_left_transform(
    generated: ModuleType,
    source: torch.Tensor,
    output: torch.Tensor,
    shape: tuple[int, int, int],
    blocks: tuple[int, int, int],
    config: ElementTransformConfig,
) -> None:
    grid = (triton.cdiv(blocks[0] * blocks[1], config.block_elements),)
    generated.left_transform_kernel[grid](
        source,
        output,
        shape[0],
        shape[1],
        blocks[0],
        blocks[1],
        blocks[0],
        blocks[1],
        config.block_elements,
        num_warps=config.warps,
        num_stages=1,
    )


def launch_weight_transform(
    generated: ModuleType,
    source: torch.Tensor,
    output: torch.Tensor,
    shape: tuple[int, int, int],
    blocks: tuple[int, int, int],
    config: WeightTransformConfig,
) -> None:
    grid = (
        triton.cdiv(blocks[1], config.block_rows),
        triton.cdiv(blocks[2], config.block_columns),
    )
    generated.right_transform_weight_kernel[grid](
        source,
        output,
        shape[2],
        shape[1],
        blocks[1],
        blocks[2],
        config.block_rows,
        config.block_columns,
        num_warps=config.warps,
        num_stages=1,
    )


def launch_output_transform(
    generated: ModuleType,
    source: torch.Tensor,
    output: torch.Tensor,
    shape: tuple[int, int, int],
    blocks: tuple[int, int, int],
    config: ElementTransformConfig,
) -> None:
    grid = (triton.cdiv(blocks[0] * blocks[2], config.block_elements),)
    generated.output_transform_kernel[grid](
        source,
        output,
        blocks[0],
        blocks[2],
        shape[0],
        shape[2],
        blocks[0],
        blocks[2],
        config.block_elements,
        num_warps=config.warps,
        num_stages=1,
    )


def element_key(stage: str, config: ElementTransformConfig) -> str:
    return f"candidate_{stage}_{config.block_elements}x{config.warps}"


def weight_key(config: WeightTransformConfig) -> str:
    return f"candidate_weight_{config.block_rows}x{config.block_columns}x{config.warps}"


def benchmark(
    candidate_name: str,
    shape: tuple[int, int, int],
    warmups: int,
    rounds: int,
    max_memory_fraction: float,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rectangular research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    candidate = CANDIDATES[candidate_name]
    candidate_dimensions = candidate.generated.DIMENSIONS
    candidate_rank = candidate.generated.RANK
    candidate_blocks = block_shape(shape, candidate_dimensions)
    rank7_dimensions = rank7_2x2x2.DIMENSIONS
    rank7_rank = rank7_2x2x2.RANK
    rank7_blocks = block_shape(shape, rank7_dimensions)
    source_elements = shape[0] * shape[1] + shape[2] * shape[1]
    output_elements = 2 * shape[0] * shape[2]
    required_bytes = (
        (source_elements + output_elements) * torch.bfloat16.itemsize
        + (
            workspace_elements(shape, candidate_dimensions, candidate_rank)
            + workspace_elements(shape, rank7_dimensions, rank7_rank)
        )
        * torch.float16.itemsize
        + LIBRARY_RESERVE_BYTES
    )
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs an estimated {required_bytes / 2**30:.2f} GiB, "
            f"exceeding the {free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = torch.full(
        (shape[0], shape[1]), 0.5, device=device, dtype=torch.bfloat16
    )
    weight = torch.full((shape[2], shape[1]), 0.5, device=device, dtype=torch.bfloat16)
    candidate_left = torch.empty(
        (candidate_rank, candidate_blocks[0], candidate_blocks[1]),
        device=device,
        dtype=torch.float16,
    )
    candidate_right = torch.empty(
        (candidate_rank, candidate_blocks[1], candidate_blocks[2]),
        device=device,
        dtype=torch.float16,
    )
    candidate_products = torch.full(
        (candidate_rank, candidate_blocks[0], candidate_blocks[2]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    candidate_output = torch.empty(
        (shape[0], shape[2]), device=device, dtype=torch.bfloat16
    )
    rank7_left = torch.empty(
        (rank7_rank, rank7_blocks[0], rank7_blocks[1]),
        device=device,
        dtype=torch.float16,
    )
    rank7_right = torch.empty(
        (rank7_rank, rank7_blocks[1], rank7_blocks[2]),
        device=device,
        dtype=torch.float16,
    )
    rank7_products = torch.full(
        (rank7_rank, rank7_blocks[0], rank7_blocks[2]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    rank7_output = torch.empty(
        (shape[0], shape[2]), device=device, dtype=torch.bfloat16
    )
    operations: dict[str, Callable[[], None]] = {}
    for config in candidate.element_configs:
        operations[element_key("left", config)] = partial(
            launch_left_transform,
            candidate.generated,
            input_tensor,
            candidate_left,
            shape,
            candidate_blocks,
            config,
        )
        operations[element_key("output", config)] = partial(
            launch_output_transform,
            candidate.generated,
            candidate_products,
            candidate_output,
            shape,
            candidate_blocks,
            config,
        )
    for config in candidate.weight_configs:
        operations[weight_key(config)] = partial(
            launch_weight_transform,
            candidate.generated,
            weight,
            candidate_right,
            shape,
            candidate_blocks,
            config,
        )
    operations["rank7_left_512x4"] = partial(
        launch_left_transform,
        rank7_2x2x2,
        input_tensor,
        rank7_left,
        shape,
        rank7_blocks,
        RANK7_ELEMENT_CONFIG,
    )
    operations["rank7_weight_8x512x8"] = partial(
        launch_weight_transform,
        rank7_2x2x2,
        weight,
        rank7_right,
        shape,
        rank7_blocks,
        RANK7_WEIGHT_CONFIG,
    )
    operations["rank7_output_512x4"] = partial(
        launch_output_transform,
        rank7_2x2x2,
        rank7_products,
        rank7_output,
        shape,
        rank7_blocks,
        RANK7_ELEMENT_CONFIG,
    )
    timings, order_schedule = measure_operations(operations, warmups, rounds)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    left_keys = [element_key("left", config) for config in candidate.element_configs]
    weight_keys = [weight_key(config) for config in candidate.weight_configs]
    output_keys = [
        element_key("output", config) for config in candidate.element_configs
    ]
    best_left = min(left_keys, key=lambda name: timings[name].median_ms)
    best_weight = min(weight_keys, key=lambda name: timings[name].median_ms)
    best_output = min(output_keys, key=lambda name: timings[name].median_ms)
    candidate_ms = sum(
        timings[name].median_ms for name in (best_left, best_weight, best_output)
    )
    rank7_ms = sum(
        timings[name].median_ms
        for name in (
            "rank7_left_512x4",
            "rank7_weight_8x512x8",
            "rank7_output_512x4",
        )
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "certificate_path": str(candidate.certificate_path),
            "certificate_sha256": candidate.generated.CERTIFICATE_SHA256,
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
        },
        "device": {
            "name": properties.name,
            "architecture": architecture,
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": total_memory,
        },
        "protocol": {
            "candidate": candidate_name,
            "shape": list(shape),
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "element_configs": [asdict(config) for config in candidate.element_configs],
            "weight_configs": [asdict(config) for config in candidate.weight_configs],
            "contract": (
                "preallocated tensors; leaf BMM and allocation excluded; "
                "bounded static launch configurations only"
            ),
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "candidate_workspace_bytes": workspace_elements(
                shape, candidate_dimensions, candidate_rank
            )
            * torch.float16.itemsize,
            "rank7_workspace_bytes": workspace_elements(
                shape, rank7_dimensions, rank7_rank
            )
            * torch.float16.itemsize,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "best_candidate_stages": {
            "left": best_left,
            "weight": best_weight,
            "output": best_output,
        },
        "transform_totals_ms": {
            "candidate": candidate_ms,
            "rank7": rank7_ms,
        },
        "candidate_over_rank7": rank7_ms / candidate_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-memory-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.candidate,
        arguments.shape,
        arguments.warmups,
        arguments.rounds,
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
