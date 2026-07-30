import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
import json
from pathlib import Path
from types import ModuleType

import torch
import triton

from benchmarks.benchmark import git_output, measure_operations, parse_shape
from research.prototypes import (
    generated_rank23_3x3x3,
    generated_rank23_55add_3x3x3,
    generated_rank23_56add_3x3x3,
)
from research.prototypes.benchmark_rectangular_products import (
    LIBRARY_RESERVE_BYTES,
    block_shape,
    workspace_elements,
)
from research.prototypes.benchmark_rectangular_stages import (
    launch_left_transform,
    launch_output_transform,
    launch_weight_transform,
)
from rdna3_fastmm.runtime import ElementTransformConfig, WeightTransformConfig


SCHEDULES: dict[str, tuple[ModuleType, Path]] = {
    "rank23_58add": (
        generated_rank23_3x3x3,
        Path("certificates/research/3x3x3_rank23_58add/certificate.json"),
    ),
    "rank23_56add": (
        generated_rank23_56add_3x3x3,
        Path("certificates/research/3x3x3_rank23_56add/certificate.json"),
    ),
    "rank23_55add": (
        generated_rank23_55add_3x3x3,
        Path("certificates/research/3x3x3_rank23_55add/certificate.json"),
    ),
}
ELEMENT_CONFIG = ElementTransformConfig(256, 2)
WEIGHT_CONFIG = WeightTransformConfig(8, 512, 8)


def benchmark(
    shape: tuple[int, int, int],
    warmups: int,
    rounds: int,
    max_memory_fraction: float,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rank-23 research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    dimensions = generated_rank23_3x3x3.DIMENSIONS
    rank = generated_rank23_3x3x3.RANK
    if any(
        generated.DIMENSIONS != dimensions or generated.RANK != rank
        for generated, _ in SCHEDULES.values()
    ):
        raise RuntimeError("rank-23 schedule dimensions do not match")
    blocks = block_shape(shape, dimensions)
    source_elements = shape[0] * shape[1] + shape[2] * shape[1]
    output_elements = shape[0] * shape[2]
    required_bytes = (
        (source_elements + output_elements) * torch.bfloat16.itemsize
        + workspace_elements(shape, dimensions, rank) * torch.float16.itemsize
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
    left_transformed = torch.empty(
        (rank, blocks[0], blocks[1]), device=device, dtype=torch.float16
    )
    right_transformed = torch.empty(
        (rank, blocks[1], blocks[2]), device=device, dtype=torch.float16
    )
    products = torch.full(
        (rank, blocks[0], blocks[2]), 0.5, device=device, dtype=torch.float16
    )
    output = torch.empty((shape[0], shape[2]), device=device, dtype=torch.bfloat16)
    operations = {}
    for name, (generated, _) in SCHEDULES.items():
        operations[f"{name}_left"] = partial(
            launch_left_transform,
            generated,
            input_tensor,
            left_transformed,
            shape,
            blocks,
            ELEMENT_CONFIG,
        )
        operations[f"{name}_weight"] = partial(
            launch_weight_transform,
            generated,
            weight,
            right_transformed,
            shape,
            blocks,
            WEIGHT_CONFIG,
        )
        operations[f"{name}_output"] = partial(
            launch_output_transform,
            generated,
            products,
            output,
            shape,
            blocks,
            ELEMENT_CONFIG,
        )
    timings, order_schedule = measure_operations(operations, warmups, rounds)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    stage_totals = {
        name: sum(
            timings[f"{name}_{stage}"].median_ms
            for stage in ("left", "weight", "output")
        )
        for name in SCHEDULES
    }
    baseline_total = stage_totals["rank23_58add"]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "certificates": {
                name: {
                    "path": str(path),
                    "sha256": generated.CERTIFICATE_SHA256,
                }
                for name, (generated, path) in SCHEDULES.items()
            },
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
            "shape": list(shape),
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "element_config": asdict(ELEMENT_CONFIG),
            "weight_config": asdict(WEIGHT_CONFIG),
            "contract": (
                "shared preallocated tensors; leaf BMM and allocation excluded; "
                "one fixed launch geometry isolates circuit schedule cost"
            ),
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "workspace_bytes": workspace_elements(shape, dimensions, rank)
            * torch.float16.itemsize,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "stage_totals_ms": stage_totals,
        "relative_to_58add": {
            name: baseline_total / total for name, total in stage_totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-memory-fraction", type=float, default=0.12)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    torch.set_grad_enabled(False)
    report = benchmark(
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
