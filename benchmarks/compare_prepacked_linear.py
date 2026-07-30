from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
from typing import cast, TypeAlias

import torch
import triton

from benchmarks.benchmark import (
    file_sha256,
    git_output,
    measure_operations,
    measure_packing,
    packing_break_even_reuses,
    parse_shape,
    validate_outputs,
)
from rdna3_fastmm.runtime import (
    PackedWeight,
    Rank7Plan,
    Rank48Plan,
    Rank49Plan,
    Workspace,
)


LinearPlan: TypeAlias = Rank7Plan | Rank48Plan | Rank49Plan
PLAN_TYPES: dict[
    str,
    type[Rank7Plan] | type[Rank48Plan] | type[Rank49Plan],
] = {
    "rank7": Rank7Plan,
    "rank49": Rank49Plan,
    "rank48": Rank48Plan,
}
ARTIFACTS = {
    "rank7": (
        Path("certificates/2x2x2_rank7_15add/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank7_2x2x2.py"),
    ),
    "rank49": (
        Path("certificates/4x4x4_rank49_159add/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank49_4x4x4.py"),
    ),
    "rank48": (
        Path("certificates/4x4x4_rank48_accurate/certificate.json"),
        Path("src/rdna3_fastmm/generated/rank48_4x4x4.py"),
    ),
}
ENVIRONMENT_KEYS = (
    "HIP_VISIBLE_DEVICES",
    "PYTORCH_TUNABLEOP_ENABLED",
    "PYTORCH_TUNABLEOP_FILENAME",
    "TORCH_BLAS_PREFER_HIPBLASLT",
)
LIBRARY_WORKSPACE_BYTES = 128 * 2**20


@dataclass(frozen=True)
class MemoryEstimate:
    source_bytes: int
    output_bytes: int
    plan_bytes: dict[str, int]
    library_workspace_bytes: int
    total_bytes: int


def estimate_memory(
    shape: tuple[int, int, int],
    plans: dict[str, LinearPlan],
) -> MemoryEstimate:
    rows, inner, columns = shape
    source_bytes = (rows * inner + columns * inner) * torch.bfloat16.itemsize
    output_bytes = (len(plans) + 1) * rows * columns * torch.bfloat16.itemsize
    plan_bytes = {
        name: plan.prepacked_workspace_bytes + plan.packed_weight_bytes
        for name, plan in plans.items()
    }
    return MemoryEstimate(
        source_bytes=source_bytes,
        output_bytes=output_bytes,
        plan_bytes=plan_bytes,
        library_workspace_bytes=LIBRARY_WORKSPACE_BYTES,
        total_bytes=(
            source_bytes
            + output_bytes
            + sum(plan_bytes.values())
            + LIBRARY_WORKSPACE_BYTES
        ),
    )


def plan_operation(
    plan: LinearPlan,
    input_tensor: torch.Tensor,
    packed_weight: PackedWeight,
    workspace: Workspace,
    output: torch.Tensor,
) -> Callable[[], None]:
    def run() -> None:
        plan.run_linear_packed(
            input_tensor,
            packed_weight,
            workspace,
            output=output,
        )

    return run


def weight_pack_operation(
    plan: LinearPlan,
    weight: torch.Tensor,
) -> Callable[[], PackedWeight]:
    def pack() -> PackedWeight:
        return plan.pack_weight(
            weight,
            max_free_memory_fraction=1.0,
        )

    return pack


def artifact_provenance() -> dict[str, dict[str, str]]:
    return {
        name: {
            "certificate_path": str(certificate),
            "certificate_sha256": file_sha256(certificate),
            "generated_module_path": str(generated),
            "generated_module_sha256": file_sha256(generated),
        }
        for name, (certificate, generated) in ARTIFACTS.items()
    }


def benchmark(
    shape: tuple[int, int, int],
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
    plans: dict[str, LinearPlan] = {
        name: plan_type(
            *shape,
            device=device,
            dtype=torch.bfloat16,
            compute_dtype=torch.float16,
        )
        for name, plan_type in PLAN_TYPES.items()
    }
    if not allow_unrecommended and not plans["rank48"].is_linear_recommended(
        has_bias=False,
        prepacked_weight=True,
    ):
        raise RuntimeError("shape is outside the measured rank-48 prepacked gate")
    memory = estimate_memory(shape, plans)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    budget_bytes = int(free_before * max_memory_fraction)
    if memory.total_bytes > budget_bytes:
        raise MemoryError(
            f"benchmark needs {memory.total_bytes / 2**30:.2f} GiB, exceeding "
            f"the {budget_bytes / 2**30:.2f} GiB budget"
        )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    rows, inner, columns = shape
    input_tensor = torch.randn(
        (rows, inner),
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        (columns, inner),
        device=device,
        dtype=torch.bfloat16,
    )
    outputs = {
        name: torch.empty(
            (rows, columns),
            device=device,
            dtype=torch.bfloat16,
        )
        for name in ("torch_bfloat16", *plans)
    }
    packed_weights: dict[str, PackedWeight] = {}
    packing: dict[str, dict[str, float | int | None]] = {}
    for name, plan in plans.items():
        packed, timing = measure_packing(weight_pack_operation(plan, weight))
        if not isinstance(packed, PackedWeight):
            raise AssertionError("Linear plan returned an incompatible packed weight")
        packed_weights[name] = packed
        source_weight_bytes = weight.numel() * weight.element_size()
        packing[name] = timing | {
            "packed_bytes": plan.packed_weight_bytes,
            "weight_expansion": plan.packed_weight_bytes / source_weight_bytes,
        }
    workspaces = {
        name: plan.allocate_workspace(
            max_free_memory_fraction=1.0,
            prepacked_right=True,
        )
        for name, plan in plans.items()
    }

    def run_torch() -> None:
        torch.mm(input_tensor, weight.T, out=outputs["torch_bfloat16"])

    operations: dict[str, Callable[[], None]] = {"torch_bfloat16": run_torch}
    operations.update(
        {
            name: plan_operation(
                plan,
                input_tensor,
                packed_weights[name],
                workspaces[name],
                outputs[name],
            )
            for name, plan in plans.items()
        }
    )
    timings, order = measure_operations(operations, warmups, rounds)
    correctness = validate_outputs(
        input_tensor,
        weight.T,
        None,
        outputs,
        "torch_bfloat16",
        tile_size,
        Rank48Plan.scheme_size,
    )
    native_ms = timings["torch_bfloat16"].median_ms
    rank48_ms = timings["rank48"].median_ms
    control_ms = {name: timings[name].median_ms for name in ("rank7", "rank49")}
    best_control_name = min(control_ms, key=control_ms.__getitem__)
    best_control_ms = control_ms[best_control_name]
    for name, plan in plans.items():
        packing[name]["break_even_reuses_vs_native"] = packing_break_even_reuses(
            cast(float, packing[name]["device_ms"]),
            native_ms,
            timings[name].median_ms,
        )
    properties = torch.cuda.get_device_properties(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    return {
        "schema_version": 1,
        "status": (
            "rank48-wins"
            if rank48_ms < best_control_ms
            else "rank48-does-not-beat-current-control"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": dirty,
            "benchmark_path": str(Path(__file__)),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "artifacts": artifact_provenance(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
        },
        "device": {
            "name": properties.name,
            "architecture": getattr(properties, "gcnArchName", ""),
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": total_memory,
        },
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
        "protocol": {
            "shape": list(shape),
            "source_dtype": "bfloat16",
            "compute_dtype": "float16",
            "bias": False,
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order,
            "timing_method": "one operation per HIP event pair",
            "allocation_contract": (
                "all hot-path operations reuse output buffers; FastMM plans also "
                "reuse their prepacked weights and workspaces"
            ),
            "correctness_reference": "CPU FP32 tiles from original BF16 inputs",
            "tile_policy": (
                f"centered {tile_size}x{tile_size} tile in each rank-48 "
                "output macroblock"
            ),
            "seed": seed,
        },
        "case": {
            "shape": list(shape),
            "timings": {name: asdict(timing) for name, timing in timings.items()},
            "speedup_vs_native": {
                name: native_ms / timing.median_ms
                for name, timing in timings.items()
                if name != "torch_bfloat16"
            },
            "rank48_speedup_vs_controls": {
                name: timing / rank48_ms for name, timing in control_ms.items()
            },
            "best_current_control": best_control_name,
            "rank48_speedup_vs_best_current_control": (best_control_ms / rank48_ms),
            "packing": packing,
            "correctness": correctness,
            "configs": {
                name: {
                    "algorithm": plan.algorithm,
                    "transform": asdict(plan.transform_config(shape[1], shape[2])),
                    "weight_transform": asdict(
                        plan.weight_transform_config(shape[1], shape[2])
                    ),
                }
                for name, plan in plans.items()
            },
            "memory": {
                "estimate": asdict(memory),
                "budget_bytes": budget_bytes,
                "free_before_bytes": free_before,
                "free_after_bytes": free_after,
                "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-unrecommended", action="store_true")
    parser.add_argument("--require-rank48-win", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if arguments.tile_size < 1:
        parser.error("tile size must be positive")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.shape,
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
    if arguments.require_rank48_win and report["status"] != "rank48-wins":
        raise RuntimeError("rank-48 did not beat the fastest current control")


if __name__ == "__main__":
    main()
