import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from typing import cast, Literal

import torch
import triton
import triton.language as tl

from benchmarks.benchmark import git_output, measure_operations, parse_shape
from benchmarks.protocol import BLAS_BACKENDS
from rdna3_fastmm.runtime import Rank7Plan
from research.prototypes.benchmark_rectangular_products import (
    LIBRARY_RESERVE_BYTES,
    block_shape,
)


Layout = Literal["rank-major", "rank-inner"]


@dataclass(frozen=True)
class KernelConfig:
    block_rows: int
    block_columns: int
    block_inner: int
    warps: int
    layout: Layout


CONFIGS = (
    KernelConfig(64, 64, 32, 4, "rank-major"),
    KernelConfig(64, 128, 32, 8, "rank-major"),
    KernelConfig(128, 64, 32, 8, "rank-major"),
    KernelConfig(64, 128, 32, 8, "rank-inner"),
)
GROUP_SIZE_ROWS = 8


@triton.jit
def rank_grid_product_kernel(
    left,
    right,
    output,
    rank_count: tl.constexpr,
    row_count: tl.constexpr,
    inner_count: tl.constexpr,
    column_count: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
    block_inner: tl.constexpr,
    group_size_rows: tl.constexpr,
    rank_inner: tl.constexpr,
):
    row_programs = tl.cdiv(row_count, block_rows)
    column_programs = tl.cdiv(column_count, block_columns)
    tile_count = row_programs * column_programs
    program_index = tl.program_id(0)
    if rank_inner:
        rank_index = program_index % rank_count
        tile_index = program_index // rank_count
    else:
        tile_index = program_index % tile_count
        rank_index = program_index // tile_count
    programs_per_group = group_size_rows * column_programs
    group_index = tile_index // programs_per_group
    first_row_program = group_index * group_size_rows
    active_group_rows = tl.minimum(row_programs - first_row_program, group_size_rows)
    row_program = first_row_program + (tile_index % active_group_rows)
    column_program = (tile_index % programs_per_group) // active_group_rows
    row_offsets = row_program * block_rows + tl.arange(0, block_rows)
    column_offsets = column_program * block_columns + tl.arange(0, block_columns)
    inner_offsets = tl.arange(0, block_inner)
    left_plane = row_count * inner_count
    right_plane = inner_count * column_count
    output_plane = row_count * column_count
    left_base = left + rank_index * left_plane
    right_base = right + rank_index * right_plane
    accumulator = tl.zeros((block_rows, block_columns), dtype=tl.float32)
    for inner_block in tl.range(0, tl.cdiv(inner_count, block_inner)):
        current_inner = inner_block * block_inner + inner_offsets
        left_values = tl.load(
            left_base + row_offsets[:, None] * inner_count + current_inner[None, :],
            mask=(row_offsets[:, None] < row_count)
            & (current_inner[None, :] < inner_count),
            other=0.0,
        )
        right_values = tl.load(
            right_base
            + current_inner[:, None] * column_count
            + column_offsets[None, :],
            mask=(current_inner[:, None] < inner_count)
            & (column_offsets[None, :] < column_count),
            other=0.0,
        )
        accumulator = tl.dot(left_values, right_values, accumulator)
    output_offsets = (
        rank_index * output_plane
        + row_offsets[:, None] * column_count
        + column_offsets[None, :]
    )
    tl.store(
        output + output_offsets,
        accumulator.to(tl.float16),
        mask=(row_offsets[:, None] < row_count)
        & (column_offsets[None, :] < column_count),
    )


def config_name(config: KernelConfig) -> str:
    return (
        f"triton_{config.block_rows}x{config.block_columns}x"
        f"{config.block_inner}_{config.warps}w_{config.layout}"
    )


def launch_rank_grid(
    left: torch.Tensor,
    right: torch.Tensor,
    output: torch.Tensor,
    config: KernelConfig,
) -> object:
    rank_count, row_count, inner_count = left.shape
    column_count = right.shape[2]
    tile_count = triton.cdiv(row_count, config.block_rows) * triton.cdiv(
        column_count, config.block_columns
    )
    return rank_grid_product_kernel[(rank_count * tile_count,)](
        left,
        right,
        output,
        rank_count,
        row_count,
        inner_count,
        column_count,
        config.block_rows,
        config.block_columns,
        config.block_inner,
        GROUP_SIZE_ROWS,
        config.layout == "rank-inner",
        num_warps=config.warps,
        num_stages=1,
    )


def assembly_integer(assembly: str, directive: str) -> int | None:
    match = re.search(rf"{re.escape(directive)}:?\s+(\d+)", assembly)
    return None if match is None else int(match.group(1))


def assembly_summary(compiled: object) -> dict[str, object]:
    raw_asm = getattr(compiled, "asm", {})
    asm = cast(dict[str, object], raw_asm)
    amdgcn = str(asm.get("amdgcn", ""))
    metadata = getattr(compiled, "metadata", None)
    return {
        "asm_keys": sorted(asm),
        "name": getattr(metadata, "name", None),
        "warps": getattr(metadata, "num_warps", None),
        "shared_bytes": getattr(metadata, "shared", None),
        "vgprs": assembly_integer(amdgcn, ".amdhsa_next_free_vgpr"),
        "private_segment_bytes": assembly_integer(
            amdgcn, ".amdhsa_private_segment_fixed_size"
        ),
        "vgpr_spill_count": assembly_integer(amdgcn, ".vgpr_spill_count"),
        "scratch_store_count": amdgcn.count("scratch_store_"),
        "scratch_load_count": amdgcn.count("scratch_load_"),
        "wmma_instruction_count": amdgcn.count("v_wmma_f32_16x16x16_f16"),
    }


def sampled_error(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float | int]:
    row_count = reference.shape[1]
    column_count = reference.shape[2]
    row_starts = (0, row_count // 2 - 2, row_count - 4)
    column_starts = (0, column_count // 2 - 2, column_count - 4)
    references = []
    candidates = []
    for row_start in row_starts:
        for column_start in column_starts:
            references.append(
                reference[:, row_start : row_start + 4, column_start : column_start + 4]
                .contiguous()
                .cpu()
                .float()
            )
            candidates.append(
                candidate[:, row_start : row_start + 4, column_start : column_start + 4]
                .contiguous()
                .cpu()
                .float()
            )
    reference_sample = torch.cat([value.flatten() for value in references])
    candidate_sample = torch.cat([value.flatten() for value in candidates])
    difference = candidate_sample - reference_sample
    finite = torch.isfinite(candidate_sample)
    safe_difference = torch.where(finite, difference, torch.zeros_like(difference))
    return {
        "sample_count": candidate_sample.numel(),
        "relative_l2_error": math.sqrt(
            float(torch.sum(safe_difference.double().square()).item())
            / float(torch.sum(reference_sample.double().square()).item())
        ),
        "maximum_absolute_error": float(safe_difference.abs().max().item()),
        "nonfinite_count": int((~finite).sum().item()),
    }


def benchmark(
    shape: tuple[int, int, int],
    warmups: int,
    rounds: int,
    max_memory_fraction: float,
    blas_backend: str,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rank-grid research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    selected_blas_backend = torch.backends.cuda.preferred_blas_library(blas_backend)
    dimensions = (Rank7Plan.scheme_size,) * 3
    blocks = block_shape(shape, dimensions)
    left_elements = Rank7Plan.rank * blocks[0] * blocks[1]
    right_elements = Rank7Plan.rank * blocks[1] * blocks[2]
    product_elements = Rank7Plan.rank * blocks[0] * blocks[2]
    required_bytes = (
        left_elements + right_elements + 2 * product_elements
    ) * torch.float16.itemsize + LIBRARY_RESERVE_BYTES
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs an estimated {required_bytes / 2**30:.2f} GiB, "
            f"exceeding the {free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.manual_seed(59)
    torch.cuda.manual_seed_all(59)
    torch.cuda.reset_peak_memory_stats(device)
    left = torch.randn(
        (Rank7Plan.rank, blocks[0], blocks[1]),
        device=device,
        dtype=torch.float16,
    )
    right = torch.randn(
        (Rank7Plan.rank, blocks[1], blocks[2]),
        device=device,
        dtype=torch.float16,
    )
    reference = torch.empty(
        (Rank7Plan.rank, blocks[0], blocks[2]),
        device=device,
        dtype=torch.float16,
    )
    candidate = torch.empty_like(reference)
    compiled_kernels: dict[str, object] = {}

    def run_bmm() -> None:
        torch.bmm(left, right, out=reference)

    operations: dict[str, Callable[[], None]] = {"torch_bmm": run_bmm}
    for config in CONFIGS:
        name = config_name(config)

        def run_candidate(
            selected_config: KernelConfig = config,
            selected_name: str = name,
        ) -> None:
            compiled_kernels[selected_name] = launch_rank_grid(
                left, right, candidate, selected_config
            )

        operations[name] = run_candidate
    timings, order_schedule = measure_operations(operations, warmups, rounds)
    best_name = min(
        (config_name(config) for config in CONFIGS),
        key=lambda name: timings[name].median_ms,
    )
    best_config = next(config for config in CONFIGS if config_name(config) == best_name)
    run_bmm()
    launch_rank_grid(left, right, candidate, best_config)
    torch.cuda.synchronize()
    correctness = sampled_error(reference, candidate)
    free_after, _ = torch.cuda.mem_get_info(device)
    bmm_ms = timings["torch_bmm"].median_ms
    best_ms = timings[best_name].median_ms
    leaf_flops = 2 * Rank7Plan.rank * blocks[0] * blocks[1] * blocks[2]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "research_basis": "SubCuber-style rank-parallel leaf grid",
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "blas_backend": str(selected_blas_backend),
        },
        "device": {
            "name": properties.name,
            "architecture": architecture,
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": total_memory,
        },
        "protocol": {
            "shape": list(shape),
            "leaf_shape": [Rank7Plan.rank, *blocks],
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "configs": [asdict(config) for config in CONFIGS],
            "contract": (
                "preallocated FP16 rank-7 leaves and products; one rank and output "
                "tile per Triton program; transforms and reconstruction excluded"
            ),
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "best_candidate": best_name,
        "candidate_over_bmm": bmm_ms / best_ms,
        "executed_leaf_tflops": {
            "torch_bmm": leaf_flops / bmm_ms / 1e9,
            "best_candidate": leaf_flops / best_ms / 1e9,
        },
        "assembly": {
            name: assembly_summary(compiled)
            for name, compiled in compiled_kernels.items()
        },
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-memory-fraction", type=float, default=0.12)
    parser.add_argument(
        "--blas-backend",
        choices=BLAS_BACKENDS,
        default="hipblas",
    )
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
        arguments.blas_backend,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)
        print(arguments.output)


if __name__ == "__main__":
    main()
