import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import torch
import triton

from benchmarks.benchmark import git_output, measure_operations, parse_shape
from benchmarks.protocol import BLAS_BACKENDS
from rdna3_fastmm.runtime import Rank7Plan

LIBRARY_RESERVE_BYTES = 128 * 2**20


@dataclass(frozen=True)
class CandidateSpec:
    dimensions: tuple[int, int, int]
    rank: int
    certificate_path: Path
    certificate_sha256: str


CANDIDATES = {
    "rank29_3x3x4": CandidateSpec(
        (3, 3, 4),
        29,
        Path("certificates/research/3x3x4_rank29_92add/certificate.json"),
        "3d2a6d75c74511d2401f5a941a089c78dcf8e9b8eeaa3e3c8ef759fbd174f8c5",
    ),
    "rank105_4x6x6": CandidateSpec(
        (4, 6, 6),
        105,
        Path("certificates/research/4x6x6_rank105_430add/certificate.json"),
        "9427b13b5509d45df8c56294f2abe6bc4e478f2aa079a1925e600ca2183739b5",
    ),
}


def block_shape(
    shape: tuple[int, int, int], scheme_dimensions: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        triton.cdiv(dimension, divisor)
        for dimension, divisor in zip(shape, scheme_dimensions, strict=True)
    )


def workspace_elements(
    shape: tuple[int, int, int],
    scheme_dimensions: tuple[int, int, int],
    rank: int,
) -> int:
    block_rows, block_inner, block_columns = block_shape(shape, scheme_dimensions)
    return rank * (
        block_rows * block_inner
        + block_inner * block_columns
        + block_rows * block_columns
    )


def benchmark(
    candidate_name: str,
    shape: tuple[int, int, int],
    warmups: int,
    rounds: int,
    max_memory_fraction: float,
    blas_backend: str,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rectangular research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    selected_blas_backend = torch.backends.cuda.preferred_blas_library(blas_backend)
    candidate = CANDIDATES[candidate_name]
    candidate_dimensions = candidate.dimensions
    candidate_rank = candidate.rank
    candidate_blocks = block_shape(shape, candidate_dimensions)
    rank7_dimensions = (Rank7Plan.scheme_size,) * 3
    rank7_blocks = block_shape(shape, rank7_dimensions)
    candidate_workspace_elements = workspace_elements(
        shape, candidate_dimensions, candidate_rank
    )
    rank7_workspace_elements = workspace_elements(
        shape, rank7_dimensions, Rank7Plan.rank
    )
    required_bytes = (
        candidate_workspace_elements + rank7_workspace_elements
    ) * torch.float16.itemsize + LIBRARY_RESERVE_BYTES
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs an estimated {required_bytes / 2**30:.2f} GiB, "
            f"exceeding the {free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.cuda.reset_peak_memory_stats(device)
    candidate_left = torch.full(
        (candidate_rank, candidate_blocks[0], candidate_blocks[1]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    candidate_right = torch.full(
        (candidate_rank, candidate_blocks[1], candidate_blocks[2]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    candidate_products = torch.empty(
        (candidate_rank, candidate_blocks[0], candidate_blocks[2]),
        device=device,
        dtype=torch.float16,
    )
    rank7_left = torch.full(
        (Rank7Plan.rank, rank7_blocks[0], rank7_blocks[1]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    rank7_right = torch.full(
        (Rank7Plan.rank, rank7_blocks[1], rank7_blocks[2]),
        0.5,
        device=device,
        dtype=torch.float16,
    )
    rank7_products = torch.empty(
        (Rank7Plan.rank, rank7_blocks[0], rank7_blocks[2]),
        device=device,
        dtype=torch.float16,
    )

    def run_candidate_products() -> None:
        torch.bmm(candidate_left, candidate_right, out=candidate_products)

    def run_rank7_products() -> None:
        torch.bmm(rank7_left, rank7_right, out=rank7_products)

    timings, order_schedule = measure_operations(
        {
            "candidate_products": run_candidate_products,
            "rank7_products": run_rank7_products,
        },
        warmups,
        rounds,
    )
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    candidate_flops = (
        2
        * candidate_rank
        * candidate_blocks[0]
        * candidate_blocks[1]
        * candidate_blocks[2]
    )
    rank7_flops = (
        2 * Rank7Plan.rank * rank7_blocks[0] * rank7_blocks[1] * rank7_blocks[2]
    )
    candidate_ms = timings["candidate_products"].median_ms
    rank7_ms = timings["rank7_products"].median_ms
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "certificate_path": str(candidate.certificate_path),
            "certificate_sha256": candidate.certificate_sha256,
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
            "candidate": candidate_name,
            "shape": list(shape),
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "contract": (
                "preallocated FP16 leaf tensors and products; transforms, "
                "reconstruction, and allocation excluded"
            ),
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "candidate_workspace_bytes": (
                candidate_workspace_elements * torch.float16.itemsize
            ),
            "rank7_workspace_bytes": rank7_workspace_elements * torch.float16.itemsize,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "leaf_shapes": {
            "candidate": [candidate_rank, *candidate_blocks],
            "rank7": [Rank7Plan.rank, *rank7_blocks],
        },
        "leaf_flops": {
            "candidate": candidate_flops,
            "rank7": rank7_flops,
        },
        "executed_multiplication_fraction": {
            "candidate": candidate_rank
            / (
                candidate_dimensions[0]
                * candidate_dimensions[1]
                * candidate_dimensions[2]
            ),
            "rank7": Rank7Plan.rank / Rank7Plan.scheme_size**3,
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "candidate_over_rank7": rank7_ms / candidate_ms,
        "executed_leaf_tflops": {
            "candidate": candidate_flops / candidate_ms / 1e9,
            "rank7": rank7_flops / rank7_ms / 1e9,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-memory-fraction", type=float, default=0.15)
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
        arguments.candidate,
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
