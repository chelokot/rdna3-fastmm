import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path

import torch
import triton

from benchmarks.benchmark import (
    git_output,
    measure_operations,
    parse_shape,
    validate_outputs,
)
from benchmarks.protocol import BLAS_BACKENDS
from rdna3_fastmm.linear import rdna3_rank7_linear
from rdna3_fastmm.runtime import (
    ElementTransformConfig,
    LinearShapeFamily,
    Rank7Plan,
    WeightTransformConfig,
    _LinearPlan,
)
from research.prototypes import (
    generated_rank23_3x3x3,
    generated_rank23_55add_3x3x3,
    generated_rank23_56add_3x3x3,
)


LIBRARY_RESERVE_BYTES = 128 * 2**20


class Rank23ResearchPlan(_LinearPlan):
    algorithm = "rank23-58add-research-v1"
    generated = generated_rank23_3x3x3
    rank = generated_rank23_3x3x3.RANK
    scheme_size = generated_rank23_3x3x3.DIMENSIONS[0]
    precision_pairs = frozenset({(torch.bfloat16, torch.float16)})
    dynamic_shapes: frozenset[tuple[int, int, int]] = frozenset()
    prepacked_shapes: frozenset[tuple[int, int, int]] = frozenset()
    linear_shape_families: tuple[LinearShapeFamily, ...] = ()
    default_transform_config = ElementTransformConfig(256, 2)
    transform_configs: dict[tuple[int, int], ElementTransformConfig] = {}
    default_weight_transform_config = WeightTransformConfig(8, 512, 8)
    weight_transform_configs: dict[tuple[int, int], WeightTransformConfig] = {}


class Rank23Addition56ResearchPlan(Rank23ResearchPlan):
    algorithm = "rank23-56add-research-v1"
    generated = generated_rank23_56add_3x3x3


class Rank23Addition55ResearchPlan(Rank23ResearchPlan):
    algorithm = "rank23-55add-research-v1"
    generated = generated_rank23_55add_3x3x3


SCHEDULES: dict[
    str,
    tuple[type[Rank23ResearchPlan], Path, Path],
] = {
    "58add": (
        Rank23ResearchPlan,
        Path("certificates/research/3x3x3_rank23_58add/certificate.json"),
        Path("research/prototypes/generated_rank23_3x3x3.py"),
    ),
    "56add": (
        Rank23Addition56ResearchPlan,
        Path("certificates/research/3x3x3_rank23_56add/certificate.json"),
        Path("research/prototypes/generated_rank23_56add_3x3x3.py"),
    ),
    "55add": (
        Rank23Addition55ResearchPlan,
        Path("certificates/research/3x3x3_rank23_55add/certificate.json"),
        Path("research/prototypes/generated_rank23_55add_3x3x3.py"),
    ),
}


def estimated_required_bytes(
    shape: tuple[int, int, int], plan: Rank23ResearchPlan, has_bias: bool
) -> int:
    rows, inner, columns = shape
    source_elements = rows * inner + columns * inner
    if has_bias:
        source_elements += columns
    rank7_block_rows = triton.cdiv(rows, Rank7Plan.scheme_size)
    rank7_block_inner = triton.cdiv(inner, Rank7Plan.scheme_size)
    rank7_block_columns = triton.cdiv(columns, Rank7Plan.scheme_size)
    rank7_workspace_elements = Rank7Plan.rank * (
        rank7_block_rows * rank7_block_inner
        + rank7_block_inner * rank7_block_columns
        + rank7_block_rows * rank7_block_columns
    )
    retained_output_elements = 6 * rows * columns
    return (
        (source_elements + retained_output_elements) * torch.bfloat16.itemsize
        + (plan.workspace_bytes + rank7_workspace_elements * torch.float16.itemsize)
        + LIBRARY_RESERVE_BYTES
    )


def benchmark(
    shape: tuple[int, int, int],
    has_bias: bool,
    warmups: int,
    rounds: int,
    tile_size: int,
    max_memory_fraction: float,
    seed: int,
    blas_backend: str,
    schedule: str,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rank-23 research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    selected_blas_backend = torch.backends.cuda.preferred_blas_library(blas_backend)
    plan_type, certificate_path, generated_path = SCHEDULES[schedule]
    plan = plan_type(
        *shape,
        device=device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    if tile_size > min(plan.shape.block_rows, plan.shape.block_columns):
        raise ValueError("sample tile must fit inside every output macroblock")
    required_bytes = estimated_required_bytes(shape, plan, has_bias)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if required_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs an estimated {required_bytes / 2**30:.2f} GiB, "
            f"exceeding the {free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    rows, inner, columns = shape
    input_tensor = torch.randn((rows, inner), device=device, dtype=torch.bfloat16)
    weight = torch.randn((columns, inner), device=device, dtype=torch.bfloat16)
    bias = (
        torch.randn((columns,), device=device, dtype=torch.bfloat16)
        if has_bias
        else None
    )
    workspace = plan.allocate_workspace(max_free_memory_fraction=1.0)
    outputs = {
        "torch_linear": torch.empty(
            (rows, columns), device=device, dtype=torch.bfloat16
        ),
        "rank23_plan": torch.empty(
            (rows, columns), device=device, dtype=torch.bfloat16
        ),
        "rank7_operator": torch.empty(
            (rows, columns), device=device, dtype=torch.bfloat16
        ),
    }

    def run_torch_linear() -> None:
        outputs["torch_linear"] = torch.nn.functional.linear(input_tensor, weight, bias)

    def run_rank23_plan() -> None:
        outputs["rank23_plan"] = plan.run_linear(input_tensor, weight, workspace, bias)

    def run_rank7_operator() -> None:
        outputs["rank7_operator"] = rdna3_rank7_linear(input_tensor, weight, bias)

    timings, order_schedule = measure_operations(
        {
            "torch_linear": run_torch_linear,
            "rank23_plan": run_rank23_plan,
            "rank7_operator": run_rank7_operator,
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
    baseline_ms = timings["torch_linear"].median_ms
    rank23_ms = timings["rank23_plan"].median_ms
    rank7_ms = timings["rank7_operator"].median_ms
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "certificate_path": str(certificate_path),
            "certificate_sha256": plan.generated.CERTIFICATE_SHA256,
            "generated_path": str(generated_path),
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
            "bias": has_bias,
            "schedule": schedule,
            "warmup_rounds": warmups,
            "measured_rounds": rounds,
            "order_schedule": order_schedule,
            "tile_size": tile_size,
            "candidate_contract": (
                "optimistic plan-level dynamic weight transform with reused workspace "
                "and allocating output"
            ),
            "control_contract": "public allocating rdna3_fastmm::linear_rank7 operator",
        },
        "memory": {
            "estimated_required_bytes": required_bytes,
            "workspace_bytes": plan.workspace_bytes,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "speedup": {
            "rank23_over_torch": baseline_ms / rank23_ms,
            "rank7_over_torch": baseline_ms / rank7_ms,
            "rank23_over_rank7": rank7_ms / rank23_ms,
        },
        "executed_multiplication_fraction": {
            "rank23": plan.rank / plan.scheme_size**3,
            "rank7": Rank7Plan.rank / Rank7Plan.scheme_size**3,
        },
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--max-memory-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), default="58add")
    parser.add_argument(
        "--blas-backend",
        choices=BLAS_BACKENDS,
        default="hipblas",
    )
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
        not arguments.no_bias,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.max_memory_fraction,
        arguments.seed,
        arguments.blas_backend,
        arguments.schedule,
    )
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)
        print(arguments.output)


if __name__ == "__main__":
    main()
