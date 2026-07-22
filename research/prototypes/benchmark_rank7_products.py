import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import triton

from benchmarks.benchmark import measure_operations, validate_outputs
from rdna3_fastmm.generated import rank7_2x2x2
from rdna3_fastmm.linear import rdna3_rank7_linear
from rdna3_fastmm.runtime import Rank7Plan
from research.prototypes import generated_rank7_transposed_output
from research.prototypes.benchmark_rank7_linear import (
    KernelConfig,
    LinearShape,
    Rank7Linear,
    parse_shape,
)


@dataclass(frozen=True)
class TransposeConfig:
    block_rows: int
    block_columns: int
    warps: int


def reconstruct_transposed_products(
    shape: LinearShape,
    products: torch.Tensor,
    output: torch.Tensor,
    config: TransposeConfig,
) -> None:
    grid = (
        triton.cdiv(shape.block_rows, config.block_rows),
        triton.cdiv(shape.block_columns, config.block_columns),
    )
    generated_rank7_transposed_output.output_transform_transposed_products_kernel[grid](
        products,
        output,
        shape.rows,
        shape.columns,
        shape.block_rows,
        shape.block_columns,
        config.block_rows,
        config.block_columns,
        num_warps=config.warps,
        num_stages=1,
    )


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def required_bytes(shape: LinearShape, block_row_alignment: int) -> int:
    source_elements = shape.rows * shape.inner + shape.columns * shape.inner
    left_elements = Rank7Plan.rank * shape.block_rows * shape.block_inner
    right_elements = Rank7Plan.rank * shape.block_inner * shape.block_columns
    product_elements = Rank7Plan.rank * shape.block_rows * shape.block_columns
    workspace_elements = 2 * (left_elements + right_elements + 2 * product_elements)
    padded_rows = align_up(shape.block_rows, block_row_alignment)
    if padded_rows != shape.block_rows:
        workspace_elements += Rank7Plan.rank * (
            padded_rows * shape.block_inner + 2 * padded_rows * shape.block_columns
        )
    retained_output_elements = 6 * shape.rows * shape.columns
    return (
        source_elements + workspace_elements + retained_output_elements
    ) * torch.bfloat16.itemsize + 128 * 2**20


def benchmark(
    dimensions: tuple[int, int, int],
    transpose_config: TransposeConfig,
    warmups: int,
    rounds: int,
    tile_size: int,
    max_memory_fraction: float,
    block_row_alignment: int,
    blas_backend: str,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the transposed-product benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    selected_blas_backend = torch.backends.cuda.preferred_blas_library(blas_backend)
    shape = LinearShape.from_dimensions(*dimensions)
    memory_bytes = required_bytes(shape, block_row_alignment)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if memory_bytes > free_before * max_memory_fraction:
        raise MemoryError(
            f"benchmark needs {memory_bytes / 2**30:.2f} GiB, exceeding the "
            f"{free_before * max_memory_fraction / 2**30:.2f} GiB budget"
        )
    transform = Rank7Plan.transform_config(shape.inner, shape.columns)
    weight = Rank7Plan.weight_transform_config(shape.inner, shape.columns)
    kernel_config = KernelConfig(
        transform.block_elements,
        transform.warps,
        weight.block_rows,
        weight.block_columns,
        weight.warps,
    )
    candidate = Rank7Linear(rank7_2x2x2, shape, device, kernel_config)
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = torch.randn(
        (shape.rows, shape.inner), device=device, dtype=torch.bfloat16
    )
    linear_weight = torch.randn(
        (shape.columns, shape.inner), device=device, dtype=torch.bfloat16
    )
    left_transformed = candidate.allocate_left()
    right_transformed = candidate.allocate_weight()
    products = candidate.allocate_products()
    transposed_products = torch.empty(
        (Rank7Plan.rank, shape.block_columns, shape.block_rows),
        device=device,
        dtype=torch.float16,
    )
    current_output = candidate.allocate_output()
    transposed_output = candidate.allocate_output()
    individual_output = candidate.allocate_output()
    candidate.transform_left(input_tensor, left_transformed)
    candidate.transform_weight(linear_weight, right_transformed)

    def run_current_bmm() -> None:
        torch.bmm(left_transformed, right_transformed, out=products)

    def run_transposed_bmm() -> None:
        torch.bmm(
            right_transformed.transpose(1, 2),
            left_transformed.transpose(1, 2),
            out=transposed_products,
        )

    def run_individual_bmm() -> None:
        for rank_index in range(Rank7Plan.rank):
            torch.mm(
                left_transformed[rank_index],
                right_transformed[rank_index],
                out=products[rank_index],
            )

    def run_chunked_bmm(chunk_sizes: tuple[int, ...]) -> None:
        rank_start = 0
        for chunk_size in chunk_sizes:
            rank_end = rank_start + chunk_size
            torch.bmm(
                left_transformed[rank_start:rank_end],
                right_transformed[rank_start:rank_end],
                out=products[rank_start:rank_end],
            )
            rank_start = rank_end

    def run_bmm_4_3() -> None:
        run_chunked_bmm((4, 3))

    def run_bmm_2_2_2_1() -> None:
        run_chunked_bmm((2, 2, 2, 1))

    product_operations = {
        "current_bmm": run_current_bmm,
        "transposed_bmm": run_transposed_bmm,
        "individual_mm": run_individual_bmm,
        "bmm_4_3": run_bmm_4_3,
        "bmm_2_2_2_1": run_bmm_2_2_2_1,
    }
    padded_rows = align_up(shape.block_rows, block_row_alignment)
    if padded_rows != shape.block_rows:
        padded_left = torch.zeros(
            (Rank7Plan.rank, padded_rows, shape.block_inner),
            device=device,
            dtype=torch.float16,
        )
        padded_left[:, : shape.block_rows].copy_(left_transformed)
        padded_products = torch.empty(
            (Rank7Plan.rank, padded_rows, shape.block_columns),
            device=device,
            dtype=torch.float16,
        )
        padded_transposed_products = torch.empty(
            (Rank7Plan.rank, shape.block_columns, padded_rows),
            device=device,
            dtype=torch.float16,
        )

        def run_padded_bmm() -> None:
            torch.bmm(padded_left, right_transformed, out=padded_products)

        def run_padded_transposed_bmm() -> None:
            torch.bmm(
                right_transformed.transpose(1, 2),
                padded_left.transpose(1, 2),
                out=padded_transposed_products,
            )

        product_operations["padded_bmm"] = run_padded_bmm
        product_operations["padded_transposed_bmm"] = run_padded_transposed_bmm
    product_timings, product_order = measure_operations(
        product_operations,
        warmups,
        rounds,
    )

    def run_current_stage() -> None:
        torch.bmm(left_transformed, right_transformed, out=products)
        candidate.reconstruct(products, current_output, None)

    def run_transposed_stage() -> None:
        torch.bmm(
            right_transformed.transpose(1, 2),
            left_transformed.transpose(1, 2),
            out=transposed_products,
        )
        reconstruct_transposed_products(
            shape,
            transposed_products,
            transposed_output,
            transpose_config,
        )

    def run_individual_stage() -> None:
        run_individual_bmm()
        candidate.reconstruct(products, individual_output, None)

    def run_bmm_4_3_stage() -> None:
        run_bmm_4_3()
        candidate.reconstruct(products, individual_output, None)

    def run_bmm_2_2_2_1_stage() -> None:
        run_bmm_2_2_2_1()
        candidate.reconstruct(products, individual_output, None)

    stage_timings, stage_order = measure_operations(
        {
            "current_products": run_current_stage,
            "transposed_products": run_transposed_stage,
            "individual_products": run_individual_stage,
            "bmm_4_3_products": run_bmm_4_3_stage,
            "bmm_2_2_2_1_products": run_bmm_2_2_2_1_stage,
        },
        warmups,
        rounds,
    )
    outputs: dict[str, torch.Tensor] = {}

    def run_torch_linear() -> None:
        outputs["torch_linear"] = torch.nn.functional.linear(
            input_tensor, linear_weight
        )

    def run_current_operator() -> None:
        outputs["current_operator"] = rdna3_rank7_linear(input_tensor, linear_weight)

    def run_transposed_operator() -> None:
        dynamic_left = candidate.allocate_left()
        dynamic_right = candidate.allocate_weight()
        dynamic_products = torch.empty_like(transposed_products)
        dynamic_output = candidate.allocate_output()
        candidate.transform_left(input_tensor, dynamic_left)
        candidate.transform_weight(linear_weight, dynamic_right)
        torch.bmm(
            dynamic_right.transpose(1, 2),
            dynamic_left.transpose(1, 2),
            out=dynamic_products,
        )
        reconstruct_transposed_products(
            shape,
            dynamic_products,
            dynamic_output,
            transpose_config,
        )
        outputs["transposed_operator"] = dynamic_output

    def run_individual_operator() -> None:
        dynamic_left = candidate.allocate_left()
        dynamic_right = candidate.allocate_weight()
        dynamic_products = candidate.allocate_products()
        dynamic_output = candidate.allocate_output()
        candidate.transform_left(input_tensor, dynamic_left)
        candidate.transform_weight(linear_weight, dynamic_right)
        for rank_index in range(Rank7Plan.rank):
            torch.mm(
                dynamic_left[rank_index],
                dynamic_right[rank_index],
                out=dynamic_products[rank_index],
            )
        candidate.reconstruct(dynamic_products, dynamic_output, None)
        outputs["individual_operator"] = dynamic_output

    full_timings, full_order = measure_operations(
        {
            "torch_linear": run_torch_linear,
            "current_operator": run_current_operator,
            "transposed_operator": run_transposed_operator,
            "individual_operator": run_individual_operator,
        },
        warmups,
        rounds,
    )
    correctness = validate_outputs(
        input_tensor,
        linear_weight.T,
        None,
        outputs,
        "torch_linear",
        tile_size,
        Rank7Plan.scheme_size,
    )
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info(device)
    product_bytes = products.numel() * products.element_size()
    return {
        "shape": list(dimensions),
        "config": {
            "production": asdict(kernel_config),
            "transposed_reconstruction": asdict(transpose_config),
            "block_row_alignment": block_row_alignment,
            "padded_block_rows": padded_rows,
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "blas_backend": str(selected_blas_backend),
        },
        "memory": {
            "estimated_required_bytes": memory_bytes,
            "product_bytes": product_bytes,
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
        "product_timings": {
            name: asdict(timing) for name, timing in product_timings.items()
        },
        "product_order": product_order,
        "full_timings": {name: asdict(timing) for name, timing in full_timings.items()},
        "full_order": full_order,
        "speedup": {
            "transposed_stage_over_current": (
                stage_timings["current_products"].median_ms
                / stage_timings["transposed_products"].median_ms
            ),
            "transposed_bmm_over_current": (
                product_timings["current_bmm"].median_ms
                / product_timings["transposed_bmm"].median_ms
            ),
            "individual_bmm_over_current": (
                product_timings["current_bmm"].median_ms
                / product_timings["individual_mm"].median_ms
            ),
            "transposed_over_current": (
                full_timings["current_operator"].median_ms
                / full_timings["transposed_operator"].median_ms
            ),
            "transposed_over_torch": (
                full_timings["torch_linear"].median_ms
                / full_timings["transposed_operator"].median_ms
            ),
            "individual_stage_over_current": (
                stage_timings["current_products"].median_ms
                / stage_timings["individual_products"].median_ms
            ),
            "individual_over_current": (
                full_timings["current_operator"].median_ms
                / full_timings["individual_operator"].median_ms
            ),
            "individual_over_torch": (
                full_timings["torch_linear"].median_ms
                / full_timings["individual_operator"].median_ms
            ),
        },
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--transpose-block-rows", type=int, default=16)
    parser.add_argument("--transpose-block-columns", type=int, default=16)
    parser.add_argument("--transpose-warps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.4)
    parser.add_argument("--block-row-alignment", type=int, default=1)
    parser.add_argument(
        "--blas-backend",
        choices=("default", "hipblas", "hipblaslt", "ck"),
        default="default",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if arguments.tile_size < 1:
        parser.error("tile size must be positive")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    if (
        arguments.block_row_alignment < 1
        or arguments.block_row_alignment.bit_count() != 1
    ):
        parser.error("block row alignment must be a positive power of two")
    transpose_config = TransposeConfig(
        arguments.transpose_block_rows,
        arguments.transpose_block_columns,
        arguments.transpose_warps,
    )
    if any(
        value < 1 or value.bit_count() != 1
        for value in (transpose_config.block_rows, transpose_config.block_columns)
    ):
        parser.error("transpose block sizes must be positive powers of two")
    if transpose_config.warps not in {1, 2, 4, 8}:
        parser.error("transpose warps must be one, two, four, or eight")
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.shape,
        transpose_config,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.max_memory_fraction,
        arguments.block_row_alignment,
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
