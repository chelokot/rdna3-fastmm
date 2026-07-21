import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import cast

import torch
import triton


@dataclass(frozen=True)
class KernelConfig:
    block_elements: int
    num_warps: int


@dataclass(frozen=True)
class TransposeKernelConfig:
    block_rows: int
    block_columns: int
    num_warps: int


@dataclass(frozen=True)
class TimingSummary:
    median_ms: float
    first_quartile_ms: float
    third_quartile_ms: float


@dataclass(frozen=True)
class CorrectnessSummary:
    sample_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    relative_l2_error: float


@dataclass(frozen=True)
class BenchmarkResult:
    shape: tuple[int, int, int]
    dimensions: tuple[int, int, int]
    rank: int
    layout: str
    products_transposed: bool
    mode: str
    baseline: TimingSummary
    candidate: TimingSummary
    speedup: float
    baseline_tflops: float
    effective_candidate_tflops: float
    padded_leaf_flop_ratio: float
    allocated_bytes: int
    correctness: CorrectnessSummary


def parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or any(dimension < 1 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return cast(tuple[int, int, int], dimensions)


def load_generated_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("generated_fastmm", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quartile(values: list[float], numerator: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * numerator / 4
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> TimingSummary:
    return TimingSummary(
        median_ms=median(values),
        first_quartile_ms=quartile(values, 1),
        third_quartile_ms=quartile(values, 3),
    )


def validate_kernel_config(name: str, config: KernelConfig) -> None:
    if config.block_elements < 1 or config.block_elements.bit_count() != 1:
        raise ValueError(f"{name} block elements must be a positive power of two")
    if config.num_warps not in (1, 2, 4, 8):
        raise ValueError(f"{name} warps must be one, two, four, or eight")


def elapsed_ms(operation: Callable[[], None]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


class GeneratedFastMM:
    def __init__(
        self,
        module: ModuleType,
        left: torch.Tensor,
        right: torch.Tensor,
        left_config: KernelConfig,
        right_config: KernelConfig,
        output_config: KernelConfig,
        transpose_config: TransposeKernelConfig,
        layout: str,
        transpose_products: bool,
    ) -> None:
        self.module = module
        self.dimensions = cast(tuple[int, int, int], module.DIMENSIONS)
        self.rank = cast(int, module.RANK)
        self.left = left
        self.right = right
        self.left_config = left_config
        self.right_config = right_config
        self.output_config = output_config
        self.transpose_config = transpose_config
        self.layout = layout
        self.transpose_products = transpose_products
        first_size, shared_size, second_size = self.dimensions
        self.block_height = triton.cdiv(left.shape[0], first_size)
        self.block_shared = triton.cdiv(left.shape[1], shared_size)
        self.block_width = triton.cdiv(right.shape[1], second_size)
        left_shape = (
            (self.rank, self.block_shared, self.block_height)
            if layout[0] == "t"
            else (self.rank, self.block_height, self.block_shared)
        )
        right_shape = (
            (self.rank, self.block_width, self.block_shared)
            if layout[1] == "t"
            else (self.rank, self.block_shared, self.block_width)
        )
        self.left_storage = torch.empty(
            left_shape,
            dtype=left.dtype,
            device=left.device,
        )
        self.right_storage = torch.empty(
            right_shape,
            dtype=right.dtype,
            device=right.device,
        )
        self.left_transformed = (
            self.left_storage.transpose(1, 2) if layout[0] == "t" else self.left_storage
        )
        self.right_transformed = (
            self.right_storage.transpose(1, 2)
            if layout[1] == "t"
            else self.right_storage
        )
        product_shape = (
            (self.rank, self.block_width, self.block_height)
            if transpose_products
            else (self.rank, self.block_height, self.block_width)
        )
        self.products = torch.empty(
            product_shape,
            dtype=left.dtype,
            device=left.device,
        )
        self.output = torch.empty(
            (left.shape[0], right.shape[1]), dtype=left.dtype, device=left.device
        )

    def transform_left(self) -> None:
        if self.layout[0] == "t":
            config = self.transpose_config
            grid = (
                triton.cdiv(self.block_height, config.block_rows),
                triton.cdiv(self.block_shared, config.block_columns),
            )
            self.module.left_transform_transposed_kernel[grid](
                self.left,
                self.left_storage,
                self.left.shape[0],
                self.left.shape[1],
                self.left_transformed.shape[1],
                self.left_transformed.shape[2],
                self.block_height,
                self.block_shared,
                config.block_rows,
                config.block_columns,
                num_warps=config.num_warps,
                num_stages=1,
            )
            return
        config = self.left_config
        plane_elements = self.block_height * self.block_shared
        grid = (triton.cdiv(plane_elements, config.block_elements),)
        self.module.left_transform_kernel[grid](
            self.left,
            self.left_storage,
            self.left.shape[0],
            self.left.shape[1],
            self.left_transformed.shape[1],
            self.left_transformed.shape[2],
            self.block_height,
            self.block_shared,
            config.block_elements,
            num_warps=config.num_warps,
            num_stages=1,
        )

    def transform_right(self) -> None:
        if self.layout[1] == "t":
            config = self.transpose_config
            grid = (
                triton.cdiv(self.block_shared, config.block_rows),
                triton.cdiv(self.block_width, config.block_columns),
            )
            self.module.right_transform_transposed_kernel[grid](
                self.right,
                self.right_storage,
                self.right.shape[0],
                self.right.shape[1],
                self.right_transformed.shape[1],
                self.right_transformed.shape[2],
                self.block_shared,
                self.block_width,
                config.block_rows,
                config.block_columns,
                num_warps=config.num_warps,
                num_stages=1,
            )
            return
        config = self.right_config
        plane_elements = self.block_shared * self.block_width
        grid = (triton.cdiv(plane_elements, config.block_elements),)
        self.module.right_transform_kernel[grid](
            self.right,
            self.right_storage,
            self.right.shape[0],
            self.right.shape[1],
            self.right_transformed.shape[1],
            self.right_transformed.shape[2],
            self.block_shared,
            self.block_width,
            config.block_elements,
            num_warps=config.num_warps,
            num_stages=1,
        )

    def multiply(self) -> None:
        if self.transpose_products:
            torch.bmm(
                self.right_transformed.transpose(1, 2),
                self.left_transformed.transpose(1, 2),
                out=self.products,
            )
            return
        torch.bmm(self.left_transformed, self.right_transformed, out=self.products)

    def reconstruct(self) -> None:
        if self.transpose_products:
            config = self.transpose_config
            grid = (
                triton.cdiv(self.block_height, config.block_rows),
                triton.cdiv(self.block_width, config.block_columns),
            )
            self.module.output_transform_transposed_products_kernel[grid](
                self.products,
                self.output,
                self.output.shape[0],
                self.output.shape[1],
                self.block_height,
                self.block_width,
                config.block_rows,
                config.block_columns,
                num_warps=config.num_warps,
                num_stages=1,
            )
            return
        config = self.output_config
        plane_elements = self.block_height * self.block_width
        grid = (triton.cdiv(plane_elements, config.block_elements),)
        self.module.output_transform_kernel[grid](
            self.products,
            self.output,
            self.products.shape[1],
            self.products.shape[2],
            self.output.shape[0],
            self.output.shape[1],
            self.block_height,
            self.block_width,
            config.block_elements,
            num_warps=config.num_warps,
            num_stages=1,
        )

    def dynamic(self) -> None:
        self.transform_left()
        self.transform_right()
        self.multiply()
        self.reconstruct()

    def prepacked(self) -> None:
        self.transform_left()
        self.multiply()
        self.reconstruct()


def allocation_bytes(
    shape: tuple[int, int, int],
    dimensions: tuple[int, int, int],
    rank: int,
    element_size: int,
) -> int:
    size_m, size_k, size_n = shape
    first_size, shared_size, second_size = dimensions
    block_height = triton.cdiv(size_m, first_size)
    block_shared = triton.cdiv(size_k, shared_size)
    block_width = triton.cdiv(size_n, second_size)
    elements = (
        size_m * size_k
        + size_k * size_n
        + 2 * size_m * size_n
        + rank
        * (
            block_height * block_shared
            + block_shared * block_width
            + block_height * block_width
        )
    )
    return elements * element_size


def padded_leaf_flop_ratio(
    shape: tuple[int, int, int],
    dimensions: tuple[int, int, int],
    rank: int,
) -> float:
    size_m, size_k, size_n = shape
    first_size, shared_size, second_size = dimensions
    block_height = triton.cdiv(size_m, first_size)
    block_shared = triton.cdiv(size_k, shared_size)
    block_width = triton.cdiv(size_n, second_size)
    return rank * block_height * block_shared * block_width / (size_m * size_k * size_n)


def validate_output(
    baseline_output: torch.Tensor,
    candidate_output: torch.Tensor,
    sample_limit: int,
) -> CorrectnessSummary:
    sample_count = min(sample_limit, baseline_output.numel())
    reference = baseline_output.flatten()[:sample_count].float()
    candidate = candidate_output.flatten()[:sample_count].float()
    difference = torch.abs(candidate - reference)
    if not torch.isfinite(candidate).all().item():
        raise ValueError("candidate produced non-finite sampled output")
    return CorrectnessSummary(
        sample_count=sample_count,
        maximum_absolute_error=float(torch.max(difference).item()),
        mean_absolute_error=float(torch.mean(difference).item()),
        relative_l2_error=float(
            (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference)
            ).item()
        ),
    )


def benchmark_pair(
    baseline: Callable[[], None],
    candidate: Callable[[], None],
    warmups: int,
    blocks: int,
) -> tuple[TimingSummary, TimingSummary]:
    for _ in range(warmups):
        baseline()
        candidate()
    torch.cuda.synchronize()
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for block in range(blocks):
        operations = (
            ((baseline, baseline_times), (candidate, candidate_times))
            if block % 2 == 0
            else ((candidate, candidate_times), (baseline, baseline_times))
        )
        for operation, timings in operations:
            timings.append(elapsed_ms(operation))
    return summarize(baseline_times), summarize(candidate_times)


def benchmark_shape(
    module: ModuleType,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    modes: tuple[str, ...],
    left_config: KernelConfig,
    right_config: KernelConfig,
    output_config: KernelConfig,
    transpose_config: TransposeKernelConfig,
    layout: str,
    transpose_products: bool,
    warmups: int,
    blocks: int,
    sample_limit: int,
    max_memory_fraction: float,
) -> list[BenchmarkResult]:
    size_m, size_k, size_n = shape
    dimensions = cast(tuple[int, int, int], module.DIMENSIONS)
    rank = cast(int, module.RANK)
    element_size = torch.empty((), dtype=dtype).element_size()
    required_bytes = allocation_bytes(shape, dimensions, rank, element_size)
    free_bytes, _ = torch.cuda.mem_get_info()
    if required_bytes > free_bytes * max_memory_fraction:
        raise MemoryError(
            f"shape requires {required_bytes / 2**30:.2f} GiB but benchmark budget is "
            f"{free_bytes * max_memory_fraction / 2**30:.2f} GiB"
        )
    left = torch.randn((size_m, size_k), device="cuda", dtype=dtype)
    right = torch.randn((size_k, size_n), device="cuda", dtype=dtype)
    baseline_output = torch.empty((size_m, size_n), device="cuda", dtype=dtype)
    fastmm = GeneratedFastMM(
        module,
        left,
        right,
        left_config,
        right_config,
        output_config,
        transpose_config,
        layout,
        transpose_products,
    )
    fastmm.transform_right()
    torch.cuda.synchronize()

    def baseline() -> None:
        torch.mm(left, right, out=baseline_output)

    baseline()
    fastmm.dynamic()
    torch.cuda.synchronize()
    correctness = validate_output(baseline_output, fastmm.output, sample_limit)
    results: list[BenchmarkResult] = []
    operation_count = 2 * size_m * size_k * size_n
    for mode in modes:
        candidate = fastmm.dynamic if mode == "dynamic" else fastmm.prepacked
        baseline_timing, candidate_timing = benchmark_pair(
            baseline, candidate, warmups, blocks
        )
        baseline_tflops = operation_count / baseline_timing.median_ms / 1e9
        effective_candidate_tflops = operation_count / candidate_timing.median_ms / 1e9
        results.append(
            BenchmarkResult(
                shape=shape,
                dimensions=fastmm.dimensions,
                rank=fastmm.rank,
                layout=layout,
                products_transposed=transpose_products,
                mode=mode,
                baseline=baseline_timing,
                candidate=candidate_timing,
                speedup=baseline_timing.median_ms / candidate_timing.median_ms,
                baseline_tflops=baseline_tflops,
                effective_candidate_tflops=effective_candidate_tflops,
                padded_leaf_flop_ratio=padded_leaf_flop_ratio(
                    shape, fastmm.dimensions, fastmm.rank
                ),
                allocated_bytes=required_bytes,
                correctness=correctness,
            )
        )
    return results


def dtype_from_name(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--mode", choices=("dynamic", "prepacked", "both"), default="both"
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--left-block-elements", type=int, default=32)
    parser.add_argument("--left-warps", type=int, default=4)
    parser.add_argument("--right-block-elements", type=int, default=64)
    parser.add_argument("--right-warps", type=int, default=8)
    parser.add_argument("--output-block-elements", type=int, default=128)
    parser.add_argument("--output-warps", type=int, default=4)
    parser.add_argument("--transpose-block-rows", type=int, default=16)
    parser.add_argument("--transpose-block-columns", type=int, default=16)
    parser.add_argument("--transpose-warps", type=int, default=4)
    parser.add_argument("--layout", choices=("nn", "nt", "tn", "tt"), default="nn")
    parser.add_argument("--transpose-products", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=65536)
    parser.add_argument("--max-memory-fraction", type=float, default=0.75)
    arguments = parser.parse_args()
    if arguments.warmups < 1 or arguments.blocks < 3:
        parser.error("warmups must be positive and blocks must be at least three")
    if arguments.sample_limit < 1:
        parser.error("sample limit must be positive")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    module = load_generated_module(arguments.module)
    modes = ("dynamic", "prepacked") if arguments.mode == "both" else (arguments.mode,)
    left_config = KernelConfig(arguments.left_block_elements, arguments.left_warps)
    right_config = KernelConfig(arguments.right_block_elements, arguments.right_warps)
    output_config = KernelConfig(
        arguments.output_block_elements, arguments.output_warps
    )
    transpose_config = TransposeKernelConfig(
        arguments.transpose_block_rows,
        arguments.transpose_block_columns,
        arguments.transpose_warps,
    )
    for name, config in (
        ("left", left_config),
        ("right", right_config),
        ("output", output_config),
    ):
        validate_kernel_config(name, config)
    validate_kernel_config(
        "transpose rows",
        KernelConfig(transpose_config.block_rows, transpose_config.num_warps),
    )
    validate_kernel_config(
        "transpose columns",
        KernelConfig(transpose_config.block_columns, transpose_config.num_warps),
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    results: list[BenchmarkResult] = []
    for shape in arguments.shape:
        results.extend(
            benchmark_shape(
                module,
                shape,
                dtype_from_name(arguments.dtype),
                modes,
                left_config,
                right_config,
                output_config,
                transpose_config,
                arguments.layout,
                arguments.transpose_products,
                arguments.warmups,
                arguments.blocks,
                arguments.sample_limit,
                arguments.max_memory_fraction,
            )
        )
        torch.cuda.empty_cache()
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
