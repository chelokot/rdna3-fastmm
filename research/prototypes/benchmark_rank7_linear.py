import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
from types import ModuleType
from typing import cast, TypeVar

import torch
import triton

from benchmarks.benchmark import measure_operations, validate_outputs


CERTIFICATE_PATH = Path("certificates/research/2x2x2_rank7_15add/certificate.json")
GENERATED_PATH = Path("research/generated/rank7_2x2x2.py")
CERTIFICATE_SHA256 = "b374fcc797ea014994453b7502625e24f70ca2f1dcf1bdbe74a716342bfefb1e"
RANK = 7
SCHEME_SIZE = 2
Result = TypeVar("Result")


@dataclass(frozen=True)
class LinearShape:
    rows: int
    inner: int
    columns: int
    block_rows: int
    block_inner: int
    block_columns: int

    @classmethod
    def from_dimensions(cls, rows: int, inner: int, columns: int) -> "LinearShape":
        return cls(
            rows=rows,
            inner=inner,
            columns=columns,
            block_rows=triton.cdiv(rows, SCHEME_SIZE),
            block_inner=triton.cdiv(inner, SCHEME_SIZE),
            block_columns=triton.cdiv(columns, SCHEME_SIZE),
        )


@dataclass(frozen=True)
class KernelConfig:
    transform_elements: int
    transform_warps: int
    weight_rows: int
    weight_columns: int
    weight_warps: int


@dataclass(frozen=True)
class MemoryEstimate:
    source_bytes: int
    transformed_bytes: int
    retained_output_bytes: int
    retained_replacement_bytes: int
    control_packed_bytes: int
    control_temporary_bytes: int
    library_workspace_bytes: int
    total_bytes: int


class Rank7Linear:
    def __init__(
        self,
        module: ModuleType,
        shape: LinearShape,
        device: torch.device,
        config: KernelConfig,
    ) -> None:
        if (
            module.DIMENSIONS != (SCHEME_SIZE, SCHEME_SIZE, SCHEME_SIZE)
            or module.RANK != RANK
            or module.CERTIFICATE_SHA256 != CERTIFICATE_SHA256
        ):
            raise ValueError("generated rank-7 module does not match the certificate")
        self.module = module
        self.shape = shape
        self.device = device
        self.config = config

    @property
    def left_transformed_shape(self) -> tuple[int, int, int]:
        return RANK, self.shape.block_rows, self.shape.block_inner

    @property
    def weight_transformed_shape(self) -> tuple[int, int, int]:
        return RANK, self.shape.block_inner, self.shape.block_columns

    @property
    def products_shape(self) -> tuple[int, int, int]:
        return RANK, self.shape.block_rows, self.shape.block_columns

    def allocate_left(self) -> torch.Tensor:
        return torch.empty(
            self.left_transformed_shape,
            device=self.device,
            dtype=torch.float16,
        )

    def allocate_weight(self) -> torch.Tensor:
        return torch.empty(
            self.weight_transformed_shape,
            device=self.device,
            dtype=torch.float16,
        )

    def allocate_products(self) -> torch.Tensor:
        return torch.empty(
            self.products_shape,
            device=self.device,
            dtype=torch.float16,
        )

    def allocate_output(self) -> torch.Tensor:
        return torch.empty(
            (self.shape.rows, self.shape.columns),
            device=self.device,
            dtype=torch.bfloat16,
        )

    def transform_left(self, source: torch.Tensor, output: torch.Tensor) -> None:
        positions = self.shape.block_rows * self.shape.block_inner
        grid = (triton.cdiv(positions, self.config.transform_elements),)
        self.module.left_transform_kernel[grid](
            source,
            output,
            self.shape.rows,
            self.shape.inner,
            self.shape.block_rows,
            self.shape.block_inner,
            self.shape.block_rows,
            self.shape.block_inner,
            self.config.transform_elements,
            num_warps=self.config.transform_warps,
            num_stages=1,
        )

    def transform_weight(self, source: torch.Tensor, output: torch.Tensor) -> None:
        grid = (
            triton.cdiv(self.shape.block_inner, self.config.weight_rows),
            triton.cdiv(self.shape.block_columns, self.config.weight_columns),
        )
        self.module.right_transform_weight_kernel[grid](
            source,
            output,
            self.shape.columns,
            self.shape.inner,
            self.shape.block_inner,
            self.shape.block_columns,
            self.config.weight_rows,
            self.config.weight_columns,
            num_warps=self.config.weight_warps,
            num_stages=1,
        )

    def reconstruct(
        self,
        products: torch.Tensor,
        output: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        positions = self.shape.block_rows * self.shape.block_columns
        grid = (triton.cdiv(positions, self.config.transform_elements),)
        kernel = (
            self.module.output_transform_kernel
            if bias is None
            else self.module.output_transform_bias_kernel
        )
        arguments = [products, output]
        if bias is not None:
            arguments.append(bias)
        kernel[grid](
            *arguments,
            self.shape.block_rows,
            self.shape.block_columns,
            self.shape.rows,
            self.shape.columns,
            self.shape.block_rows,
            self.shape.block_columns,
            self.config.transform_elements,
            num_warps=self.config.transform_warps,
            num_stages=1,
        )

    def run_transformed(
        self,
        input_tensor: torch.Tensor,
        transformed_weight: torch.Tensor,
        bias: torch.Tensor | None,
        left_transformed: torch.Tensor,
        products: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.transform_left(input_tensor, left_transformed)
        torch.bmm(left_transformed, transformed_weight, out=products)
        self.reconstruct(products, output, bias)

    def run_dynamic(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        left_transformed: torch.Tensor,
        transformed_weight: torch.Tensor,
        products: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.transform_weight(weight, transformed_weight)
        self.run_transformed(
            input_tensor,
            transformed_weight,
            bias,
            left_transformed,
            products,
            output,
        )


def parse_shape(value: str) -> tuple[int, int, int]:
    dimensions = tuple(int(part) for part in value.split(","))
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return cast(tuple[int, int, int], dimensions)


def load_generated_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("research_rank7_2x2x2", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load generated module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def validate_power_of_two(name: str, value: int) -> None:
    if value < 1 or value.bit_count() != 1:
        raise ValueError(f"{name} must be a positive power of two")


def estimate_memory(
    shape: LinearShape, has_bias: bool, modes: tuple[str, ...]
) -> MemoryEstimate:
    bfloat16_bytes = torch.bfloat16.itemsize
    float16_bytes = torch.float16.itemsize
    source_elements = (
        shape.rows * shape.inner
        + shape.columns * shape.inner
        + (shape.columns if has_bias else 0)
    )
    left_elements = RANK * shape.block_rows * shape.block_inner
    product_elements = RANK * shape.block_rows * shape.block_columns
    weight_elements = RANK * shape.block_inner * shape.block_columns
    transformed_elements = left_elements + product_elements
    if "dynamic" in modes:
        transformed_elements += weight_elements
    if "prepacked" in modes:
        transformed_elements += weight_elements
    operation_count = 1 + 2 * len(modes)
    retained_output_bytes = (
        operation_count * shape.rows * shape.columns * bfloat16_bytes
    )
    retained_replacement_bytes = shape.rows * shape.columns * bfloat16_bytes
    control_packed_bytes = 0
    if "prepacked" in modes:
        control_packed_bytes = (
            shape.columns * shape.inner + (shape.columns if has_bias else 0)
        ) * float16_bytes
    control_temporary_elements = shape.rows * shape.inner + shape.rows * shape.columns
    if "dynamic" in modes:
        control_temporary_elements += shape.columns * shape.inner
        if has_bias:
            control_temporary_elements += shape.columns
    source_bytes = source_elements * bfloat16_bytes
    transformed_bytes = transformed_elements * float16_bytes
    control_temporary_bytes = control_temporary_elements * float16_bytes
    library_workspace_bytes = 128 * 2**20
    return MemoryEstimate(
        source_bytes=source_bytes,
        transformed_bytes=transformed_bytes,
        retained_output_bytes=retained_output_bytes,
        retained_replacement_bytes=retained_replacement_bytes,
        control_packed_bytes=control_packed_bytes,
        control_temporary_bytes=control_temporary_bytes,
        library_workspace_bytes=library_workspace_bytes,
        total_bytes=(
            source_bytes
            + transformed_bytes
            + retained_output_bytes
            + retained_replacement_bytes
            + control_packed_bytes
            + control_temporary_bytes
            + library_workspace_bytes
        ),
    )


def elapsed_result(operation: Callable[[], Result]) -> tuple[Result, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def measure_control_pack(
    weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None, float]:
    def pack() -> tuple[torch.Tensor, torch.Tensor | None]:
        return (
            weight.to(torch.float16),
            None if bias is None else bias.to(torch.float16),
        )

    warmup_weight, warmup_bias = pack()
    torch.cuda.synchronize()
    del warmup_weight, warmup_bias
    torch.cuda.empty_cache()
    (packed_weight, packed_bias), milliseconds = elapsed_result(pack)
    return packed_weight, packed_bias, milliseconds


def measure_rank7_pack(
    candidate: Rank7Linear, weight: torch.Tensor
) -> tuple[torch.Tensor, float]:
    def pack() -> torch.Tensor:
        transformed = candidate.allocate_weight()
        candidate.transform_weight(weight, transformed)
        return transformed

    warmup = pack()
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    transformed, milliseconds = elapsed_result(pack)
    return transformed, milliseconds


def break_even_uses(
    packing_ms: float, baseline_ms: float, candidate_ms: float
) -> float | None:
    saved_ms = baseline_ms - candidate_ms
    return packing_ms / saved_ms if saved_ms > 0 else None


def benchmark(
    shape_tuple: tuple[int, int, int],
    has_bias: bool,
    modes: tuple[str, ...],
    config: KernelConfig,
    warmups: int,
    rounds: int,
    tile_size: int,
    max_memory_fraction: float,
    seed: int,
) -> dict[str, object]:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("the rank-7 research benchmark requires ROCm")
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    if architecture != "gfx1100":
        raise RuntimeError(f"expected gfx1100, found {architecture or 'unknown'}")
    shape = LinearShape.from_dimensions(*shape_tuple)
    if tile_size > min(shape.block_rows, shape.block_columns):
        raise ValueError("sample tile must fit inside every output macroblock")
    memory = estimate_memory(shape, has_bias, modes)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    budget_bytes = int(free_before * max_memory_fraction)
    if memory.total_bytes > budget_bytes:
        raise MemoryError(
            f"benchmark needs an estimated {memory.total_bytes / 2**30:.2f} GiB, "
            f"exceeding the {budget_bytes / 2**30:.2f} GiB budget"
        )
    module = load_generated_module(GENERATED_PATH)
    candidate = Rank7Linear(module, shape, device, config)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    input_tensor = torch.randn(
        (shape.rows, shape.inner), device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(
        (shape.columns, shape.inner), device=device, dtype=torch.bfloat16
    )
    bias = (
        torch.randn((shape.columns,), device=device, dtype=torch.bfloat16)
        if has_bias
        else None
    )
    left_transformed = candidate.allocate_left()
    products = candidate.allocate_products()
    dynamic_weight = candidate.allocate_weight() if "dynamic" in modes else None
    dynamic_output = candidate.allocate_output() if "dynamic" in modes else None
    packed_output = candidate.allocate_output() if "prepacked" in modes else None
    control_packed_weight: torch.Tensor | None = None
    control_packed_bias: torch.Tensor | None = None
    rank7_packed_weight: torch.Tensor | None = None
    packing: dict[str, dict[str, float | int | None]] = {}
    if "prepacked" in modes:
        control_packed_weight, control_packed_bias, control_pack_ms = (
            measure_control_pack(weight, bias)
        )
        rank7_packed_weight, rank7_pack_ms = measure_rank7_pack(candidate, weight)
        source_weight_bytes = weight.numel() * weight.element_size()
        packing = {
            "fp16_control": {
                "milliseconds": control_pack_ms,
                "bytes": control_packed_weight.numel()
                * control_packed_weight.element_size(),
                "weight_expansion": (
                    control_packed_weight.numel()
                    * control_packed_weight.element_size()
                    / source_weight_bytes
                ),
            },
            "rank7": {
                "milliseconds": rank7_pack_ms,
                "bytes": rank7_packed_weight.numel()
                * rank7_packed_weight.element_size(),
                "weight_expansion": (
                    rank7_packed_weight.numel()
                    * rank7_packed_weight.element_size()
                    / source_weight_bytes
                ),
            },
        }
    outputs: dict[str, torch.Tensor] = {}

    def run_native_bfloat16() -> None:
        outputs["torch_bfloat16_linear"] = torch.nn.functional.linear(
            input_tensor, weight, bias
        )

    operations = {"torch_bfloat16_linear": run_native_bfloat16}
    if "dynamic" in modes:
        if dynamic_weight is None or dynamic_output is None:
            raise RuntimeError("dynamic buffers were not allocated")

        def run_control_dynamic() -> None:
            control_input = input_tensor.to(torch.float16)
            control_weight = weight.to(torch.float16)
            control_bias = None if bias is None else bias.to(torch.float16)
            outputs["fp16_control_dynamic"] = torch.nn.functional.linear(
                control_input, control_weight, control_bias
            ).to(torch.bfloat16)

        def run_rank7_dynamic() -> None:
            candidate.run_dynamic(
                input_tensor,
                weight,
                bias,
                left_transformed,
                dynamic_weight,
                products,
                dynamic_output,
            )
            outputs["rank7_dynamic"] = dynamic_output

        operations["fp16_control_dynamic"] = run_control_dynamic
        operations["rank7_dynamic"] = run_rank7_dynamic
    if "prepacked" in modes:
        if (
            control_packed_weight is None
            or rank7_packed_weight is None
            or packed_output is None
        ):
            raise RuntimeError("prepacked buffers were not allocated")

        def run_control_prepacked() -> None:
            control_input = input_tensor.to(torch.float16)
            outputs["fp16_control_prepacked"] = torch.nn.functional.linear(
                control_input, control_packed_weight, control_packed_bias
            ).to(torch.bfloat16)

        def run_rank7_prepacked() -> None:
            candidate.run_transformed(
                input_tensor,
                rank7_packed_weight,
                bias,
                left_transformed,
                products,
                packed_output,
            )
            outputs["rank7_prepacked"] = packed_output

        operations["fp16_control_prepacked"] = run_control_prepacked
        operations["rank7_prepacked"] = run_rank7_prepacked
    timings, order = measure_operations(operations, warmups, rounds)
    correctness = validate_outputs(
        input_tensor,
        weight.T,
        bias,
        outputs,
        "torch_bfloat16_linear",
        tile_size,
        SCHEME_SIZE,
    )
    torch.cuda.synchronize()
    native_ms = timings["torch_bfloat16_linear"].median_ms
    speedups = {
        name: native_ms / timing.median_ms
        for name, timing in timings.items()
        if name != "torch_bfloat16_linear"
    }
    if "prepacked" in modes:
        packing["fp16_control"]["break_even_uses_vs_native"] = break_even_uses(
            control_pack_ms,
            native_ms,
            timings["fp16_control_prepacked"].median_ms,
        )
        packing["rank7"]["break_even_uses_vs_native"] = break_even_uses(
            rank7_pack_ms,
            native_ms,
            timings["rank7_prepacked"].median_ms,
        )
    return {
        "schema_version": 1,
        "status": "research-only-buffer-reuse-not-dispatched",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "shape": list(shape_tuple),
        "linear_contract": {
            "input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "weight_layout": "contiguous[N,K]",
            "leaf_compute_dtype": "float16",
            "output_dtype": "bfloat16",
            "bias": has_bias,
        },
        "timings": {name: asdict(timing) for name, timing in timings.items()},
        "speedup_vs_torch_bfloat16_linear": speedups,
        "packing": packing,
        "correctness": correctness,
        "memory": {
            "estimate": asdict(memory),
            "free_before_bytes": free_before,
            "total_device_bytes": total_memory,
            "budget_bytes": budget_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
        "protocol": {
            "modes": list(modes),
            "warmups": warmups,
            "rounds": rounds,
            "alternating_order": order,
            "sample_tile_size": tile_size,
            "seed": seed,
            "max_memory_fraction": max_memory_fraction,
            "kernel_config": asdict(config),
            "timing_contract": {
                "native_and_fp16_controls": (
                    "allocate conversion and output tensors during each invocation"
                ),
                "rank7_candidates": (
                    "reuse preallocated transformed, product, and output buffers"
                ),
                "interpretation": (
                    "plan-level screening only; speedups are not allocation-equivalent"
                ),
            },
            "fp16_control_dynamic": (
                "cast BF16 input, weight, and bias to FP16 per invocation; "
                "run ordinary F.linear; cast output to BF16"
            ),
            "fp16_control_prepacked": (
                "retain FP16 weight and bias; cast input per invocation; run "
                "ordinary F.linear; cast output to BF16"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "device": properties.name,
            "architecture": architecture,
            "compute_units": properties.multi_processor_count,
        },
        "artifacts": {
            "certificate": str(CERTIFICATE_PATH),
            "certificate_sha256": file_sha256(CERTIFICATE_PATH),
            "generated_module": str(GENERATED_PATH),
            "generated_module_sha256": file_sha256(GENERATED_PATH),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output("status", "--short"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument(
        "--mode", choices=("dynamic", "prepacked", "both"), default="both"
    )
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-memory-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--transform-elements", type=int, default=256)
    parser.add_argument("--transform-warps", type=int, default=2)
    parser.add_argument("--weight-rows", type=int, default=8)
    parser.add_argument("--weight-columns", type=int, default=512)
    parser.add_argument("--weight-warps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 1 <= arguments.warmups <= 5:
        parser.error("warmups must be between one and five")
    if not 3 <= arguments.rounds <= 15:
        parser.error("rounds must be between three and fifteen")
    if not 1 <= arguments.tile_size <= 64:
        parser.error("tile size must be between one and 64")
    if not 0 < arguments.max_memory_fraction <= 0.5:
        parser.error("memory fraction must be in the interval (0, 0.5]")
    config = KernelConfig(
        transform_elements=arguments.transform_elements,
        transform_warps=arguments.transform_warps,
        weight_rows=arguments.weight_rows,
        weight_columns=arguments.weight_columns,
        weight_warps=arguments.weight_warps,
    )
    for name, value in (
        ("transform elements", config.transform_elements),
        ("weight rows", config.weight_rows),
        ("weight columns", config.weight_columns),
    ):
        try:
            validate_power_of_two(name, value)
        except ValueError as error:
            parser.error(str(error))
    if config.transform_warps not in (1, 2, 4, 8) or config.weight_warps not in (
        1,
        2,
        4,
        8,
    ):
        parser.error("warp counts must be one, two, four, or eight")
    modes = ("dynamic", "prepacked") if arguments.mode == "both" else (arguments.mode,)
    torch.set_grad_enabled(False)
    report = benchmark(
        arguments.shape,
        arguments.bias,
        modes,
        config,
        arguments.warmups,
        arguments.rounds,
        arguments.tile_size,
        arguments.max_memory_fraction,
        arguments.seed,
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)


if __name__ == "__main__":
    main()
