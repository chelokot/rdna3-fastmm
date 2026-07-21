from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

import torch
import triton

from rdna3_fastmm.generated import rank49_4x4x4, rank343_8x8x8


def _validate_generated_module(
    module: ModuleType,
    dimensions: tuple[int, int, int],
    rank: int,
    certificate_sha256: str,
) -> None:
    if (
        module.DIMENSIONS != dimensions
        or module.RANK != rank
        or module.CERTIFICATE_SHA256 != certificate_sha256
    ):
        raise RuntimeError("generated kernel metadata does not match the backend")


_validate_generated_module(
    rank49_4x4x4,
    (4, 4, 4),
    49,
    "a3c4121dfd09607045255628dd94b60133c24c46b65f1089e89c6564ba522561",
)
_validate_generated_module(
    rank343_8x8x8,
    (8, 8, 8),
    343,
    "c7afccb09f8491e484af1696c25c809757c4af649ee7220e79f25d1c7afe3955",
)


@dataclass(frozen=True)
class MatrixShape:
    rows: int
    inner: int
    columns: int
    block_rows: int
    block_inner: int
    block_columns: int

    @classmethod
    def from_dimensions(
        cls, rows: int, inner: int, columns: int, scheme_size: int
    ) -> "MatrixShape":
        if min(rows, inner, columns) < 1:
            raise ValueError("matrix dimensions must be positive")
        return cls(
            rows=rows,
            inner=inner,
            columns=columns,
            block_rows=triton.cdiv(rows, scheme_size),
            block_inner=triton.cdiv(inner, scheme_size),
            block_columns=triton.cdiv(columns, scheme_size),
        )

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return self.rows, self.inner, self.columns

    def workspace_elements(self, rank: int) -> int:
        return rank * (
            self.block_rows * self.block_inner
            + self.block_inner * self.block_columns
            + self.block_rows * self.block_columns
        )


@dataclass
class Workspace:
    algorithm: str
    output_dtype: torch.dtype
    compute_dtype: torch.dtype
    left_transformed: torch.Tensor
    right_transformed: torch.Tensor | None
    products: torch.Tensor


@dataclass(frozen=True)
class PackedRight:
    algorithm: str
    source_dtype: torch.dtype
    compute_dtype: torch.dtype
    transformed: torch.Tensor
    source_shape: tuple[int, int]


@dataclass(frozen=True)
class LinearShapeFamily:
    minimum_rows: int
    maximum_rows: int
    inner: int
    columns: int
    has_bias: bool

    def matches(self, shape: tuple[int, int, int], has_bias: bool) -> bool:
        rows, inner, columns = shape
        return (
            self.minimum_rows <= rows <= self.maximum_rows
            and inner == self.inner
            and columns == self.columns
            and has_bias == self.has_bias
        )


@dataclass(frozen=True)
class WeightTransformConfig:
    block_rows: int
    block_columns: int
    warps: int


class _Plan(ABC):
    algorithm: str
    rank: int
    scheme_size: int
    precision_pairs: frozenset[tuple[torch.dtype, torch.dtype]]
    dynamic_shapes: frozenset[tuple[int, int, int]]
    prepacked_shapes: frozenset[tuple[int, int, int]]

    def __init__(
        self,
        rows: int,
        inner: int,
        columns: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if (
            device.type != "cuda"
            or torch.version.hip is None
            or not torch.cuda.is_available()
        ):
            raise ValueError("the RDNA3 backend requires a ROCm device")
        resolved_compute_dtype = dtype if compute_dtype is None else compute_dtype
        if (dtype, resolved_compute_dtype) not in self.precision_pairs:
            raise ValueError(
                f"{self.algorithm} does not support {dtype} inputs with "
                f"{resolved_compute_dtype} leaf products"
            )
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        self.device = torch.device("cuda", device_index)
        self.dtype = dtype
        self.compute_dtype = resolved_compute_dtype
        self.shape = MatrixShape.from_dimensions(rows, inner, columns, self.scheme_size)

    @property
    def workspace_bytes(self) -> int:
        return self._workspace_bytes(prepacked_right=False)

    @property
    def prepacked_workspace_bytes(self) -> int:
        return self._workspace_bytes(prepacked_right=True)

    @property
    def packed_right_bytes(self) -> int:
        shape = self.shape
        return (
            self.rank
            * shape.block_inner
            * shape.block_columns
            * self.compute_dtype.itemsize
        )

    def is_recommended(self, *, prepacked_right: bool = False) -> bool:
        shapes = self.prepacked_shapes if prepacked_right else self.dynamic_shapes
        return (
            self.dtype == torch.float16
            and self.compute_dtype == torch.float16
            and self.shape.dimensions in shapes
            and is_tested_runtime(self.device)
        )

    def allocate_workspace(
        self,
        max_free_memory_fraction: float = 0.75,
        *,
        prepacked_right: bool = False,
    ) -> Workspace:
        workspace_bytes = self._workspace_bytes(prepacked_right)
        self._ensure_memory_budget(
            workspace_bytes,
            max_free_memory_fraction,
            f"{self.algorithm} workspace",
        )
        shape = self.shape
        return Workspace(
            algorithm=self.algorithm,
            output_dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            left_transformed=torch.empty(
                (self.rank, shape.block_rows, shape.block_inner),
                device=self.device,
                dtype=self.compute_dtype,
            ),
            right_transformed=(
                None
                if prepacked_right
                else torch.empty(
                    (self.rank, shape.block_inner, shape.block_columns),
                    device=self.device,
                    dtype=self.compute_dtype,
                )
            ),
            products=torch.empty(
                (self.rank, shape.block_rows, shape.block_columns),
                device=self.device,
                dtype=self.compute_dtype,
            ),
        )

    def pack_right(
        self, right: torch.Tensor, max_free_memory_fraction: float = 0.75
    ) -> PackedRight:
        self._validate_right(right)
        self._ensure_memory_budget(
            self.packed_right_bytes,
            max_free_memory_fraction,
            f"{self.algorithm} packed right matrix",
        )
        shape = self.shape
        transformed = torch.empty(
            (self.rank, shape.block_inner, shape.block_columns),
            device=self.device,
            dtype=self.compute_dtype,
        )
        self._transform_right(right, transformed)
        return PackedRight(
            algorithm=self.algorithm,
            source_dtype=self.dtype,
            compute_dtype=self.compute_dtype,
            transformed=transformed,
            source_shape=(shape.inner, shape.columns),
        )

    def run(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        workspace: Workspace,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_left(left)
        self._validate_right(right)
        self._validate_workspace(workspace, require_right=True)
        result = self._output(output)
        self._validate_output_storage(
            result,
            left,
            right,
            workspace.left_transformed,
            workspace.products,
            workspace.right_transformed,
        )
        self._transform_left(left, workspace.left_transformed)
        right_transformed = workspace.right_transformed
        if right_transformed is None:
            raise ValueError("dynamic execution requires a dynamic workspace")
        self._transform_right(right, right_transformed)
        torch.bmm(
            workspace.left_transformed,
            right_transformed,
            out=workspace.products,
        )
        self._reconstruct(workspace.products, result)
        return result

    def run_packed(
        self,
        left: torch.Tensor,
        right: PackedRight,
        workspace: Workspace,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_left(left)
        self._validate_packed_right(right)
        self._validate_workspace(workspace, require_right=False)
        result = self._output(output)
        self._validate_separate_storage(
            right.transformed,
            workspace.left_transformed,
            workspace.products,
        )
        self._validate_output_storage(
            result,
            left,
            right.transformed,
            workspace.left_transformed,
            workspace.products,
            workspace.right_transformed,
        )
        self._transform_left(left, workspace.left_transformed)
        torch.bmm(
            workspace.left_transformed,
            right.transformed,
            out=workspace.products,
        )
        self._reconstruct(workspace.products, result)
        return result

    @abstractmethod
    def _transform_left(self, source: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def _transform_right(self, source: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def _reconstruct(self, products: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError

    def _validate_left(self, tensor: torch.Tensor) -> None:
        self._validate_tensor(tensor, (self.shape.rows, self.shape.inner), "left")

    def _validate_right(self, tensor: torch.Tensor) -> None:
        self._validate_tensor(tensor, (self.shape.inner, self.shape.columns), "right")

    def _validate_weight(self, tensor: torch.Tensor) -> None:
        self._validate_tensor(tensor, (self.shape.columns, self.shape.inner), "weight")

    def _validate_bias(self, tensor: torch.Tensor) -> None:
        self._validate_tensor(tensor, (self.shape.columns,), "bias")

    def _validate_tensor(
        self, tensor: torch.Tensor, shape: tuple[int, ...], name: str
    ) -> None:
        if tensor.shape != shape:
            raise ValueError(
                f"{name} matrix has shape {tuple(tensor.shape)}, expected {shape}"
            )
        if tensor.device != self.device or tensor.dtype != self.dtype:
            raise ValueError(f"{name} matrix device or dtype does not match the plan")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} matrix must be contiguous")
        if tensor.requires_grad:
            raise ValueError("the experimental RDNA3 backend does not support autograd")

    def _validate_workspace(self, workspace: Workspace, *, require_right: bool) -> None:
        if (
            workspace.algorithm != self.algorithm
            or workspace.output_dtype != self.dtype
            or workspace.compute_dtype != self.compute_dtype
        ):
            raise ValueError("workspace does not match the plan")
        shape = self.shape
        expected = [
            (
                workspace.left_transformed,
                (self.rank, shape.block_rows, shape.block_inner),
            ),
            (
                workspace.products,
                (self.rank, shape.block_rows, shape.block_columns),
            ),
        ]
        if require_right:
            right_transformed = workspace.right_transformed
            if right_transformed is None:
                raise ValueError("dynamic execution requires a dynamic workspace")
            expected.append(
                (
                    right_transformed,
                    (self.rank, shape.block_inner, shape.block_columns),
                )
            )
        for tensor, tensor_shape in expected:
            if (
                tensor.shape != tensor_shape
                or tensor.device != self.device
                or tensor.dtype != self.compute_dtype
                or not tensor.is_contiguous()
                or tensor.requires_grad
            ):
                raise ValueError("workspace does not match the plan")
        self._validate_separate_storage(*(tensor for tensor, _ in expected))

    def _validate_packed_right(self, packed: PackedRight) -> None:
        shape = self.shape
        if (
            packed.algorithm != self.algorithm
            or packed.source_dtype != self.dtype
            or packed.compute_dtype != self.compute_dtype
        ):
            raise ValueError("packed right matrix does not match the plan")
        if packed.source_shape != (shape.inner, shape.columns):
            raise ValueError("packed right matrix shape does not match the plan")
        transformed = packed.transformed
        if (
            transformed.shape != (self.rank, shape.block_inner, shape.block_columns)
            or transformed.device != self.device
            or transformed.dtype != self.compute_dtype
            or not transformed.is_contiguous()
            or transformed.requires_grad
        ):
            raise ValueError("packed right matrix does not match the plan")

    def _output(self, output: torch.Tensor | None) -> torch.Tensor:
        shape = (self.shape.rows, self.shape.columns)
        if output is None:
            return torch.empty(shape, device=self.device, dtype=self.dtype)
        self._validate_tensor(output, shape, "output")
        return output

    def _workspace_bytes(self, prepacked_right: bool) -> int:
        shape = self.shape
        elements = self.rank * (
            shape.block_rows * shape.block_inner
            + shape.block_rows * shape.block_columns
        )
        if not prepacked_right:
            elements += self.rank * shape.block_inner * shape.block_columns
        return elements * self.compute_dtype.itemsize

    def _ensure_memory_budget(
        self, required_bytes: int, max_free_memory_fraction: float, name: str
    ) -> None:
        if not 0 < max_free_memory_fraction <= 1:
            raise ValueError("memory fraction must be in the interval (0, 1]")
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        budget_bytes = free_bytes * max_free_memory_fraction
        if required_bytes > budget_bytes:
            raise MemoryError(
                f"{name} needs {required_bytes / 2**30:.2f} GiB, "
                f"exceeding the {budget_bytes / 2**30:.2f} GiB budget"
            )

    def _validate_output_storage(
        self, output: torch.Tensor, *tensors: torch.Tensor | None
    ) -> None:
        for tensor in tensors:
            if tensor is not None and self._storage_overlaps(output, tensor):
                raise ValueError("output storage must not overlap inputs or workspace")

    def _validate_separate_storage(self, *tensors: torch.Tensor) -> None:
        for index, tensor in enumerate(tensors):
            if any(
                self._storage_overlaps(tensor, other) for other in tensors[index + 1 :]
            ):
                raise ValueError("work buffers must not overlap")

    @staticmethod
    def _storage_overlaps(first: torch.Tensor, second: torch.Tensor) -> bool:
        first_start = first.data_ptr()
        first_end = first_start + first.numel() * first.element_size()
        second_start = second.data_ptr()
        second_end = second_start + second.numel() * second.element_size()
        return first_start < second_end and second_start < first_end


class Rank49Plan(_Plan):
    algorithm = "rank49-v1"
    rank = rank49_4x4x4.RANK
    scheme_size = rank49_4x4x4.DIMENSIONS[0]
    precision_pairs = frozenset(
        {
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float16),
        }
    )
    dynamic_shapes = frozenset(
        {
            (12_288, 12_288, 12_288),
            (16_384, 16_384, 16_384),
        }
    )
    prepacked_shapes = dynamic_shapes | {(8_192, 8_192, 8_192)}
    linear_shape_families = (
        LinearShapeFamily(5_120, 9_216, 4_608, 12_288, False),
        LinearShapeFamily(5_632, 9_216, 12_288, 4_608, False),
        LinearShapeFamily(19_968, 19_968, 4_096, 16_384, True),
        LinearShapeFamily(19_968, 19_968, 16_384, 4_096, True),
    )
    default_weight_transform_config = WeightTransformConfig(2, 1_024, 4)
    weight_transform_configs = {
        (4_096, 12_288): WeightTransformConfig(8, 512, 8),
        (12_288, 4_608): WeightTransformConfig(8, 512, 8),
        (4_096, 16_384): WeightTransformConfig(4, 512, 8),
        (16_384, 4_096): WeightTransformConfig(4, 512, 8),
    }

    @classmethod
    def has_measured_linear_win(
        cls, shape: tuple[int, int, int], *, has_bias: bool
    ) -> bool:
        return any(
            family.matches(shape, has_bias) for family in cls.linear_shape_families
        )

    def is_linear_recommended(self, *, has_bias: bool) -> bool:
        return (
            self.dtype == torch.bfloat16
            and self.compute_dtype == torch.float16
            and self.has_measured_linear_win(self.shape.dimensions, has_bias=has_bias)
            and is_tested_runtime(self.device)
        )

    @classmethod
    def weight_transform_config(cls, inner: int, columns: int) -> WeightTransformConfig:
        return cls.weight_transform_configs.get(
            (inner, columns), cls.default_weight_transform_config
        )

    def _transform_left(self, source: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_elements = 256
        grid = (triton.cdiv(shape.block_rows * shape.block_inner, block_elements),)
        rank49_4x4x4.left_transform_kernel[grid](
            source,
            output,
            shape.rows,
            shape.inner,
            shape.block_rows,
            shape.block_inner,
            shape.block_rows,
            shape.block_inner,
            block_elements,
            num_warps=2,
            num_stages=1,
        )

    def _transform_right(self, source: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_elements = 256
        grid = (triton.cdiv(shape.block_inner * shape.block_columns, block_elements),)
        rank49_4x4x4.right_transform_kernel[grid](
            source,
            output,
            shape.inner,
            shape.columns,
            shape.block_inner,
            shape.block_columns,
            shape.block_inner,
            shape.block_columns,
            block_elements,
            num_warps=2,
            num_stages=1,
        )

    def _transform_weight(self, source: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        config = self.weight_transform_config(shape.inner, shape.columns)
        grid = (
            triton.cdiv(shape.block_inner, config.block_rows),
            triton.cdiv(shape.block_columns, config.block_columns),
        )
        rank49_4x4x4.right_transform_weight_kernel[grid](
            source,
            output,
            shape.columns,
            shape.inner,
            shape.block_inner,
            shape.block_columns,
            config.block_rows,
            config.block_columns,
            num_warps=config.warps,
            num_stages=1,
        )

    def _reconstruct(self, products: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_elements = 256
        grid = (triton.cdiv(shape.block_rows * shape.block_columns, block_elements),)
        rank49_4x4x4.output_transform_kernel[grid](
            products,
            output,
            shape.block_rows,
            shape.block_columns,
            shape.rows,
            shape.columns,
            shape.block_rows,
            shape.block_columns,
            block_elements,
            num_warps=2,
            num_stages=1,
        )

    def _reconstruct_bias(
        self, products: torch.Tensor, bias: torch.Tensor, output: torch.Tensor
    ) -> None:
        shape = self.shape
        block_elements = 256
        grid = (triton.cdiv(shape.block_rows * shape.block_columns, block_elements),)
        rank49_4x4x4.output_transform_bias_kernel[grid](
            products,
            output,
            bias,
            shape.block_rows,
            shape.block_columns,
            shape.rows,
            shape.columns,
            shape.block_rows,
            shape.block_columns,
            block_elements,
            num_warps=2,
            num_stages=1,
        )

    def run_linear(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        workspace: Workspace,
        bias: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_left(input_tensor)
        self._validate_weight(weight)
        if bias is not None:
            self._validate_bias(bias)
        self._validate_workspace(workspace, require_right=True)
        result = self._output(output)
        self._validate_output_storage(
            result,
            input_tensor,
            weight,
            bias,
            workspace.left_transformed,
            workspace.products,
            workspace.right_transformed,
        )
        self._transform_left(input_tensor, workspace.left_transformed)
        right_transformed = workspace.right_transformed
        if right_transformed is None:
            raise ValueError("linear execution requires a dynamic workspace")
        self._transform_weight(weight, right_transformed)
        torch.bmm(
            workspace.left_transformed,
            right_transformed,
            out=workspace.products,
        )
        if bias is None:
            self._reconstruct(workspace.products, result)
        else:
            self._reconstruct_bias(workspace.products, bias, result)
        return result


class Rank343Plan(_Plan):
    algorithm = "rank343-mfma-v1"
    rank = rank343_8x8x8.RANK
    scheme_size = rank343_8x8x8.DIMENSIONS[0]
    precision_pairs = frozenset({(torch.float16, torch.float16)})
    dynamic_shapes = frozenset()
    prepacked_shapes = frozenset({(16_384, 16_384, 16_384)})

    def __init__(
        self,
        rows: int,
        inner: int,
        columns: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(rows, inner, columns, device, dtype, compute_dtype)
        self._output_coefficients = torch.tensor(
            rank343_8x8x8.MFMA_OUTPUT_COEFFICIENTS,
            device=self.device,
            dtype=self.compute_dtype,
        )

    def _transform_left(self, source: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_elements = 1024
        grid = (triton.cdiv(shape.block_rows * shape.block_inner, block_elements),)
        rank343_8x8x8.left_transform_kernel[grid](
            source,
            output,
            shape.rows,
            shape.inner,
            shape.block_rows,
            shape.block_inner,
            shape.block_rows,
            shape.block_inner,
            block_elements,
            num_warps=8,
            num_stages=1,
        )

    def _transform_right(self, source: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_elements = 512
        grid = (triton.cdiv(shape.block_inner * shape.block_columns, block_elements),)
        rank343_8x8x8.right_transform_kernel[grid](
            source,
            output,
            shape.inner,
            shape.columns,
            shape.block_inner,
            shape.block_columns,
            shape.block_inner,
            shape.block_columns,
            block_elements,
            num_warps=8,
            num_stages=1,
        )

    def _reconstruct(self, products: torch.Tensor, output: torch.Tensor) -> None:
        shape = self.shape
        block_positions = 128
        grid = (triton.cdiv(shape.block_rows * shape.block_columns, block_positions),)
        rank343_8x8x8.output_transform_mfma_kernel[grid](
            products,
            self._output_coefficients,
            output,
            shape.rows,
            shape.columns,
            shape.block_rows,
            shape.block_columns,
            block_positions,
            16,
            num_warps=2,
            num_stages=1,
            matrix_instr_nonkdim=16,
            kpack=2,
            waves_per_eu=2,
        )


Rank49Workspace = Workspace
Rank343Workspace = Workspace


def is_tested_runtime(device: torch.device) -> bool:
    hip_version = torch.version.hip
    if hip_version is None or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(device)
    architecture = getattr(properties, "gcnArchName", "")
    return (
        architecture == "gfx1100"
        and properties.multi_processor_count == 48
        and torch.__version__.startswith("2.9.1+")
        and hip_version.startswith("6.4")
        and triton.__version__ == "3.5.1"
    )
