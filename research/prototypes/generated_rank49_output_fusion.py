import triton
import triton.language as tl

DIMENSIONS = (4, 4, 4)
RANK = 49
CERTIFICATE_SHA256 = "a3c4121dfd09607045255628dd94b60133c24c46b65f1089e89c6564ba522561"

FUSED_OUTPUT_COEFFICIENTS = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, -1, -1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, -1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    (1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0),
    (0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
    (-1, -1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0),
    (0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
    (-1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0),
    (-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1),
    (-1, 0, -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0),
    (0, 1, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0),
    (1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1),
    (1, 0, 1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 1, 1, 1, 1),
    (1, 0, 1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 1, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, -1, -1, -1, -1),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1),
    (-1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, -1, 0, -1, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0),
    (1, 1, 0, 0, -1, -1, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1),
    (1, 0, 0, 0, -1, -1, 0, 0, -1, 0, -1, 0, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
)


@triton.jit
def fused_product_output_kernel(
    left_transformed,
    right_transformed,
    coefficients,
    output,
    bias,
    block_rows: tl.constexpr,
    block_inner: tl.constexpr,
    block_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_bias: tl.constexpr,
):
    local_rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    local_columns = tl.program_id(1) * block_n + tl.arange(0, block_n)
    output_blocks = tl.arange(0, 16)
    left_plane_elements = block_rows * block_inner
    right_plane_elements = block_inner * block_columns
    accumulator = tl.zeros((block_m, 16, block_n), tl.float32)
    for rank_index in tl.range(0, 49, loop_unroll_factor=1):
        product = tl.zeros((block_m, block_n), tl.float32)
        for inner_start in tl.range(
            0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
        ):
            inner_offsets = inner_start + tl.arange(0, block_k)
            left_values = tl.load(
                left_transformed
                + rank_index * left_plane_elements
                + local_rows[:, None] * block_inner
                + inner_offsets[None, :],
                mask=(local_rows[:, None] < block_rows)
                & (inner_offsets[None, :] < block_inner),
                other=0.0,
            ).to(tl.float16)
            right_values = tl.load(
                right_transformed
                + rank_index * right_plane_elements
                + inner_offsets[:, None] * block_columns
                + local_columns[None, :],
                mask=(inner_offsets[:, None] < block_inner)
                & (local_columns[None, :] < block_columns),
                other=0.0,
            ).to(tl.float16)
            product = tl.dot(
                left_values,
                right_values,
                acc=product,
                out_dtype=tl.float32,
            )
        coefficient_values = tl.load(coefficients + rank_index * 16 + output_blocks).to(
            tl.float32
        )
        accumulator += product[:, None, :] * coefficient_values[None, :, None]
    output_block_rows = output_blocks // 4
    output_block_columns = output_blocks % 4
    output_rows = (
        output_block_rows[None, :, None] * block_rows + local_rows[:, None, None]
    )
    output_column_indices = (
        output_block_columns[None, :, None] * block_columns
        + local_columns[None, None, :]
    )
    output_mask = (
        (local_rows[:, None, None] < block_rows)
        & (local_columns[None, None, :] < block_columns)
        & (output_rows < output_row_count)
        & (output_column_indices < output_columns)
    )
    if has_bias:
        bias_values = tl.load(
            bias
            + output_block_columns[:, None] * block_columns
            + local_columns[None, :],
            mask=(
                (local_columns[None, :] < block_columns)
                & (
                    output_block_columns[:, None] * block_columns
                    + local_columns[None, :]
                    < output_columns
                )
            ),
            other=0.0,
        ).to(tl.float32)
        accumulator += bias_values[None, :, :]
    tl.store(
        output + output_rows * output_columns + output_column_indices,
        accumulator,
        mask=output_mask,
    )


@triton.jit
def atomic_product_output_kernel(
    left_transformed,
    right_transformed,
    coefficients,
    accumulator,
    block_rows: tl.constexpr,
    block_inner: tl.constexpr,
    block_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    local_rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    local_columns = tl.program_id(1) * block_n + tl.arange(0, block_n)
    rank_index = tl.program_id(2)
    left_plane_elements = block_rows * block_inner
    right_plane_elements = block_inner * block_columns
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + rank_index * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + rank_index * right_plane_elements
            + inner_offsets[:, None] * block_columns
            + local_columns[None, :],
            mask=(inner_offsets[:, None] < block_inner)
            & (local_columns[None, :] < block_columns),
            other=0.0,
        ).to(tl.float16)
        product = tl.dot(
            left_values,
            right_values,
            acc=product,
            out_dtype=tl.float32,
        )
    for output_block in tl.range(0, 16, loop_unroll_factor=1):
        coefficient_value = tl.load(coefficients + rank_index * 16 + output_block).to(
            tl.float32
        )
        output_block_row = output_block // 4
        output_block_column = output_block % 4
        output_rows = output_block_row * block_rows + local_rows[:, None]
        output_column_indices = (
            output_block_column * block_columns + local_columns[None, :]
        )
        output_mask = (
            (local_rows[:, None] < block_rows)
            & (local_columns[None, :] < block_columns)
            & (output_rows < output_row_count)
            & (output_column_indices < output_columns)
            & (coefficient_value != 0.0)
        )
        tl.atomic_add(
            accumulator + output_rows * output_columns + output_column_indices,
            product * coefficient_value,
            mask=output_mask,
            sem="relaxed",
        )


@triton.jit
def atomic_output_finalize_kernel(
    accumulator,
    output,
    bias,
    output_elements: tl.constexpr,
    output_columns: tl.constexpr,
    block_elements: tl.constexpr,
    has_bias: tl.constexpr,
    clear_accumulator: tl.constexpr,
):
    offsets = tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    output_mask = offsets < output_elements
    values = tl.load(accumulator + offsets, mask=output_mask, other=0.0).to(tl.float32)
    if has_bias:
        values += tl.load(
            bias + offsets % output_columns, mask=output_mask, other=0.0
        ).to(tl.float32)
    tl.store(output + offsets, values, mask=output_mask)
    if clear_accumulator:
        tl.store(accumulator + offsets, 0.0, mask=output_mask)
