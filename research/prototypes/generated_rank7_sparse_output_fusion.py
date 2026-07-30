import triton
import triton.language as tl

DIMENSIONS = (2, 2, 2)
RANK = 7
CERTIFICATE_SHA256 = "b374fcc797ea014994453b7502625e24f70ca2f1dcf1bdbe74a716342bfefb1e"


@triton.jit
def sparse_fused_product_output_kernel(
    left_transformed,
    right_transformed,
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
    left_plane_elements = block_rows * block_inner
    right_plane_elements = block_inner * block_columns
    accumulator_0 = tl.zeros((block_m, block_n), tl.float32)
    accumulator_1 = tl.zeros((block_m, block_n), tl.float32)
    accumulator_2 = tl.zeros((block_m, block_n), tl.float32)
    accumulator_3 = tl.zeros((block_m, block_n), tl.float32)
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 0 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 0 * right_plane_elements
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
    accumulator_3 += product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 1 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 1 * right_plane_elements
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
    accumulator_0 -= product
    accumulator_2 += product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 2 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 2 * right_plane_elements
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
    accumulator_1 += product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 3 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 3 * right_plane_elements
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
    accumulator_0 += product
    accumulator_1 += product
    accumulator_2 -= product
    accumulator_3 -= product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 4 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 4 * right_plane_elements
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
    accumulator_2 += product
    accumulator_3 += product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 5 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 5 * right_plane_elements
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
    accumulator_0 += product
    accumulator_2 -= product
    accumulator_3 -= product
    product = tl.zeros((block_m, block_n), tl.float32)
    for inner_start in tl.range(
        0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
    ):
        inner_offsets = inner_start + tl.arange(0, block_k)
        left_values = tl.load(
            left_transformed
            + 6 * left_plane_elements
            + local_rows[:, None] * block_inner
            + inner_offsets[None, :],
            mask=(local_rows[:, None] < block_rows)
            & (inner_offsets[None, :] < block_inner),
            other=0.0,
        ).to(tl.float16)
        right_values = tl.load(
            right_transformed
            + 6 * right_plane_elements
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
    accumulator_0 += product
    if has_bias:
        bias_0 = tl.load(
            bias + 0 * block_columns + local_columns,
            mask=(local_columns < block_columns)
            & (0 * block_columns + local_columns < output_columns),
            other=0.0,
        ).to(tl.float32)
        bias_1 = tl.load(
            bias + 1 * block_columns + local_columns,
            mask=(local_columns < block_columns)
            & (1 * block_columns + local_columns < output_columns),
            other=0.0,
        ).to(tl.float32)
        accumulator_0 += bias_0
        accumulator_1 += bias_1
        accumulator_2 += bias_0
        accumulator_3 += bias_1
    tl.store(
        output
        + (0 * block_rows + local_rows[:, None]) * output_columns
        + 0 * block_columns
        + local_columns[None, :],
        accumulator_0,
        mask=(local_rows[:, None] < block_rows)
        & (local_columns[None, :] < block_columns)
        & (0 * block_rows + local_rows[:, None] < output_row_count)
        & (0 * block_columns + local_columns[None, :] < output_columns),
    )
    tl.store(
        output
        + (0 * block_rows + local_rows[:, None]) * output_columns
        + 1 * block_columns
        + local_columns[None, :],
        accumulator_1,
        mask=(local_rows[:, None] < block_rows)
        & (local_columns[None, :] < block_columns)
        & (0 * block_rows + local_rows[:, None] < output_row_count)
        & (1 * block_columns + local_columns[None, :] < output_columns),
    )
    tl.store(
        output
        + (1 * block_rows + local_rows[:, None]) * output_columns
        + 0 * block_columns
        + local_columns[None, :],
        accumulator_2,
        mask=(local_rows[:, None] < block_rows)
        & (local_columns[None, :] < block_columns)
        & (1 * block_rows + local_rows[:, None] < output_row_count)
        & (0 * block_columns + local_columns[None, :] < output_columns),
    )
    tl.store(
        output
        + (1 * block_rows + local_rows[:, None]) * output_columns
        + 1 * block_columns
        + local_columns[None, :],
        accumulator_3,
        mask=(local_rows[:, None] < block_rows)
        & (local_columns[None, :] < block_columns)
        & (1 * block_rows + local_rows[:, None] < output_row_count)
        & (1 * block_columns + local_columns[None, :] < output_columns),
    )
