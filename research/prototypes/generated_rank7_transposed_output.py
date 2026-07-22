import triton
import triton.language as tl

DIMENSIONS = (2, 2, 2)
RANK = 7
CERTIFICATE_SHA256 = "b374fcc797ea014994453b7502625e24f70ca2f1dcf1bdbe74a716342bfefb1e"


@triton.jit
def output_transform_transposed_products_kernel(
    source,
    output,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_rows_per_program: tl.constexpr,
    block_columns_per_program: tl.constexpr,
):
    local_rows = (
        tl.program_id(0) * block_rows_per_program
        + tl.arange(0, block_rows_per_program)
    )
    local_columns = (
        tl.program_id(1) * block_columns_per_program
        + tl.arange(0, block_columns_per_program)
    )
    plane_elements = block_height * block_width
    source_offsets = (
        local_columns[:, None] * block_height + local_rows[None, :]
    )
    source_mask = (
        (local_columns[:, None] < block_width)
        & (local_rows[None, :] < block_height)
    )
    signal_2 = tl.trans(
        tl.load(
            source
            + 2 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    signal_3 = tl.trans(
        tl.load(
            source
            + 3 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    tl.store(
        output + (0 * block_height + local_rows[:, None]) * output_columns + 1 * block_width + local_columns[None, :],
        signal_2 + signal_3,
        mask=(local_rows[:, None] < block_height) & (local_columns[None, :] < block_width) & (0 * block_height + local_rows[:, None] < output_row_count) & (1 * block_width + local_columns[None, :] < output_columns),
    )
    signal_5 = tl.trans(
        tl.load(
            source
            + 5 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    signal_7 = signal_3 + signal_5
    signal_1 = tl.trans(
        tl.load(
            source
            + 1 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    signal_6 = tl.trans(
        tl.load(
            source
            + 6 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    tl.store(
        output + (0 * block_height + local_rows[:, None]) * output_columns + 0 * block_width + local_columns[None, :],
        -signal_1 + signal_7 + signal_6,
        mask=(local_rows[:, None] < block_height) & (local_columns[None, :] < block_width) & (0 * block_height + local_rows[:, None] < output_row_count) & (0 * block_width + local_columns[None, :] < output_columns),
    )
    signal_4 = tl.trans(
        tl.load(
            source
            + 4 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    signal_8 = signal_4 - signal_7
    tl.store(
        output + (1 * block_height + local_rows[:, None]) * output_columns + 0 * block_width + local_columns[None, :],
        signal_1 + signal_8,
        mask=(local_rows[:, None] < block_height) & (local_columns[None, :] < block_width) & (1 * block_height + local_rows[:, None] < output_row_count) & (0 * block_width + local_columns[None, :] < output_columns),
    )
    signal_0 = tl.trans(
        tl.load(
            source
            + 0 * plane_elements
            + source_offsets,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
    )
    tl.store(
        output + (1 * block_height + local_rows[:, None]) * output_columns + 1 * block_width + local_columns[None, :],
        signal_0 + signal_8,
        mask=(local_rows[:, None] < block_height) & (local_columns[None, :] < block_width) & (1 * block_height + local_rows[:, None] < output_row_count) & (1 * block_width + local_columns[None, :] < output_columns),
    )
