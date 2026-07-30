import triton
import triton.language as tl

DIMENSIONS = (3, 3, 4)
RANK = 29
CERTIFICATE_SHA256 = "3d2a6d75c74511d2401f5a941a089c78dcf8e9b8eeaa3e3c8ef759fbd174f8c5"


@triton.jit
def left_transform_kernel(
    source,
    output,
    source_row_count: tl.constexpr,
    source_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_elements: tl.constexpr,
):
    element_offsets = (
        tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    )
    plane_elements = block_height * block_width
    element_mask = element_offsets < plane_elements
    local_rows = element_offsets // block_width
    local_columns = element_offsets % block_width
    signal_3 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_1 + signal_3, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_3 - signal_4, mask=element_mask
    )
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_1 + signal_4, mask=element_mask
    )
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_5, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_0 + signal_3, mask=element_mask
    )
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_0 - signal_1, mask=element_mask
    )
    signal_6 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_6, mask=element_mask
    )
    signal_7 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_7, mask=element_mask
    )
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    signal_9 = signal_4 - signal_7
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_9, mask=element_mask
    )
    signal_10 = signal_0 - signal_6
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_10, mask=element_mask
    )
    signal_11 = signal_0 + signal_2
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_11, mask=element_mask
    )
    signal_12 = signal_4 + signal_5
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_12, mask=element_mask
    )
    signal_13 = signal_1 - signal_10
    tl.store(
        output + 2 * plane_elements + element_offsets, -signal_13, mask=element_mask
    )
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_7 - signal_13, mask=element_mask
    )
    signal_14 = signal_5 + signal_9
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_14, mask=element_mask
    )
    signal_15 = signal_1 + signal_12
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_15, mask=element_mask
    )
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_15 + signal_2, mask=element_mask
    )
    signal_16 = signal_2 + signal_10
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_14 + signal_16 - signal_8, mask=element_mask
    )
    signal_17 = signal_3 + signal_11
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_17 + signal_5, mask=element_mask
    )
    signal_18 = signal_3 - signal_9
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_18, mask=element_mask
    )
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_18 - signal_6, mask=element_mask
    )


@triton.jit
def right_transform_kernel(
    source,
    output,
    source_row_count: tl.constexpr,
    source_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_elements: tl.constexpr,
):
    element_offsets = (
        tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    )
    plane_elements = block_height * block_width
    element_mask = element_offsets < plane_elements
    local_rows = element_offsets // block_width
    local_columns = element_offsets % block_width
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_6 + signal_11, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_2 + signal_8, mask=element_mask
    )
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_6, mask=element_mask
    )
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_11, mask=element_mask
    )
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_3 + signal_6 + signal_7, mask=element_mask
    )
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_3 - signal_4, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_0 - signal_1, mask=element_mask
    )
    signal_10 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_2 - signal_10 - signal_11, mask=element_mask
    )
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_4, mask=element_mask
    )
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_4 + signal_7 - signal_11, mask=element_mask
    )
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_6 - signal_8 - signal_10, mask=element_mask
    )
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_0 + signal_3 - signal_8, mask=element_mask
    )
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_5, mask=element_mask
    )
    signal_9 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_9, mask=element_mask
    )
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_11, mask=element_mask
    )
    signal_12 = signal_9 + signal_10
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_12, mask=element_mask
    )
    signal_13 = signal_0 + signal_4
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_13 + signal_2, mask=element_mask
    )
    signal_14 = signal_5 - signal_7
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_14, mask=element_mask
    )
    signal_15 = signal_6 - signal_12
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_15, mask=element_mask
    )
    signal_16 = signal_3 - signal_14
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    signal_17 = signal_2 - signal_12
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_1 + signal_17 - signal_16, mask=element_mask
    )
    signal_18 = signal_1 - signal_13
    tl.store(
        output + 3 * plane_elements + element_offsets, -signal_18, mask=element_mask
    )
    tl.store(
        output + 8 * plane_elements + element_offsets, -signal_5 - signal_18 - signal_15, mask=element_mask
    )


@triton.jit
def right_transform_weight_kernel(
    source,
    output,
    source_row_count: tl.constexpr,
    source_columns: tl.constexpr,
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
    output_offsets = (
        local_rows[:, None] * block_width + local_columns[None, :]
    )
    output_mask = (
        (local_rows[:, None] < block_height)
        & (local_columns[None, :] < block_width)
    )
    signal_6 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_11 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 0 * plane_elements + output_offsets,
        signal_6 + signal_11,
        mask=output_mask,
    )
    signal_2 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_8 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 1 * plane_elements + output_offsets,
        signal_2 + signal_8,
        mask=output_mask,
    )
    tl.store(
        output + 9 * plane_elements + output_offsets,
        signal_6,
        mask=output_mask,
    )
    tl.store(
        output + 10 * plane_elements + output_offsets,
        signal_8,
        mask=output_mask,
    )
    tl.store(
        output + 11 * plane_elements + output_offsets,
        signal_11,
        mask=output_mask,
    )
    tl.store(
        output + 12 * plane_elements + output_offsets,
        signal_2,
        mask=output_mask,
    )
    signal_3 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_7 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 14 * plane_elements + output_offsets,
        signal_3 + signal_6 + signal_7,
        mask=output_mask,
    )
    signal_4 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 15 * plane_elements + output_offsets,
        signal_3 - signal_4,
        mask=output_mask,
    )
    signal_0 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_1 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 17 * plane_elements + output_offsets,
        signal_0 - signal_1,
        mask=output_mask,
    )
    signal_10 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 18 * plane_elements + output_offsets,
        signal_2 - signal_10 - signal_11,
        mask=output_mask,
    )
    tl.store(
        output + 19 * plane_elements + output_offsets,
        signal_4,
        mask=output_mask,
    )
    tl.store(
        output + 20 * plane_elements + output_offsets,
        signal_4 + signal_7 - signal_11,
        mask=output_mask,
    )
    tl.store(
        output + 21 * plane_elements + output_offsets,
        signal_6 - signal_8 - signal_10,
        mask=output_mask,
    )
    tl.store(
        output + 22 * plane_elements + output_offsets,
        signal_0 + signal_3 - signal_8,
        mask=output_mask,
    )
    tl.store(
        output + 23 * plane_elements + output_offsets,
        signal_3,
        mask=output_mask,
    )
    tl.store(
        output + 24 * plane_elements + output_offsets,
        signal_1,
        mask=output_mask,
    )
    signal_5 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 25 * plane_elements + output_offsets,
        signal_5,
        mask=output_mask,
    )
    signal_9 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 26 * plane_elements + output_offsets,
        signal_9,
        mask=output_mask,
    )
    tl.store(
        output + 27 * plane_elements + output_offsets,
        signal_8,
        mask=output_mask,
    )
    tl.store(
        output + 28 * plane_elements + output_offsets,
        signal_11,
        mask=output_mask,
    )
    signal_12 = signal_9 + signal_10
    tl.store(
        output + 6 * plane_elements + output_offsets,
        signal_12,
        mask=output_mask,
    )
    signal_13 = signal_0 + signal_4
    tl.store(
        output + 13 * plane_elements + output_offsets,
        signal_13 + signal_2,
        mask=output_mask,
    )
    signal_14 = signal_5 - signal_7
    tl.store(
        output + 16 * plane_elements + output_offsets,
        signal_14,
        mask=output_mask,
    )
    signal_15 = signal_6 - signal_12
    tl.store(
        output + 5 * plane_elements + output_offsets,
        signal_15,
        mask=output_mask,
    )
    signal_16 = signal_3 - signal_14
    tl.store(
        output + 2 * plane_elements + output_offsets,
        signal_16,
        mask=output_mask,
    )
    signal_17 = signal_2 - signal_12
    tl.store(
        output + 4 * plane_elements + output_offsets,
        signal_17,
        mask=output_mask,
    )
    tl.store(
        output + 7 * plane_elements + output_offsets,
        signal_1 + signal_17 - signal_16,
        mask=output_mask,
    )
    signal_18 = signal_1 - signal_13
    tl.store(
        output + 3 * plane_elements + output_offsets,
        -signal_18,
        mask=output_mask,
    )
    tl.store(
        output + 8 * plane_elements + output_offsets,
        -signal_5 - signal_18 - signal_15,
        mask=output_mask,
    )


@triton.jit
def output_transform_kernel(
    source,
    output,
    source_row_count: tl.constexpr,
    source_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_elements: tl.constexpr,
):
    element_offsets = (
        tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    )
    plane_elements = block_height * block_width
    element_mask = element_offsets < plane_elements
    local_rows = element_offsets // block_width
    local_columns = element_offsets % block_width
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_3 - signal_17 - signal_19 + signal_24 + signal_27, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_24 + signal_25 + signal_26, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_4 - signal_5 - signal_6 + signal_9 + signal_12 - signal_26, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, -signal_2 - signal_16 + signal_23 + signal_25 + signal_28, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = signal_1 - signal_12
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_29 - signal_23 - signal_13 - signal_15 + signal_22, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_30 = signal_0 - signal_9
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_30 + signal_20 - signal_14 + signal_15 + signal_19, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_31 = signal_11 - signal_30
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_14 + signal_23 + signal_31, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_32 = signal_10 - signal_29
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_21 + signal_9 - signal_32, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_33 = signal_18 + signal_31
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_33 + signal_2 - signal_4 + signal_7 + signal_24 + signal_14, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_12 - signal_33, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_34 = signal_13 + signal_32
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_34 - signal_19, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_21 - signal_3 - signal_5 - signal_8 + signal_34 + signal_25, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )


@triton.jit
def output_transform_bias_kernel(
    source,
    output,
    bias,
    source_row_count: tl.constexpr,
    source_columns: tl.constexpr,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_elements: tl.constexpr,
):
    element_offsets = (
        tl.program_id(0) * block_elements + tl.arange(0, block_elements)
    )
    plane_elements = block_height * block_width
    element_mask = element_offsets < plane_elements
    local_rows = element_offsets // block_width
    local_columns = element_offsets % block_width
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_2 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_3 - signal_17 - signal_19 + signal_24 + signal_27 + bias_2, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_5 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_24 + signal_25 + signal_26 + bias_5, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_8 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_4 - signal_5 - signal_6 + signal_9 + signal_12 - signal_26 + bias_8, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_11 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, -signal_2 - signal_16 + signal_23 + signal_25 + signal_28 + bias_11, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = signal_1 - signal_12
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_0 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_29 - signal_23 - signal_13 - signal_15 + signal_22 + bias_0, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_30 = signal_0 - signal_9
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_10 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_30 + signal_20 - signal_14 + signal_15 + signal_19 + bias_10, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_31 = signal_11 - signal_30
    bias_9 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_14 + signal_23 + signal_31 + bias_9, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_32 = signal_10 - signal_29
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_7 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_21 + signal_9 - signal_32 + bias_7, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_33 = signal_18 + signal_31
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_3 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_33 + signal_2 - signal_4 + signal_7 + signal_24 + signal_14 + bias_3, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    bias_6 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_12 - signal_33 + bias_6, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_34 = signal_13 + signal_32
    bias_1 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_34 - signal_19 + bias_1, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_4 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_21 - signal_3 - signal_5 - signal_8 + signal_34 + signal_25 + bias_4, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
