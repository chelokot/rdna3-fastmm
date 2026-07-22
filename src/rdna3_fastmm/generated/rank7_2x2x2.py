import triton
import triton.language as tl

DIMENSIONS = (2, 2, 2)
RANK = 7
CERTIFICATE_SHA256 = "b374fcc797ea014994453b7502625e24f70ca2f1dcf1bdbe74a716342bfefb1e"


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
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_0, mask=element_mask
    )
    signal_2 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_0 + signal_2, mask=element_mask
    )
    signal_4 = signal_2 + signal_3
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_4, mask=element_mask
    )
    signal_5 = signal_0 + signal_4
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_5, mask=element_mask
    )
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_5 + signal_1, mask=element_mask
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
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_0 - signal_1, mask=element_mask
    )
    signal_3 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_2 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    signal_4 = signal_0 - signal_2
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_4, mask=element_mask
    )
    signal_5 = signal_1 - signal_4
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_3 - signal_5, mask=element_mask
    )
    tl.store(
        output + 5 * plane_elements + element_offsets, -signal_5, mask=element_mask
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
        output + 1 * plane_elements + output_offsets,
        signal_0 - signal_1,
        mask=output_mask,
    )
    signal_3 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 2 * plane_elements + output_offsets,
        signal_3,
        mask=output_mask,
    )
    tl.store(
        output + 3 * plane_elements + output_offsets,
        signal_1,
        mask=output_mask,
    )
    signal_2 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 6 * plane_elements + output_offsets,
        signal_2,
        mask=output_mask,
    )
    signal_4 = signal_0 - signal_2
    tl.store(
        output + 4 * plane_elements + output_offsets,
        signal_4,
        mask=output_mask,
    )
    signal_5 = signal_1 - signal_4
    tl.store(
        output + 0 * plane_elements + output_offsets,
        signal_3 - signal_5,
        mask=output_mask,
    )
    tl.store(
        output + 5 * plane_elements + output_offsets,
        -signal_5,
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
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_2 + signal_3, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_7 = signal_3 + signal_5
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_1 + signal_7 + signal_6, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_8 = signal_4 - signal_7
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_1 + signal_8, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_0 + signal_8, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
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
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_2 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_2 + signal_3 + bias_2, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_7 = signal_3 + signal_5
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_0 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_1 + signal_7 + signal_6 + bias_0, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_8 = signal_4 - signal_7
    bias_1 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_1 + signal_8 + bias_1, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_3 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_0 + signal_8 + bias_3, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
