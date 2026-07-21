import triton
import triton.language as tl

DIMENSIONS = (4, 4, 4)
RANK = 49
CERTIFICATE_SHA256 = "a3c4121dfd09607045255628dd94b60133c24c46b65f1089e89c6564ba522561"


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
    signal_15 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_15, mask=element_mask
    )
    signal_11 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_11, mask=element_mask
    )
    signal_10 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_10, mask=element_mask
    )
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_7, mask=element_mask
    )
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_5, mask=element_mask
    )
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_0, mask=element_mask
    )
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_0 + signal_4, mask=element_mask
    )
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_16 = signal_2 + signal_6
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    signal_9 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_17 = signal_9 + signal_11
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    signal_18 = signal_7 + signal_16
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_18, mask=element_mask
    )
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_18 + signal_3, mask=element_mask
    )
    signal_12 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_19 = signal_4 + signal_12
    signal_20 = signal_6 + signal_7
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_20, mask=element_mask
    )
    signal_13 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_21 = signal_5 + signal_13
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_21, mask=element_mask
    )
    signal_22 = signal_4 + signal_5
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_22, mask=element_mask
    )
    signal_14 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_23 = signal_12 + signal_14
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_24 = signal_0 + signal_8
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_24, mask=element_mask
    )
    signal_25 = signal_19 + signal_21
    tl.store(
        output + 29 * plane_elements + element_offsets, signal_25, mask=element_mask
    )
    signal_26 = signal_10 + signal_14
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_26, mask=element_mask
    )
    signal_27 = signal_24 + signal_25
    tl.store(
        output + 33 * plane_elements + element_offsets, signal_27, mask=element_mask
    )
    signal_28 = signal_15 + signal_26
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_28, mask=element_mask
    )
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_28 + signal_11, mask=element_mask
    )
    signal_29 = signal_27 + signal_28
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_29, mask=element_mask
    )
    signal_30 = signal_1 + signal_17
    tl.store(
        output + 37 * plane_elements + element_offsets, signal_30, mask=element_mask
    )
    tl.store(
        output + 41 * plane_elements + element_offsets, signal_29 + signal_30, mask=element_mask
    )
    signal_31 = signal_8 + signal_10
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_31, mask=element_mask
    )
    signal_32 = signal_13 + signal_15
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_32, mask=element_mask
    )
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_23 + signal_32, mask=element_mask
    )
    signal_33 = signal_14 + signal_15
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_33, mask=element_mask
    )
    signal_34 = signal_23 + signal_31
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    signal_35 = signal_19 + signal_24
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_35, mask=element_mask
    )
    signal_36 = signal_32 + signal_34
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_36, mask=element_mask
    )
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_36 + signal_17, mask=element_mask
    )
    signal_37 = signal_0 + signal_22
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_37, mask=element_mask
    )
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_37 + signal_1, mask=element_mask
    )
    signal_38 = signal_3 + signal_30
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_38, mask=element_mask
    )
    signal_39 = signal_1 + signal_9
    tl.store(
        output + 30 * plane_elements + element_offsets, signal_39, mask=element_mask
    )
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_27 + signal_39, mask=element_mask
    )
    signal_40 = signal_15 + signal_21
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_40, mask=element_mask
    )
    tl.store(
        output + 42 * plane_elements + element_offsets, signal_40 + signal_7, mask=element_mask
    )
    signal_41 = signal_25 + signal_33
    tl.store(
        output + 36 * plane_elements + element_offsets, signal_41, mask=element_mask
    )
    tl.store(
        output + 43 * plane_elements + element_offsets, signal_41 + signal_20, mask=element_mask
    )
    signal_42 = signal_10 + signal_24
    tl.store(
        output + 38 * plane_elements + element_offsets, signal_42, mask=element_mask
    )
    tl.store(
        output + 45 * plane_elements + element_offsets, signal_42 + signal_2, mask=element_mask
    )
    signal_43 = signal_26 + signal_35
    tl.store(
        output + 39 * plane_elements + element_offsets, signal_43, mask=element_mask
    )
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_43 + signal_16, mask=element_mask
    )
    signal_44 = signal_18 + signal_29
    tl.store(
        output + 47 * plane_elements + element_offsets, signal_44, mask=element_mask
    )
    tl.store(
        output + 48 * plane_elements + element_offsets, signal_44 + signal_38, mask=element_mask
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
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_1 - signal_3, mask=element_mask
    )
    signal_15 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_15, mask=element_mask
    )
    signal_11 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_11, mask=element_mask
    )
    signal_14 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_14, mask=element_mask
    )
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_7, mask=element_mask
    )
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_3, mask=element_mask
    )
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_6, mask=element_mask
    )
    signal_13 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_13, mask=element_mask
    )
    signal_9 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 45 * plane_elements + element_offsets, signal_9, mask=element_mask
    )
    signal_12 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 48 * plane_elements + element_offsets, signal_12, mask=element_mask
    )
    signal_10 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_16 = signal_10 - signal_14
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_17 = signal_2 - signal_3
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_18 = signal_5 - signal_13
    tl.store(
        output + 30 * plane_elements + element_offsets, signal_18, mask=element_mask
    )
    signal_19 = signal_7 - signal_18
    tl.store(
        output + 37 * plane_elements + element_offsets, -signal_19, mask=element_mask
    )
    signal_20 = signal_5 - signal_7
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_20, mask=element_mask
    )
    signal_21 = signal_11 - signal_16
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_15 - signal_21, mask=element_mask
    )
    tl.store(
        output + 19 * plane_elements + element_offsets, -signal_21, mask=element_mask
    )
    signal_22 = signal_10 - signal_11
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_22, mask=element_mask
    )
    signal_23 = signal_2 - signal_6
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_23, mask=element_mask
    )
    signal_24 = signal_15 - signal_19
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_24, mask=element_mask
    )
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_25 = signal_8 - signal_9
    tl.store(
        output + 43 * plane_elements + element_offsets, signal_25, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_26 = signal_0 - signal_1
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_27 = signal_4 - signal_6
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_27, mask=element_mask
    )
    signal_28 = signal_0 - signal_4
    signal_29 = signal_17 - signal_26
    tl.store(
        output + 8 * plane_elements + element_offsets, -signal_29, mask=element_mask
    )
    signal_30 = signal_23 - signal_28
    tl.store(
        output + 11 * plane_elements + element_offsets, -signal_30, mask=element_mask
    )
    signal_31 = signal_12 - signal_27
    tl.store(
        output + 6 * plane_elements + element_offsets, -signal_31 + signal_14, mask=element_mask
    )
    tl.store(
        output + 41 * plane_elements + element_offsets, -signal_31, mask=element_mask
    )
    signal_32 = signal_25 + signal_29
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_22 - signal_32, mask=element_mask
    )
    tl.store(
        output + 36 * plane_elements + element_offsets, -signal_32, mask=element_mask
    )
    signal_33 = signal_31 - signal_32
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_33 - signal_19, mask=element_mask
    )
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_33, mask=element_mask
    )
    signal_34 = signal_1 - signal_9
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    signal_35 = signal_8 - signal_12
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_28 - signal_35, mask=element_mask
    )
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_35, mask=element_mask
    )
    signal_36 = signal_4 - signal_12
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_36, mask=element_mask
    )
    signal_37 = signal_25 - signal_26
    tl.store(
        output + 29 * plane_elements + element_offsets, -signal_37, mask=element_mask
    )
    signal_38 = signal_21 - signal_33
    tl.store(
        output + 0 * plane_elements + element_offsets, -signal_38 + signal_24, mask=element_mask
    )
    tl.store(
        output + 5 * plane_elements + element_offsets, -signal_38, mask=element_mask
    )
    signal_39 = signal_3 - signal_34
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_11 - signal_39, mask=element_mask
    )
    tl.store(
        output + 38 * plane_elements + element_offsets, -signal_39, mask=element_mask
    )
    signal_40 = signal_30 + signal_35
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_16 - signal_40, mask=element_mask
    )
    tl.store(
        output + 39 * plane_elements + element_offsets, -signal_40, mask=element_mask
    )
    signal_41 = signal_27 + signal_29
    tl.store(
        output + 7 * plane_elements + element_offsets, -signal_41 + signal_20, mask=element_mask
    )
    tl.store(
        output + 12 * plane_elements + element_offsets, -signal_41, mask=element_mask
    )
    signal_42 = signal_6 - signal_17
    tl.store(
        output + 21 * plane_elements + element_offsets, -signal_42 + signal_7, mask=element_mask
    )
    tl.store(
        output + 26 * plane_elements + element_offsets, -signal_42, mask=element_mask
    )
    signal_43 = signal_36 + signal_37
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_18 - signal_43, mask=element_mask
    )
    tl.store(
        output + 33 * plane_elements + element_offsets, -signal_43, mask=element_mask
    )
    signal_44 = signal_12 - signal_25
    tl.store(
        output + 42 * plane_elements + element_offsets, -signal_44 + signal_13, mask=element_mask
    )
    tl.store(
        output + 47 * plane_elements + element_offsets, -signal_44, mask=element_mask
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
    signal_1 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_3 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 10 * plane_elements + output_offsets,
        signal_1 - signal_3,
        mask=output_mask,
    )
    signal_15 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 16 * plane_elements + output_offsets,
        signal_15,
        mask=output_mask,
    )
    signal_11 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 17 * plane_elements + output_offsets,
        signal_11,
        mask=output_mask,
    )
    signal_14 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 20 * plane_elements + output_offsets,
        signal_14,
        mask=output_mask,
    )
    signal_7 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 23 * plane_elements + output_offsets,
        signal_7,
        mask=output_mask,
    )
    tl.store(
        output + 24 * plane_elements + output_offsets,
        signal_3,
        mask=output_mask,
    )
    signal_6 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 27 * plane_elements + output_offsets,
        signal_6,
        mask=output_mask,
    )
    signal_13 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 44 * plane_elements + output_offsets,
        signal_13,
        mask=output_mask,
    )
    signal_9 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 45 * plane_elements + output_offsets,
        signal_9,
        mask=output_mask,
    )
    signal_12 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 48 * plane_elements + output_offsets,
        signal_12,
        mask=output_mask,
    )
    signal_10 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_16 = signal_10 - signal_14
    tl.store(
        output + 18 * plane_elements + output_offsets,
        signal_16,
        mask=output_mask,
    )
    signal_2 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_17 = signal_2 - signal_3
    tl.store(
        output + 22 * plane_elements + output_offsets,
        signal_17,
        mask=output_mask,
    )
    signal_5 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_18 = signal_5 - signal_13
    tl.store(
        output + 30 * plane_elements + output_offsets,
        signal_18,
        mask=output_mask,
    )
    signal_19 = signal_7 - signal_18
    tl.store(
        output + 37 * plane_elements + output_offsets,
        -signal_19,
        mask=output_mask,
    )
    signal_20 = signal_5 - signal_7
    tl.store(
        output + 9 * plane_elements + output_offsets,
        signal_20,
        mask=output_mask,
    )
    signal_21 = signal_11 - signal_16
    tl.store(
        output + 14 * plane_elements + output_offsets,
        signal_15 - signal_21,
        mask=output_mask,
    )
    tl.store(
        output + 19 * plane_elements + output_offsets,
        -signal_21,
        mask=output_mask,
    )
    signal_22 = signal_10 - signal_11
    tl.store(
        output + 15 * plane_elements + output_offsets,
        signal_22,
        mask=output_mask,
    )
    signal_23 = signal_2 - signal_6
    tl.store(
        output + 25 * plane_elements + output_offsets,
        signal_23,
        mask=output_mask,
    )
    signal_24 = signal_15 - signal_19
    tl.store(
        output + 2 * plane_elements + output_offsets,
        signal_24,
        mask=output_mask,
    )
    signal_8 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_25 = signal_8 - signal_9
    tl.store(
        output + 43 * plane_elements + output_offsets,
        signal_25,
        mask=output_mask,
    )
    signal_0 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_26 = signal_0 - signal_1
    signal_4 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_27 = signal_4 - signal_6
    tl.store(
        output + 13 * plane_elements + output_offsets,
        signal_27,
        mask=output_mask,
    )
    signal_28 = signal_0 - signal_4
    signal_29 = signal_17 - signal_26
    tl.store(
        output + 8 * plane_elements + output_offsets,
        -signal_29,
        mask=output_mask,
    )
    signal_30 = signal_23 - signal_28
    tl.store(
        output + 11 * plane_elements + output_offsets,
        -signal_30,
        mask=output_mask,
    )
    signal_31 = signal_12 - signal_27
    tl.store(
        output + 6 * plane_elements + output_offsets,
        -signal_31 + signal_14,
        mask=output_mask,
    )
    tl.store(
        output + 41 * plane_elements + output_offsets,
        -signal_31,
        mask=output_mask,
    )
    signal_32 = signal_25 + signal_29
    tl.store(
        output + 1 * plane_elements + output_offsets,
        signal_22 - signal_32,
        mask=output_mask,
    )
    tl.store(
        output + 36 * plane_elements + output_offsets,
        -signal_32,
        mask=output_mask,
    )
    signal_33 = signal_31 - signal_32
    tl.store(
        output + 35 * plane_elements + output_offsets,
        signal_33 - signal_19,
        mask=output_mask,
    )
    tl.store(
        output + 40 * plane_elements + output_offsets,
        signal_33,
        mask=output_mask,
    )
    signal_34 = signal_1 - signal_9
    tl.store(
        output + 31 * plane_elements + output_offsets,
        signal_34,
        mask=output_mask,
    )
    signal_35 = signal_8 - signal_12
    tl.store(
        output + 32 * plane_elements + output_offsets,
        signal_28 - signal_35,
        mask=output_mask,
    )
    tl.store(
        output + 46 * plane_elements + output_offsets,
        signal_35,
        mask=output_mask,
    )
    signal_36 = signal_4 - signal_12
    tl.store(
        output + 34 * plane_elements + output_offsets,
        signal_36,
        mask=output_mask,
    )
    signal_37 = signal_25 - signal_26
    tl.store(
        output + 29 * plane_elements + output_offsets,
        -signal_37,
        mask=output_mask,
    )
    signal_38 = signal_21 - signal_33
    tl.store(
        output + 0 * plane_elements + output_offsets,
        -signal_38 + signal_24,
        mask=output_mask,
    )
    tl.store(
        output + 5 * plane_elements + output_offsets,
        -signal_38,
        mask=output_mask,
    )
    signal_39 = signal_3 - signal_34
    tl.store(
        output + 3 * plane_elements + output_offsets,
        signal_11 - signal_39,
        mask=output_mask,
    )
    tl.store(
        output + 38 * plane_elements + output_offsets,
        -signal_39,
        mask=output_mask,
    )
    signal_40 = signal_30 + signal_35
    tl.store(
        output + 4 * plane_elements + output_offsets,
        signal_16 - signal_40,
        mask=output_mask,
    )
    tl.store(
        output + 39 * plane_elements + output_offsets,
        -signal_40,
        mask=output_mask,
    )
    signal_41 = signal_27 + signal_29
    tl.store(
        output + 7 * plane_elements + output_offsets,
        -signal_41 + signal_20,
        mask=output_mask,
    )
    tl.store(
        output + 12 * plane_elements + output_offsets,
        -signal_41,
        mask=output_mask,
    )
    signal_42 = signal_6 - signal_17
    tl.store(
        output + 21 * plane_elements + output_offsets,
        -signal_42 + signal_7,
        mask=output_mask,
    )
    tl.store(
        output + 26 * plane_elements + output_offsets,
        -signal_42,
        mask=output_mask,
    )
    signal_43 = signal_36 + signal_37
    tl.store(
        output + 28 * plane_elements + output_offsets,
        signal_18 - signal_43,
        mask=output_mask,
    )
    tl.store(
        output + 33 * plane_elements + output_offsets,
        -signal_43,
        mask=output_mask,
    )
    signal_44 = signal_12 - signal_25
    tl.store(
        output + 42 * plane_elements + output_offsets,
        -signal_44 + signal_13,
        mask=output_mask,
    )
    tl.store(
        output + 47 * plane_elements + output_offsets,
        -signal_44,
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
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_49 = signal_24 + signal_38
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_50 = signal_26 + signal_40
    signal_51 = signal_49 + signal_50
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = signal_12 - signal_51
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_53 = signal_10 + signal_52
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_54 = signal_22 + signal_36
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_55 = signal_31 + signal_33
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = signal_25 + signal_39
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = signal_32 - signal_56
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_58 = signal_8 - signal_53
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = signal_29 - signal_55
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_60 = signal_17 + signal_24
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_16 + signal_60 + signal_23, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_61 = signal_54 - signal_58
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_62 = signal_19 + signal_26
    signal_63 = signal_23 + signal_49
    signal_64 = signal_60 + signal_62
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_65 = signal_5 - signal_51
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_66 = signal_27 + signal_41
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_67 = signal_3 + signal_65
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = signal_45 + signal_47
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_69 = signal_21 + signal_35
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_70 = signal_37 + signal_63
    signal_71 = signal_57 + signal_59
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_72 = signal_7 + signal_11
    signal_73 = signal_61 - signal_66
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_74 = signal_46 + signal_56
    signal_75 = signal_31 - signal_70
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_76 = signal_15 - signal_64
    signal_77 = signal_53 + signal_69
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_78 = signal_34 - signal_59
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_79 = signal_1 - signal_67
    signal_80 = signal_72 - signal_77
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_81 = signal_28 + signal_57
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_82 = signal_13 + signal_73
    signal_48 = tl.load(
        source + 48 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_68 + signal_48 - signal_43 - signal_82, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_78 + signal_82, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_83 = signal_30 + signal_75
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_2 + signal_3 + signal_83, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_84 = signal_54 - signal_79
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_84 + signal_78 - signal_66 + signal_6, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_71 - signal_84 + signal_4, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_85 = signal_22 + signal_76
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_20 - signal_85 + signal_27, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_86 = signal_9 + signal_10
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_86 + signal_70 + signal_45 + signal_44, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_86 + signal_83, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_87 = signal_11 - signal_61
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_87 + signal_71, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_88 = signal_68 - signal_74
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_88 + signal_43 - signal_87, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_80 + signal_42 - signal_88, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_89 = signal_18 + signal_25
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_89 + signal_85, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_14 - signal_64 + signal_89 + signal_21, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_90 = signal_55 - signal_81
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_80 - signal_90, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_0 - signal_67 + signal_4 - signal_90 - signal_69, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
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
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_49 = signal_24 + signal_38
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_50 = signal_26 + signal_40
    signal_51 = signal_49 + signal_50
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = signal_12 - signal_51
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_53 = signal_10 + signal_52
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_54 = signal_22 + signal_36
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_55 = signal_31 + signal_33
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = signal_25 + signal_39
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = signal_32 - signal_56
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_58 = signal_8 - signal_53
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = signal_29 - signal_55
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_60 = signal_17 + signal_24
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_12 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_16 + signal_60 + signal_23 + bias_12, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_61 = signal_54 - signal_58
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_62 = signal_19 + signal_26
    signal_63 = signal_23 + signal_49
    signal_64 = signal_60 + signal_62
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_65 = signal_5 - signal_51
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_66 = signal_27 + signal_41
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_67 = signal_3 + signal_65
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = signal_45 + signal_47
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_69 = signal_21 + signal_35
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_70 = signal_37 + signal_63
    signal_71 = signal_57 + signal_59
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_72 = signal_7 + signal_11
    signal_73 = signal_61 - signal_66
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_74 = signal_46 + signal_56
    signal_75 = signal_31 - signal_70
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_76 = signal_15 - signal_64
    signal_77 = signal_53 + signal_69
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_78 = signal_34 - signal_59
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_79 = signal_1 - signal_67
    signal_80 = signal_72 - signal_77
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_81 = signal_28 + signal_57
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_82 = signal_13 + signal_73
    signal_48 = tl.load(
        source + 48 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_0 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_68 + signal_48 - signal_43 - signal_82 + bias_0, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    bias_2 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_78 + signal_82 + bias_2, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_83 = signal_30 + signal_75
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_14 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_2 + signal_3 + signal_83 + bias_14, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_84 = signal_54 - signal_79
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_10 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_84 + signal_78 - signal_66 + signal_6 + bias_10, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_11 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_71 - signal_84 + signal_4 + bias_11, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_85 = signal_22 + signal_76
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_8 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_20 - signal_85 + signal_27 + bias_8, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_86 = signal_9 + signal_10
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_4 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_86 + signal_70 + signal_45 + signal_44 + bias_4, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    bias_6 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_86 + signal_83 + bias_6, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_87 = signal_11 - signal_61
    bias_3 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_87 + signal_71 + bias_3, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_88 = signal_68 - signal_74
    bias_1 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_88 + signal_43 - signal_87 + bias_1, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_5 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_80 + signal_42 - signal_88 + bias_5, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_89 = signal_18 + signal_25
    bias_9 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_89 + signal_85 + bias_9, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_13 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_14 - signal_64 + signal_89 + signal_21 + bias_13, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_90 = signal_55 - signal_81
    bias_7 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_80 - signal_90 + bias_7, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_15 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_0 - signal_67 + signal_4 - signal_90 - signal_69 + bias_15, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
