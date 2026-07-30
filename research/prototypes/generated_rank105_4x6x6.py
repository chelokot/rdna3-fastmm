import triton
import triton.language as tl

DIMENSIONS = (4, 6, 6)
RANK = 105
CERTIFICATE_SHA256 = "9427b13b5509d45df8c56294f2abe6bc4e478f2aa079a1925e600ca2183739b5"


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
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 60 * plane_elements + element_offsets, signal_6, mask=element_mask
    )
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 61 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 62 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    signal_8 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 63 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 64 * plane_elements + element_offsets, signal_0 + signal_1, mask=element_mask
    )
    tl.store(
        output + 65 * plane_elements + element_offsets, signal_0 + signal_2, mask=element_mask
    )
    tl.store(
        output + 67 * plane_elements + element_offsets, signal_1 + signal_6, mask=element_mask
    )
    signal_21 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 75 * plane_elements + element_offsets, signal_21, mask=element_mask
    )
    signal_16 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 76 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    signal_17 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 77 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    signal_23 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 78 * plane_elements + element_offsets, signal_23, mask=element_mask
    )
    signal_15 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 79 * plane_elements + element_offsets, signal_15 + signal_16, mask=element_mask
    )
    tl.store(
        output + 80 * plane_elements + element_offsets, signal_15 + signal_17, mask=element_mask
    )
    tl.store(
        output + 82 * plane_elements + element_offsets, signal_16 + signal_21, mask=element_mask
    )
    signal_22 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_24 = signal_16 - signal_22
    tl.store(
        output + 84 * plane_elements + element_offsets, signal_24, mask=element_mask
    )
    signal_10 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_25 = signal_10 + signal_11
    signal_4 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_26 = signal_4 - signal_25
    signal_27 = signal_4 - signal_10
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_27 - signal_24, mask=element_mask
    )
    signal_28 = signal_4 - signal_16
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_28, mask=element_mask
    )
    signal_13 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_19 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_29 = signal_13 - signal_19
    tl.store(
        output + 99 * plane_elements + element_offsets, signal_29 + signal_24, mask=element_mask
    )
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_9 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_30 = signal_3 - signal_9
    signal_31 = signal_9 + signal_10
    signal_32 = signal_13 + signal_16
    tl.store(
        output + 91 * plane_elements + element_offsets, signal_32, mask=element_mask
    )
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_33 = signal_1 - signal_7
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_33 + signal_24, mask=element_mask
    )
    tl.store(
        output + 39 * plane_elements + element_offsets, signal_33 - signal_29, mask=element_mask
    )
    tl.store(
        output + 54 * plane_elements + element_offsets, signal_33 + signal_27, mask=element_mask
    )
    tl.store(
        output + 69 * plane_elements + element_offsets, signal_33, mask=element_mask
    )
    signal_34 = signal_1 + signal_4
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    signal_35 = signal_1 - signal_13
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_35, mask=element_mask
    )
    signal_36 = signal_1 + signal_16
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_36, mask=element_mask
    )
    signal_37 = signal_0 - signal_6
    tl.store(
        output + 66 * plane_elements + element_offsets, signal_37, mask=element_mask
    )
    signal_38 = signal_6 + signal_7
    tl.store(
        output + 53 * plane_elements + element_offsets, signal_38 + signal_31, mask=element_mask
    )
    tl.store(
        output + 68 * plane_elements + element_offsets, signal_38, mask=element_mask
    )
    signal_39 = signal_30 + signal_37
    tl.store(
        output + 51 * plane_elements + element_offsets, signal_39, mask=element_mask
    )
    signal_40 = signal_8 - signal_33
    tl.store(
        output + 72 * plane_elements + element_offsets, -signal_40, mask=element_mask
    )
    tl.store(
        output + 73 * plane_elements + element_offsets, -signal_40 + signal_2, mask=element_mask
    )
    signal_41 = signal_15 - signal_21
    tl.store(
        output + 81 * plane_elements + element_offsets, signal_41, mask=element_mask
    )
    signal_42 = signal_21 + signal_22
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_38 + signal_42, mask=element_mask
    )
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_31 - signal_42, mask=element_mask
    )
    tl.store(
        output + 83 * plane_elements + element_offsets, signal_42, mask=element_mask
    )
    signal_43 = signal_23 - signal_24
    tl.store(
        output + 87 * plane_elements + element_offsets, -signal_43, mask=element_mask
    )
    tl.store(
        output + 88 * plane_elements + element_offsets, -signal_43 + signal_17, mask=element_mask
    )
    signal_44 = signal_37 + signal_41
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_44, mask=element_mask
    )
    signal_45 = signal_7 + signal_8
    tl.store(
        output + 55 * plane_elements + element_offsets, signal_45 + signal_25, mask=element_mask
    )
    tl.store(
        output + 70 * plane_elements + element_offsets, signal_45, mask=element_mask
    )
    signal_20 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_46 = signal_20 - signal_29
    signal_47 = signal_22 + signal_23
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_45 + signal_47, mask=element_mask
    )
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_25 - signal_47, mask=element_mask
    )
    tl.store(
        output + 85 * plane_elements + element_offsets, signal_47, mask=element_mask
    )
    signal_48 = signal_40 - signal_46
    tl.store(
        output + 42 * plane_elements + element_offsets, -signal_48, mask=element_mask
    )
    signal_49 = signal_8 + signal_23
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_49, mask=element_mask
    )
    signal_50 = signal_6 + signal_21
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_50, mask=element_mask
    )
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_36 + signal_50, mask=element_mask
    )
    signal_51 = signal_6 + signal_9
    tl.store(
        output + 45 * plane_elements + element_offsets, signal_51, mask=element_mask
    )
    tl.store(
        output + 52 * plane_elements + element_offsets, signal_34 + signal_51, mask=element_mask
    )
    signal_52 = signal_8 - signal_20
    tl.store(
        output + 33 * plane_elements + element_offsets, signal_52, mask=element_mask
    )
    signal_53 = signal_43 + signal_46
    tl.store(
        output + 102 * plane_elements + element_offsets, -signal_53, mask=element_mask
    )
    signal_54 = signal_9 - signal_21
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_54, mask=element_mask
    )
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_28 + signal_54, mask=element_mask
    )
    signal_55 = signal_19 + signal_20
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_45 - signal_55, mask=element_mask
    )
    tl.store(
        output + 100 * plane_elements + element_offsets, signal_55 + signal_47, mask=element_mask
    )
    signal_56 = signal_20 + signal_23
    tl.store(
        output + 93 * plane_elements + element_offsets, signal_56, mask=element_mask
    )
    signal_57 = signal_0 + signal_3
    tl.store(
        output + 49 * plane_elements + element_offsets, signal_57 + signal_34, mask=element_mask
    )
    signal_12 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_18 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_58 = signal_12 - signal_18
    signal_59 = signal_26 + signal_43
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_59, mask=element_mask
    )
    signal_60 = signal_6 - signal_18
    tl.store(
        output + 30 * plane_elements + element_offsets, signal_60, mask=element_mask
    )
    tl.store(
        output + 37 * plane_elements + element_offsets, signal_35 + signal_60, mask=element_mask
    )
    signal_61 = signal_18 + signal_19
    tl.store(
        output + 38 * plane_elements + element_offsets, signal_38 - signal_61, mask=element_mask
    )
    tl.store(
        output + 98 * plane_elements + element_offsets, signal_61 + signal_42, mask=element_mask
    )
    signal_5 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_62 = signal_5 - signal_17
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_62, mask=element_mask
    )
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_59 + signal_62, mask=element_mask
    )
    signal_63 = signal_37 - signal_58
    tl.store(
        output + 36 * plane_elements + element_offsets, signal_63, mask=element_mask
    )
    signal_64 = signal_2 + signal_5
    tl.store(
        output + 47 * plane_elements + element_offsets, signal_64, mask=element_mask
    )
    tl.store(
        output + 50 * plane_elements + element_offsets, signal_57 + signal_64, mask=element_mask
    )
    signal_14 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_65 = signal_14 + signal_17
    tl.store(
        output + 92 * plane_elements + element_offsets, signal_65, mask=element_mask
    )
    tl.store(
        output + 103 * plane_elements + element_offsets, -signal_53 + signal_65, mask=element_mask
    )
    signal_66 = signal_2 - signal_14
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_66, mask=element_mask
    )
    tl.store(
        output + 43 * plane_elements + element_offsets, -signal_48 + signal_66, mask=element_mask
    )
    signal_67 = signal_2 + signal_17
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_67, mask=element_mask
    )
    signal_68 = signal_30 - signal_41
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_68, mask=element_mask
    )
    signal_69 = signal_41 + signal_58
    tl.store(
        output + 96 * plane_elements + element_offsets, signal_69, mask=element_mask
    )
    signal_70 = signal_12 + signal_15
    tl.store(
        output + 94 * plane_elements + element_offsets, signal_70 + signal_32, mask=element_mask
    )
    tl.store(
        output + 95 * plane_elements + element_offsets, signal_70 + signal_65, mask=element_mask
    )
    signal_71 = signal_0 + signal_15
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_71 + signal_36, mask=element_mask
    )
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_71 + signal_67, mask=element_mask
    )
    signal_72 = signal_40 + signal_43
    tl.store(
        output + 12 * plane_elements + element_offsets, -signal_72, mask=element_mask
    )
    tl.store(
        output + 13 * plane_elements + element_offsets, -signal_72 + signal_67, mask=element_mask
    )
    signal_73 = signal_11 - signal_23
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_73, mask=element_mask
    )
    signal_74 = signal_44 + signal_67
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_74, mask=element_mask
    )
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_74 - signal_49, mask=element_mask
    )
    signal_75 = signal_3 - signal_15
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_75 + signal_28, mask=element_mask
    )
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_75 + signal_62, mask=element_mask
    )
    signal_76 = signal_62 + signal_68
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_76, mask=element_mask
    )
    tl.store(
        output + 29 * plane_elements + element_offsets, -signal_73 + signal_76, mask=element_mask
    )
    signal_77 = signal_0 - signal_12
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_77 + signal_35, mask=element_mask
    )
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_77 + signal_66, mask=element_mask
    )
    signal_78 = signal_63 + signal_66
    tl.store(
        output + 41 * plane_elements + element_offsets, signal_78, mask=element_mask
    )
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_78 - signal_52, mask=element_mask
    )
    signal_79 = signal_8 + signal_11
    tl.store(
        output + 48 * plane_elements + element_offsets, signal_79, mask=element_mask
    )
    signal_80 = signal_39 + signal_64
    tl.store(
        output + 56 * plane_elements + element_offsets, signal_80, mask=element_mask
    )
    tl.store(
        output + 59 * plane_elements + element_offsets, -signal_79 + signal_80, mask=element_mask
    )
    signal_81 = signal_26 - signal_40
    tl.store(
        output + 57 * plane_elements + element_offsets, signal_81, mask=element_mask
    )
    tl.store(
        output + 58 * plane_elements + element_offsets, signal_81 + signal_64, mask=element_mask
    )
    signal_82 = signal_2 + signal_37
    tl.store(
        output + 71 * plane_elements + element_offsets, signal_82, mask=element_mask
    )
    tl.store(
        output + 74 * plane_elements + element_offsets, -signal_8 + signal_82, mask=element_mask
    )
    signal_83 = signal_17 + signal_41
    tl.store(
        output + 86 * plane_elements + element_offsets, signal_83, mask=element_mask
    )
    tl.store(
        output + 89 * plane_elements + element_offsets, -signal_23 + signal_83, mask=element_mask
    )
    signal_84 = signal_18 + signal_21
    tl.store(
        output + 90 * plane_elements + element_offsets, signal_84, mask=element_mask
    )
    tl.store(
        output + 97 * plane_elements + element_offsets, signal_32 + signal_84, mask=element_mask
    )
    signal_85 = signal_65 + signal_69
    tl.store(
        output + 101 * plane_elements + element_offsets, signal_85, mask=element_mask
    )
    tl.store(
        output + 104 * plane_elements + element_offsets, -signal_56 + signal_85, mask=element_mask
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
    signal_22 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_1 + signal_22, mask=element_mask
    )
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_27 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_6 + signal_27, mask=element_mask
    )
    signal_8 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_29 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_8 + signal_29, mask=element_mask
    )
    signal_13 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_34 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_13 + signal_34, mask=element_mask
    )
    signal_26 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_26 + signal_29, mask=element_mask
    )
    signal_31 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_31 + signal_34, mask=element_mask
    )
    signal_11 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_8 + signal_11, mask=element_mask
    )
    signal_16 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 43 * plane_elements + element_offsets, signal_13 + signal_16, mask=element_mask
    )
    tl.store(
        output + 49 * plane_elements + element_offsets, signal_22, mask=element_mask
    )
    signal_23 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 50 * plane_elements + element_offsets, signal_23, mask=element_mask
    )
    tl.store(
        output + 53 * plane_elements + element_offsets, signal_27, mask=element_mask
    )
    tl.store(
        output + 55 * plane_elements + element_offsets, signal_29, mask=element_mask
    )
    signal_33 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 56 * plane_elements + element_offsets, signal_23 - signal_33, mask=element_mask
    )
    tl.store(
        output + 58 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    tl.store(
        output + 59 * plane_elements + element_offsets, signal_33, mask=element_mask
    )
    tl.store(
        output + 70 * plane_elements + element_offsets, signal_11 - signal_29, mask=element_mask
    )
    tl.store(
        output + 73 * plane_elements + element_offsets, signal_16 - signal_34, mask=element_mask
    )
    tl.store(
        output + 85 * plane_elements + element_offsets, signal_8 - signal_26, mask=element_mask
    )
    tl.store(
        output + 88 * plane_elements + element_offsets, signal_13 - signal_31, mask=element_mask
    )
    tl.store(
        output + 94 * plane_elements + element_offsets, signal_1, mask=element_mask
    )
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 95 * plane_elements + element_offsets, signal_2, mask=element_mask
    )
    tl.store(
        output + 98 * plane_elements + element_offsets, signal_6, mask=element_mask
    )
    tl.store(
        output + 100 * plane_elements + element_offsets, signal_8, mask=element_mask
    )
    signal_12 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    tl.store(
        output + 101 * plane_elements + element_offsets, signal_2 - signal_12, mask=element_mask
    )
    tl.store(
        output + 103 * plane_elements + element_offsets, signal_13, mask=element_mask
    )
    tl.store(
        output + 104 * plane_elements + element_offsets, signal_12, mask=element_mask
    )
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_5 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_36 = signal_3 - signal_5
    signal_20 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_32 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_37 = signal_20 - signal_32
    signal_24 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_38 = signal_20 + signal_24
    signal_39 = signal_26 - signal_32
    signal_25 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_40 = signal_25 - signal_26
    signal_10 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_41 = signal_10 - signal_11
    signal_17 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_42 = signal_16 - signal_17
    signal_35 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 5 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (5 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_43 = signal_23 - signal_35
    signal_44 = signal_25 - signal_31
    signal_45 = signal_26 - signal_31
    signal_46 = signal_5 + signal_42
    signal_47 = signal_31 + signal_37
    signal_18 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_48 = signal_18 - signal_38
    signal_49 = signal_10 - signal_16
    signal_50 = signal_34 + signal_43
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_50 + signal_47, mask=element_mask
    )
    tl.store(
        output + 47 * plane_elements + element_offsets, signal_50, mask=element_mask
    )
    tl.store(
        output + 62 * plane_elements + element_offsets, signal_46 - signal_50, mask=element_mask
    )
    signal_21 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_51 = signal_21 - signal_23
    signal_4 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_52 = signal_4 - signal_41
    signal_19 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_53 = signal_19 - signal_40
    signal_54 = signal_20 + signal_23
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_54, mask=element_mask
    )
    signal_9 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_55 = signal_9 - signal_36
    signal_56 = signal_5 - signal_23
    tl.store(
        output + 65 * plane_elements + element_offsets, signal_56, mask=element_mask
    )
    signal_14 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_57 = signal_13 - signal_14
    signal_58 = signal_11 - signal_17
    signal_59 = signal_11 - signal_16
    signal_30 = tl.load(
        source + (5 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (5 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_60 = signal_30 + signal_39
    signal_15 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_61 = signal_15 + signal_58
    signal_62 = signal_18 - signal_19
    signal_63 = signal_30 - signal_62
    signal_64 = signal_4 + signal_15
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_65 = signal_0 - signal_2
    signal_66 = signal_3 - signal_64
    signal_67 = signal_29 - signal_35
    signal_68 = signal_2 + signal_57
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_68 + signal_50, mask=element_mask
    )
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_68 + signal_46, mask=element_mask
    )
    tl.store(
        output + 77 * plane_elements + element_offsets, signal_68 - signal_47, mask=element_mask
    )
    tl.store(
        output + 92 * plane_elements + element_offsets, signal_68, mask=element_mask
    )
    signal_69 = signal_24 - signal_44
    signal_70 = signal_6 + signal_13
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_71 = signal_7 - signal_70
    tl.store(
        output + 84 * plane_elements + element_offsets, -signal_69 - signal_71, mask=element_mask
    )
    tl.store(
        output + 99 * plane_elements + element_offsets, -signal_71, mask=element_mask
    )
    signal_72 = signal_9 - signal_49
    tl.store(
        output + 39 * plane_elements + element_offsets, signal_72 - signal_71, mask=element_mask
    )
    signal_73 = signal_6 - signal_65
    tl.store(
        output + 30 * plane_elements + element_offsets, -signal_73 - signal_55, mask=element_mask
    )
    tl.store(
        output + 75 * plane_elements + element_offsets, -signal_73 - signal_48, mask=element_mask
    )
    tl.store(
        output + 90 * plane_elements + element_offsets, -signal_73, mask=element_mask
    )
    signal_74 = signal_6 + signal_9
    tl.store(
        output + 38 * plane_elements + element_offsets, signal_74, mask=element_mask
    )
    signal_75 = signal_2 + signal_5
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_75, mask=element_mask
    )
    signal_76 = signal_27 - signal_51
    tl.store(
        output + 0 * plane_elements + element_offsets, -signal_73 - signal_76, mask=element_mask
    )
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_48 - signal_76, mask=element_mask
    )
    tl.store(
        output + 45 * plane_elements + element_offsets, -signal_76, mask=element_mask
    )
    tl.store(
        output + 60 * plane_elements + element_offsets, -signal_55 + signal_76, mask=element_mask
    )
    signal_77 = signal_33 + signal_67
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_77 + signal_60, mask=element_mask
    )
    tl.store(
        output + 48 * plane_elements + element_offsets, signal_77, mask=element_mask
    )
    tl.store(
        output + 63 * plane_elements + element_offsets, -signal_77 + signal_61, mask=element_mask
    )
    signal_78 = signal_21 - signal_33
    signal_79 = signal_15 - signal_33
    tl.store(
        output + 71 * plane_elements + element_offsets, signal_56 - signal_79, mask=element_mask
    )
    tl.store(
        output + 74 * plane_elements + element_offsets, signal_79, mask=element_mask
    )
    signal_80 = signal_30 + signal_33
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_54 - signal_80, mask=element_mask
    )
    tl.store(
        output + 29 * plane_elements + element_offsets, signal_80, mask=element_mask
    )
    signal_81 = signal_7 - signal_8
    signal_82 = signal_2 - signal_20
    tl.store(
        output + 80 * plane_elements + element_offsets, signal_82, mask=element_mask
    )
    signal_83 = signal_2 + signal_23
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_83, mask=element_mask
    )
    signal_84 = signal_6 - signal_24
    tl.store(
        output + 83 * plane_elements + element_offsets, signal_84, mask=element_mask
    )
    signal_85 = signal_1 - signal_81
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_85 + signal_52, mask=element_mask
    )
    tl.store(
        output + 76 * plane_elements + element_offsets, signal_85 - signal_53, mask=element_mask
    )
    tl.store(
        output + 91 * plane_elements + element_offsets, signal_85, mask=element_mask
    )
    signal_86 = signal_8 - signal_13
    tl.store(
        output + 42 * plane_elements + element_offsets, signal_86 + signal_59, mask=element_mask
    )
    tl.store(
        output + 87 * plane_elements + element_offsets, signal_86 - signal_45, mask=element_mask
    )
    tl.store(
        output + 102 * plane_elements + element_offsets, signal_86, mask=element_mask
    )
    signal_87 = signal_24 + signal_27
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_87, mask=element_mask
    )
    signal_88 = signal_27 + signal_34
    signal_89 = signal_0 - signal_12
    signal_90 = signal_12 - signal_14
    signal_91 = signal_1 - signal_89
    tl.store(
        output + 36 * plane_elements + element_offsets, signal_66 - signal_91, mask=element_mask
    )
    tl.store(
        output + 81 * plane_elements + element_offsets, signal_63 - signal_91, mask=element_mask
    )
    tl.store(
        output + 96 * plane_elements + element_offsets, -signal_91, mask=element_mask
    )
    signal_92 = signal_8 + signal_90
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_92 + signal_77, mask=element_mask
    )
    tl.store(
        output + 33 * plane_elements + element_offsets, signal_92 + signal_61, mask=element_mask
    )
    tl.store(
        output + 78 * plane_elements + element_offsets, signal_92 - signal_60, mask=element_mask
    )
    tl.store(
        output + 93 * plane_elements + element_offsets, signal_92, mask=element_mask
    )
    signal_93 = signal_29 - signal_34
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_86 + signal_93, mask=element_mask
    )
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_45 + signal_93, mask=element_mask
    )
    tl.store(
        output + 57 * plane_elements + element_offsets, signal_93, mask=element_mask
    )
    tl.store(
        output + 72 * plane_elements + element_offsets, signal_59 - signal_93, mask=element_mask
    )
    signal_28 = tl.load(
        source + (4 * block_height + local_rows) * source_columns + 4 * block_width + local_columns, mask=element_mask & (4 * block_height + local_rows < source_row_count) & (4 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_94 = signal_28 - signal_88
    tl.store(
        output + 9 * plane_elements + element_offsets, -signal_94 - signal_71, mask=element_mask
    )
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_69 - signal_94, mask=element_mask
    )
    tl.store(
        output + 54 * plane_elements + element_offsets, -signal_94, mask=element_mask
    )
    tl.store(
        output + 69 * plane_elements + element_offsets, signal_72 + signal_94, mask=element_mask
    )
    signal_95 = signal_22 - signal_78
    tl.store(
        output + 6 * plane_elements + element_offsets, -signal_95 - signal_91, mask=element_mask
    )
    tl.store(
        output + 21 * plane_elements + element_offsets, -signal_63 - signal_95, mask=element_mask
    )
    tl.store(
        output + 51 * plane_elements + element_offsets, -signal_95, mask=element_mask
    )
    tl.store(
        output + 66 * plane_elements + element_offsets, signal_66 + signal_95, mask=element_mask
    )
    signal_96 = signal_1 + signal_4
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_96, mask=element_mask
    )
    tl.store(
        output + 37 * plane_elements + element_offsets, signal_96 - signal_74, mask=element_mask
    )
    signal_97 = signal_28 - signal_29
    signal_98 = signal_12 + signal_33
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_83 - signal_98, mask=element_mask
    )
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_98, mask=element_mask
    )
    signal_99 = signal_1 - signal_6
    tl.store(
        output + 97 * plane_elements + element_offsets, signal_99, mask=element_mask
    )
    signal_100 = signal_12 - signal_30
    tl.store(
        output + 86 * plane_elements + element_offsets, signal_82 - signal_100, mask=element_mask
    )
    tl.store(
        output + 89 * plane_elements + element_offsets, signal_100, mask=element_mask
    )
    signal_101 = signal_1 - signal_19
    tl.store(
        output + 79 * plane_elements + element_offsets, signal_101, mask=element_mask
    )
    tl.store(
        output + 82 * plane_elements + element_offsets, signal_101 - signal_84, mask=element_mask
    )
    signal_102 = signal_22 - signal_97
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_85 + signal_102, mask=element_mask
    )
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_53 + signal_102, mask=element_mask
    )
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_102, mask=element_mask
    )
    tl.store(
        output + 61 * plane_elements + element_offsets, signal_52 - signal_102, mask=element_mask
    )
    signal_103 = signal_9 - signal_27
    tl.store(
        output + 68 * plane_elements + element_offsets, signal_103, mask=element_mask
    )
    signal_104 = signal_22 - signal_27
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_99 + signal_104, mask=element_mask
    )
    tl.store(
        output + 52 * plane_elements + element_offsets, signal_104, mask=element_mask
    )
    signal_105 = signal_19 + signal_22
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_105, mask=element_mask
    )
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_105 - signal_87, mask=element_mask
    )
    signal_106 = signal_12 + signal_15
    tl.store(
        output + 41 * plane_elements + element_offsets, signal_75 - signal_106, mask=element_mask
    )
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_106, mask=element_mask
    )
    signal_107 = signal_4 - signal_22
    tl.store(
        output + 64 * plane_elements + element_offsets, signal_107, mask=element_mask
    )
    tl.store(
        output + 67 * plane_elements + element_offsets, signal_107 - signal_103, mask=element_mask
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
    signal_22 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 4 * plane_elements + output_offsets,
        signal_1 + signal_22,
        mask=output_mask,
    )
    signal_6 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_27 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 8 * plane_elements + output_offsets,
        signal_6 + signal_27,
        mask=output_mask,
    )
    signal_8 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_29 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 10 * plane_elements + output_offsets,
        signal_8 + signal_29,
        mask=output_mask,
    )
    signal_13 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_34 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 13 * plane_elements + output_offsets,
        signal_13 + signal_34,
        mask=output_mask,
    )
    signal_26 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 25 * plane_elements + output_offsets,
        signal_26 + signal_29,
        mask=output_mask,
    )
    signal_31 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 28 * plane_elements + output_offsets,
        signal_31 + signal_34,
        mask=output_mask,
    )
    signal_11 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 40 * plane_elements + output_offsets,
        signal_8 + signal_11,
        mask=output_mask,
    )
    signal_16 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 43 * plane_elements + output_offsets,
        signal_13 + signal_16,
        mask=output_mask,
    )
    tl.store(
        output + 49 * plane_elements + output_offsets,
        signal_22,
        mask=output_mask,
    )
    signal_23 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 50 * plane_elements + output_offsets,
        signal_23,
        mask=output_mask,
    )
    tl.store(
        output + 53 * plane_elements + output_offsets,
        signal_27,
        mask=output_mask,
    )
    tl.store(
        output + 55 * plane_elements + output_offsets,
        signal_29,
        mask=output_mask,
    )
    signal_33 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 56 * plane_elements + output_offsets,
        signal_23 - signal_33,
        mask=output_mask,
    )
    tl.store(
        output + 58 * plane_elements + output_offsets,
        signal_34,
        mask=output_mask,
    )
    tl.store(
        output + 59 * plane_elements + output_offsets,
        signal_33,
        mask=output_mask,
    )
    tl.store(
        output + 70 * plane_elements + output_offsets,
        signal_11 - signal_29,
        mask=output_mask,
    )
    tl.store(
        output + 73 * plane_elements + output_offsets,
        signal_16 - signal_34,
        mask=output_mask,
    )
    tl.store(
        output + 85 * plane_elements + output_offsets,
        signal_8 - signal_26,
        mask=output_mask,
    )
    tl.store(
        output + 88 * plane_elements + output_offsets,
        signal_13 - signal_31,
        mask=output_mask,
    )
    tl.store(
        output + 94 * plane_elements + output_offsets,
        signal_1,
        mask=output_mask,
    )
    signal_2 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 95 * plane_elements + output_offsets,
        signal_2,
        mask=output_mask,
    )
    tl.store(
        output + 98 * plane_elements + output_offsets,
        signal_6,
        mask=output_mask,
    )
    tl.store(
        output + 100 * plane_elements + output_offsets,
        signal_8,
        mask=output_mask,
    )
    signal_12 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    tl.store(
        output + 101 * plane_elements + output_offsets,
        signal_2 - signal_12,
        mask=output_mask,
    )
    tl.store(
        output + 103 * plane_elements + output_offsets,
        signal_13,
        mask=output_mask,
    )
    tl.store(
        output + 104 * plane_elements + output_offsets,
        signal_12,
        mask=output_mask,
    )
    signal_3 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_5 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_36 = signal_3 - signal_5
    signal_20 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_32 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_37 = signal_20 - signal_32
    signal_24 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_38 = signal_20 + signal_24
    signal_39 = signal_26 - signal_32
    signal_25 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_40 = signal_25 - signal_26
    signal_10 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_41 = signal_10 - signal_11
    signal_17 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_42 = signal_16 - signal_17
    signal_35 = tl.trans(
        tl.load(
            source + (5 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(5 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_43 = signal_23 - signal_35
    signal_44 = signal_25 - signal_31
    signal_45 = signal_26 - signal_31
    signal_46 = signal_5 + signal_42
    signal_47 = signal_31 + signal_37
    signal_18 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_48 = signal_18 - signal_38
    signal_49 = signal_10 - signal_16
    signal_50 = signal_34 + signal_43
    tl.store(
        output + 17 * plane_elements + output_offsets,
        signal_50 + signal_47,
        mask=output_mask,
    )
    tl.store(
        output + 47 * plane_elements + output_offsets,
        signal_50,
        mask=output_mask,
    )
    tl.store(
        output + 62 * plane_elements + output_offsets,
        signal_46 - signal_50,
        mask=output_mask,
    )
    signal_21 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_51 = signal_21 - signal_23
    signal_4 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_52 = signal_4 - signal_41
    signal_19 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_53 = signal_19 - signal_40
    signal_54 = signal_20 + signal_23
    tl.store(
        output + 20 * plane_elements + output_offsets,
        signal_54,
        mask=output_mask,
    )
    signal_9 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_55 = signal_9 - signal_36
    signal_56 = signal_5 - signal_23
    tl.store(
        output + 65 * plane_elements + output_offsets,
        signal_56,
        mask=output_mask,
    )
    signal_14 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_57 = signal_13 - signal_14
    signal_58 = signal_11 - signal_17
    signal_59 = signal_11 - signal_16
    signal_30 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 5 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (5 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_60 = signal_30 + signal_39
    signal_15 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_61 = signal_15 + signal_58
    signal_62 = signal_18 - signal_19
    signal_63 = signal_30 - signal_62
    signal_64 = signal_4 + signal_15
    signal_0 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_65 = signal_0 - signal_2
    signal_66 = signal_3 - signal_64
    signal_67 = signal_29 - signal_35
    signal_68 = signal_2 + signal_57
    tl.store(
        output + 2 * plane_elements + output_offsets,
        signal_68 + signal_50,
        mask=output_mask,
    )
    tl.store(
        output + 32 * plane_elements + output_offsets,
        signal_68 + signal_46,
        mask=output_mask,
    )
    tl.store(
        output + 77 * plane_elements + output_offsets,
        signal_68 - signal_47,
        mask=output_mask,
    )
    tl.store(
        output + 92 * plane_elements + output_offsets,
        signal_68,
        mask=output_mask,
    )
    signal_69 = signal_24 - signal_44
    signal_70 = signal_6 + signal_13
    signal_7 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_71 = signal_7 - signal_70
    tl.store(
        output + 84 * plane_elements + output_offsets,
        -signal_69 - signal_71,
        mask=output_mask,
    )
    tl.store(
        output + 99 * plane_elements + output_offsets,
        -signal_71,
        mask=output_mask,
    )
    signal_72 = signal_9 - signal_49
    tl.store(
        output + 39 * plane_elements + output_offsets,
        signal_72 - signal_71,
        mask=output_mask,
    )
    signal_73 = signal_6 - signal_65
    tl.store(
        output + 30 * plane_elements + output_offsets,
        -signal_73 - signal_55,
        mask=output_mask,
    )
    tl.store(
        output + 75 * plane_elements + output_offsets,
        -signal_73 - signal_48,
        mask=output_mask,
    )
    tl.store(
        output + 90 * plane_elements + output_offsets,
        -signal_73,
        mask=output_mask,
    )
    signal_74 = signal_6 + signal_9
    tl.store(
        output + 38 * plane_elements + output_offsets,
        signal_74,
        mask=output_mask,
    )
    signal_75 = signal_2 + signal_5
    tl.store(
        output + 35 * plane_elements + output_offsets,
        signal_75,
        mask=output_mask,
    )
    signal_76 = signal_27 - signal_51
    tl.store(
        output + 0 * plane_elements + output_offsets,
        -signal_73 - signal_76,
        mask=output_mask,
    )
    tl.store(
        output + 15 * plane_elements + output_offsets,
        signal_48 - signal_76,
        mask=output_mask,
    )
    tl.store(
        output + 45 * plane_elements + output_offsets,
        -signal_76,
        mask=output_mask,
    )
    tl.store(
        output + 60 * plane_elements + output_offsets,
        -signal_55 + signal_76,
        mask=output_mask,
    )
    signal_77 = signal_33 + signal_67
    tl.store(
        output + 18 * plane_elements + output_offsets,
        signal_77 + signal_60,
        mask=output_mask,
    )
    tl.store(
        output + 48 * plane_elements + output_offsets,
        signal_77,
        mask=output_mask,
    )
    tl.store(
        output + 63 * plane_elements + output_offsets,
        -signal_77 + signal_61,
        mask=output_mask,
    )
    signal_78 = signal_21 - signal_33
    signal_79 = signal_15 - signal_33
    tl.store(
        output + 71 * plane_elements + output_offsets,
        signal_56 - signal_79,
        mask=output_mask,
    )
    tl.store(
        output + 74 * plane_elements + output_offsets,
        signal_79,
        mask=output_mask,
    )
    signal_80 = signal_30 + signal_33
    tl.store(
        output + 26 * plane_elements + output_offsets,
        signal_54 - signal_80,
        mask=output_mask,
    )
    tl.store(
        output + 29 * plane_elements + output_offsets,
        signal_80,
        mask=output_mask,
    )
    signal_81 = signal_7 - signal_8
    signal_82 = signal_2 - signal_20
    tl.store(
        output + 80 * plane_elements + output_offsets,
        signal_82,
        mask=output_mask,
    )
    signal_83 = signal_2 + signal_23
    tl.store(
        output + 5 * plane_elements + output_offsets,
        signal_83,
        mask=output_mask,
    )
    signal_84 = signal_6 - signal_24
    tl.store(
        output + 83 * plane_elements + output_offsets,
        signal_84,
        mask=output_mask,
    )
    signal_85 = signal_1 - signal_81
    tl.store(
        output + 31 * plane_elements + output_offsets,
        signal_85 + signal_52,
        mask=output_mask,
    )
    tl.store(
        output + 76 * plane_elements + output_offsets,
        signal_85 - signal_53,
        mask=output_mask,
    )
    tl.store(
        output + 91 * plane_elements + output_offsets,
        signal_85,
        mask=output_mask,
    )
    signal_86 = signal_8 - signal_13
    tl.store(
        output + 42 * plane_elements + output_offsets,
        signal_86 + signal_59,
        mask=output_mask,
    )
    tl.store(
        output + 87 * plane_elements + output_offsets,
        signal_86 - signal_45,
        mask=output_mask,
    )
    tl.store(
        output + 102 * plane_elements + output_offsets,
        signal_86,
        mask=output_mask,
    )
    signal_87 = signal_24 + signal_27
    tl.store(
        output + 23 * plane_elements + output_offsets,
        signal_87,
        mask=output_mask,
    )
    signal_88 = signal_27 + signal_34
    signal_89 = signal_0 - signal_12
    signal_90 = signal_12 - signal_14
    signal_91 = signal_1 - signal_89
    tl.store(
        output + 36 * plane_elements + output_offsets,
        signal_66 - signal_91,
        mask=output_mask,
    )
    tl.store(
        output + 81 * plane_elements + output_offsets,
        signal_63 - signal_91,
        mask=output_mask,
    )
    tl.store(
        output + 96 * plane_elements + output_offsets,
        -signal_91,
        mask=output_mask,
    )
    signal_92 = signal_8 + signal_90
    tl.store(
        output + 3 * plane_elements + output_offsets,
        signal_92 + signal_77,
        mask=output_mask,
    )
    tl.store(
        output + 33 * plane_elements + output_offsets,
        signal_92 + signal_61,
        mask=output_mask,
    )
    tl.store(
        output + 78 * plane_elements + output_offsets,
        signal_92 - signal_60,
        mask=output_mask,
    )
    tl.store(
        output + 93 * plane_elements + output_offsets,
        signal_92,
        mask=output_mask,
    )
    signal_93 = signal_29 - signal_34
    tl.store(
        output + 12 * plane_elements + output_offsets,
        signal_86 + signal_93,
        mask=output_mask,
    )
    tl.store(
        output + 27 * plane_elements + output_offsets,
        signal_45 + signal_93,
        mask=output_mask,
    )
    tl.store(
        output + 57 * plane_elements + output_offsets,
        signal_93,
        mask=output_mask,
    )
    tl.store(
        output + 72 * plane_elements + output_offsets,
        signal_59 - signal_93,
        mask=output_mask,
    )
    signal_28 = tl.trans(
        tl.load(
            source + (4 * block_width + local_columns[:, None]) * source_columns + 4 * block_height + local_rows[None, :], mask=(4 * block_width + local_columns[:, None] < source_row_count) & (4 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_94 = signal_28 - signal_88
    tl.store(
        output + 9 * plane_elements + output_offsets,
        -signal_94 - signal_71,
        mask=output_mask,
    )
    tl.store(
        output + 24 * plane_elements + output_offsets,
        signal_69 - signal_94,
        mask=output_mask,
    )
    tl.store(
        output + 54 * plane_elements + output_offsets,
        -signal_94,
        mask=output_mask,
    )
    tl.store(
        output + 69 * plane_elements + output_offsets,
        signal_72 + signal_94,
        mask=output_mask,
    )
    signal_95 = signal_22 - signal_78
    tl.store(
        output + 6 * plane_elements + output_offsets,
        -signal_95 - signal_91,
        mask=output_mask,
    )
    tl.store(
        output + 21 * plane_elements + output_offsets,
        -signal_63 - signal_95,
        mask=output_mask,
    )
    tl.store(
        output + 51 * plane_elements + output_offsets,
        -signal_95,
        mask=output_mask,
    )
    tl.store(
        output + 66 * plane_elements + output_offsets,
        signal_66 + signal_95,
        mask=output_mask,
    )
    signal_96 = signal_1 + signal_4
    tl.store(
        output + 34 * plane_elements + output_offsets,
        signal_96,
        mask=output_mask,
    )
    tl.store(
        output + 37 * plane_elements + output_offsets,
        signal_96 - signal_74,
        mask=output_mask,
    )
    signal_97 = signal_28 - signal_29
    signal_98 = signal_12 + signal_33
    tl.store(
        output + 11 * plane_elements + output_offsets,
        signal_83 - signal_98,
        mask=output_mask,
    )
    tl.store(
        output + 14 * plane_elements + output_offsets,
        signal_98,
        mask=output_mask,
    )
    signal_99 = signal_1 - signal_6
    tl.store(
        output + 97 * plane_elements + output_offsets,
        signal_99,
        mask=output_mask,
    )
    signal_100 = signal_12 - signal_30
    tl.store(
        output + 86 * plane_elements + output_offsets,
        signal_82 - signal_100,
        mask=output_mask,
    )
    tl.store(
        output + 89 * plane_elements + output_offsets,
        signal_100,
        mask=output_mask,
    )
    signal_101 = signal_1 - signal_19
    tl.store(
        output + 79 * plane_elements + output_offsets,
        signal_101,
        mask=output_mask,
    )
    tl.store(
        output + 82 * plane_elements + output_offsets,
        signal_101 - signal_84,
        mask=output_mask,
    )
    signal_102 = signal_22 - signal_97
    tl.store(
        output + 1 * plane_elements + output_offsets,
        signal_85 + signal_102,
        mask=output_mask,
    )
    tl.store(
        output + 16 * plane_elements + output_offsets,
        signal_53 + signal_102,
        mask=output_mask,
    )
    tl.store(
        output + 46 * plane_elements + output_offsets,
        signal_102,
        mask=output_mask,
    )
    tl.store(
        output + 61 * plane_elements + output_offsets,
        signal_52 - signal_102,
        mask=output_mask,
    )
    signal_103 = signal_9 - signal_27
    tl.store(
        output + 68 * plane_elements + output_offsets,
        signal_103,
        mask=output_mask,
    )
    signal_104 = signal_22 - signal_27
    tl.store(
        output + 7 * plane_elements + output_offsets,
        signal_99 + signal_104,
        mask=output_mask,
    )
    tl.store(
        output + 52 * plane_elements + output_offsets,
        signal_104,
        mask=output_mask,
    )
    signal_105 = signal_19 + signal_22
    tl.store(
        output + 19 * plane_elements + output_offsets,
        signal_105,
        mask=output_mask,
    )
    tl.store(
        output + 22 * plane_elements + output_offsets,
        signal_105 - signal_87,
        mask=output_mask,
    )
    signal_106 = signal_12 + signal_15
    tl.store(
        output + 41 * plane_elements + output_offsets,
        signal_75 - signal_106,
        mask=output_mask,
    )
    tl.store(
        output + 44 * plane_elements + output_offsets,
        signal_106,
        mask=output_mask,
    )
    signal_107 = signal_4 - signal_22
    tl.store(
        output + 64 * plane_elements + output_offsets,
        signal_107,
        mask=output_mask,
    )
    tl.store(
        output + 67 * plane_elements + output_offsets,
        signal_107 - signal_103,
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
    signal_80 = tl.load(
        source + 80 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_86 = tl.load(
        source + 86 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_105 = signal_80 - signal_86
    signal_85 = tl.load(
        source + 85 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_87 = tl.load(
        source + 87 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_106 = signal_85 + signal_87
    signal_95 = tl.load(
        source + 95 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_101 = tl.load(
        source + 101 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_107 = signal_95 - signal_101
    signal_65 = tl.load(
        source + 65 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_71 = tl.load(
        source + 71 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_108 = signal_65 - signal_71
    signal_100 = tl.load(
        source + 100 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_102 = tl.load(
        source + 102 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_109 = signal_100 + signal_102
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_110 = signal_5 - signal_11
    signal_55 = tl.load(
        source + 55 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = tl.load(
        source + 57 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_111 = signal_55 + signal_57
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_112 = signal_10 + signal_12
    signal_50 = tl.load(
        source + 50 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = tl.load(
        source + 56 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_113 = signal_50 - signal_56
    signal_70 = tl.load(
        source + 70 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_72 = tl.load(
        source + 72 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_114 = signal_70 + signal_72
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_115 = signal_45 + signal_113
    signal_104 = tl.load(
        source + 104 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_116 = signal_104 - signal_107
    signal_89 = tl.load(
        source + 89 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_117 = signal_89 - signal_105
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_118 = signal_46 - signal_111
    signal_73 = tl.load(
        source + 73 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_119 = signal_73 + signal_114
    signal_74 = tl.load(
        source + 74 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_120 = signal_74 - signal_108
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_121 = signal_0 + signal_110
    signal_76 = tl.load(
        source + 76 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_122 = signal_76 - signal_106
    signal_91 = tl.load(
        source + 91 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_123 = signal_91 - signal_109
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_124 = signal_1 - signal_112
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_125 = signal_20 - signal_26
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_126 = signal_35 - signal_41
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_127 = signal_40 + signal_42
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_128 = signal_25 + signal_27
    signal_75 = tl.load(
        source + 75 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_82 = tl.load(
        source + 82 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_129 = signal_75 - signal_82
    signal_61 = tl.load(
        source + 61 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = tl.load(
        source + 68 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_130 = signal_61 - signal_68
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_131 = signal_2 - signal_5
    signal_77 = tl.load(
        source + 77 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_88 = tl.load(
        source + 88 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_132 = signal_77 - signal_88
    signal_94 = tl.load(
        source + 94 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_133 = signal_94 + signal_107
    signal_51 = tl.load(
        source + 51 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_134 = signal_51 + signal_115
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_135 = signal_30 - signal_121
    signal_62 = tl.load(
        source + 62 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_136 = signal_62 - signal_119
    signal_49 = tl.load(
        source + 49 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = tl.load(
        source + 52 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_137 = signal_49 - signal_52
    signal_97 = tl.load(
        source + 97 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_98 = tl.load(
        source + 98 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_138 = signal_97 + signal_98
    signal_64 = tl.load(
        source + 64 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_139 = signal_64 + signal_119
    signal_60 = tl.load(
        source + 60 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_140 = signal_60 - signal_120
    signal_83 = tl.load(
        source + 83 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_141 = signal_83 - signal_117
    signal_142 = signal_122 - signal_123
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_143 = signal_7 + signal_22
    signal_93 = tl.load(
        source + 93 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_144 = signal_93 + signal_116
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_145 = signal_3 - signal_110
    signal_58 = tl.load(
        source + 58 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_146 = signal_58 + signal_111
    signal_90 = tl.load(
        source + 90 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_147 = signal_90 - signal_97
    signal_148 = signal_123 - signal_124
    signal_149 = signal_124 - signal_128
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_150 = signal_13 - signal_43
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_151 = signal_14 + signal_29
    signal_54 = tl.load(
        source + 54 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_152 = signal_52 + signal_54
    signal_67 = tl.load(
        source + 67 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_153 = signal_60 - signal_67
    signal_92 = tl.load(
        source + 92 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_103 = tl.load(
        source + 103 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_154 = signal_92 - signal_103
    signal_66 = tl.load(
        source + 66 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_155 = signal_66 + signal_108
    signal_156 = signal_58 - signal_118
    signal_48 = tl.load(
        source + 48 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = tl.load(
        source + 59 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_157 = signal_48 + signal_59
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_158 = signal_15 + signal_121
    signal_79 = tl.load(
        source + 79 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_159 = signal_79 + signal_105
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_160 = signal_44 - signal_126
    signal_69 = tl.load(
        source + 69 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_161 = signal_67 + signal_69
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_162 = signal_8 + signal_23
    signal_163 = signal_127 - signal_148
    signal_164 = signal_129 + signal_159
    signal_165 = signal_133 + signal_147
    signal_166 = signal_90 + signal_98
    signal_167 = signal_59 - signal_115
    signal_53 = tl.load(
        source + 53 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_168 = signal_53 - signal_167
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_169 = signal_16 + signal_149
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_170 = signal_4 + signal_19
    signal_84 = tl.load(
        source + 84 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_171 = signal_83 + signal_84
    signal_172 = signal_118 - signal_152
    signal_78 = tl.load(
        source + 78 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_173 = signal_78 + signal_117
    signal_174 = signal_114 + signal_161
    signal_175 = signal_153 + signal_155
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_176 = signal_47 - signal_50
    signal_177 = signal_80 - signal_132
    signal_63 = tl.load(
        source + 63 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_178 = signal_63 + signal_120
    signal_179 = signal_55 + signal_113
    signal_180 = signal_95 + signal_109
    signal_181 = signal_125 + signal_158
    signal_182 = signal_134 + signal_137
    signal_81 = tl.load(
        source + 81 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_183 = signal_81 + signal_164
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_181 + signal_170 - signal_182 + signal_6 - signal_143 - signal_183 + signal_21, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_184 = signal_75 + signal_141
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_181 - signal_184 + signal_162 - signal_168 - signal_151, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_96 = tl.load(
        source + 96 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_185 = signal_96 + signal_165
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_183 + signal_185, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_186 = signal_116 - signal_166
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_184 - signal_186, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_187 = signal_49 + signal_156
    signal_188 = signal_79 + signal_88
    signal_189 = signal_122 - signal_169
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_190 = signal_13 + signal_28
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_187 + signal_170 - signal_188 + signal_189 + signal_190, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_191 = signal_53 - signal_172
    signal_192 = signal_82 + signal_171
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_191 + signal_143 + signal_162 + signal_9 + signal_189 - signal_192 + signal_24, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_193 = signal_94 + signal_103
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_142 - signal_188 + signal_193, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_99 = tl.load(
        source + 99 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_194 = signal_99 + signal_138
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_142 - signal_192 + signal_194, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_195 = signal_112 - signal_131
    signal_196 = signal_146 - signal_176
    signal_197 = signal_106 + signal_177
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_128 - signal_196 + signal_195 - signal_197 + signal_190 - signal_17 + signal_20, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_198 = signal_10 - signal_145
    signal_199 = signal_85 - signal_173
    signal_200 = signal_157 - signal_179
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_199 + signal_25 + signal_198 + signal_200 - signal_151 - signal_18 + signal_125, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_201 = signal_154 - signal_180
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_201 - signal_197, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_202 = signal_100 - signal_144
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_199 + signal_202, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_203 = signal_64 + signal_175
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_203 + signal_182, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_204 = signal_68 + signal_140
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_168 + signal_204, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_205 = signal_7 - signal_37
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_206 = signal_4 - signal_34
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, -signal_36 + signal_206 + signal_203 + signal_6 - signal_205 - signal_185 - signal_135 - signal_126, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_207 = signal_8 - signal_38
    signal_208 = signal_14 - signal_160
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_204 + signal_186 + signal_207 - signal_135 - signal_208, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_209 = signal_61 - signal_139
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_209 + signal_187, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_210 = signal_130 - signal_174
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_210 + signal_191, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_211 = signal_31 - signal_163
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, signal_211 + signal_206 - signal_193 - signal_209 + signal_150, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_194 + signal_205 + signal_207 + signal_9 - signal_210 - signal_39 + signal_211, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_212 = signal_65 - signal_136
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, signal_212 + signal_196, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, signal_201 - signal_127 + signal_195 + signal_212 + signal_150 + signal_32 - signal_35, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_213 = signal_70 - signal_178
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, -signal_200 + signal_213, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, -signal_40 + signal_213 + signal_198 - signal_202 - signal_208 + signal_33, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
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
    signal_80 = tl.load(
        source + 80 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_86 = tl.load(
        source + 86 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_105 = signal_80 - signal_86
    signal_85 = tl.load(
        source + 85 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_87 = tl.load(
        source + 87 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_106 = signal_85 + signal_87
    signal_95 = tl.load(
        source + 95 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_101 = tl.load(
        source + 101 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_107 = signal_95 - signal_101
    signal_65 = tl.load(
        source + 65 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_71 = tl.load(
        source + 71 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_108 = signal_65 - signal_71
    signal_100 = tl.load(
        source + 100 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_102 = tl.load(
        source + 102 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_109 = signal_100 + signal_102
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_110 = signal_5 - signal_11
    signal_55 = tl.load(
        source + 55 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = tl.load(
        source + 57 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_111 = signal_55 + signal_57
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_112 = signal_10 + signal_12
    signal_50 = tl.load(
        source + 50 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = tl.load(
        source + 56 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_113 = signal_50 - signal_56
    signal_70 = tl.load(
        source + 70 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_72 = tl.load(
        source + 72 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_114 = signal_70 + signal_72
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_115 = signal_45 + signal_113
    signal_104 = tl.load(
        source + 104 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_116 = signal_104 - signal_107
    signal_89 = tl.load(
        source + 89 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_117 = signal_89 - signal_105
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_118 = signal_46 - signal_111
    signal_73 = tl.load(
        source + 73 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_119 = signal_73 + signal_114
    signal_74 = tl.load(
        source + 74 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_120 = signal_74 - signal_108
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_121 = signal_0 + signal_110
    signal_76 = tl.load(
        source + 76 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_122 = signal_76 - signal_106
    signal_91 = tl.load(
        source + 91 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_123 = signal_91 - signal_109
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_124 = signal_1 - signal_112
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_125 = signal_20 - signal_26
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_126 = signal_35 - signal_41
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_127 = signal_40 + signal_42
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_128 = signal_25 + signal_27
    signal_75 = tl.load(
        source + 75 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_82 = tl.load(
        source + 82 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_129 = signal_75 - signal_82
    signal_61 = tl.load(
        source + 61 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = tl.load(
        source + 68 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_130 = signal_61 - signal_68
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_131 = signal_2 - signal_5
    signal_77 = tl.load(
        source + 77 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_88 = tl.load(
        source + 88 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_132 = signal_77 - signal_88
    signal_94 = tl.load(
        source + 94 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_133 = signal_94 + signal_107
    signal_51 = tl.load(
        source + 51 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_134 = signal_51 + signal_115
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_135 = signal_30 - signal_121
    signal_62 = tl.load(
        source + 62 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_136 = signal_62 - signal_119
    signal_49 = tl.load(
        source + 49 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = tl.load(
        source + 52 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_137 = signal_49 - signal_52
    signal_97 = tl.load(
        source + 97 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_98 = tl.load(
        source + 98 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_138 = signal_97 + signal_98
    signal_64 = tl.load(
        source + 64 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_139 = signal_64 + signal_119
    signal_60 = tl.load(
        source + 60 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_140 = signal_60 - signal_120
    signal_83 = tl.load(
        source + 83 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_141 = signal_83 - signal_117
    signal_142 = signal_122 - signal_123
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_143 = signal_7 + signal_22
    signal_93 = tl.load(
        source + 93 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_144 = signal_93 + signal_116
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_145 = signal_3 - signal_110
    signal_58 = tl.load(
        source + 58 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_146 = signal_58 + signal_111
    signal_90 = tl.load(
        source + 90 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_147 = signal_90 - signal_97
    signal_148 = signal_123 - signal_124
    signal_149 = signal_124 - signal_128
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_150 = signal_13 - signal_43
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_151 = signal_14 + signal_29
    signal_54 = tl.load(
        source + 54 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_152 = signal_52 + signal_54
    signal_67 = tl.load(
        source + 67 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_153 = signal_60 - signal_67
    signal_92 = tl.load(
        source + 92 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_103 = tl.load(
        source + 103 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_154 = signal_92 - signal_103
    signal_66 = tl.load(
        source + 66 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_155 = signal_66 + signal_108
    signal_156 = signal_58 - signal_118
    signal_48 = tl.load(
        source + 48 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = tl.load(
        source + 59 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_157 = signal_48 + signal_59
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_158 = signal_15 + signal_121
    signal_79 = tl.load(
        source + 79 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_159 = signal_79 + signal_105
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_160 = signal_44 - signal_126
    signal_69 = tl.load(
        source + 69 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_161 = signal_67 + signal_69
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_162 = signal_8 + signal_23
    signal_163 = signal_127 - signal_148
    signal_164 = signal_129 + signal_159
    signal_165 = signal_133 + signal_147
    signal_166 = signal_90 + signal_98
    signal_167 = signal_59 - signal_115
    signal_53 = tl.load(
        source + 53 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_168 = signal_53 - signal_167
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_169 = signal_16 + signal_149
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_170 = signal_4 + signal_19
    signal_84 = tl.load(
        source + 84 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_171 = signal_83 + signal_84
    signal_172 = signal_118 - signal_152
    signal_78 = tl.load(
        source + 78 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_173 = signal_78 + signal_117
    signal_174 = signal_114 + signal_161
    signal_175 = signal_153 + signal_155
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_176 = signal_47 - signal_50
    signal_177 = signal_80 - signal_132
    signal_63 = tl.load(
        source + 63 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_178 = signal_63 + signal_120
    signal_179 = signal_55 + signal_113
    signal_180 = signal_95 + signal_109
    signal_181 = signal_125 + signal_158
    signal_182 = signal_134 + signal_137
    signal_81 = tl.load(
        source + 81 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_183 = signal_81 + signal_164
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_0 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_181 + signal_170 - signal_182 + signal_6 - signal_143 - signal_183 + signal_21 + bias_0, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_184 = signal_75 + signal_141
    bias_1 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_181 - signal_184 + signal_162 - signal_168 - signal_151 + bias_1, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_96 = tl.load(
        source + 96 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_185 = signal_96 + signal_165
    bias_2 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_183 + signal_185 + bias_2, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_186 = signal_116 - signal_166
    bias_3 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, -signal_184 - signal_186 + bias_3, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_187 = signal_49 + signal_156
    signal_188 = signal_79 + signal_88
    signal_189 = signal_122 - signal_169
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_190 = signal_13 + signal_28
    bias_4 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_187 + signal_170 - signal_188 + signal_189 + signal_190 + bias_4, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_191 = signal_53 - signal_172
    signal_192 = signal_82 + signal_171
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_5 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, -signal_191 + signal_143 + signal_162 + signal_9 + signal_189 - signal_192 + signal_24 + bias_5, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_193 = signal_94 + signal_103
    bias_6 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_142 - signal_188 + signal_193 + bias_6, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_99 = tl.load(
        source + 99 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_194 = signal_99 + signal_138
    bias_7 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_142 - signal_192 + signal_194 + bias_7, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_195 = signal_112 - signal_131
    signal_196 = signal_146 - signal_176
    signal_197 = signal_106 + signal_177
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_8 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_128 - signal_196 + signal_195 - signal_197 + signal_190 - signal_17 + signal_20 + bias_8, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_198 = signal_10 - signal_145
    signal_199 = signal_85 - signal_173
    signal_200 = signal_157 - signal_179
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_9 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_199 + signal_25 + signal_198 + signal_200 - signal_151 - signal_18 + signal_125 + bias_9, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_201 = signal_154 - signal_180
    bias_10 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_201 - signal_197 + bias_10, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_202 = signal_100 - signal_144
    bias_11 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, -signal_199 + signal_202 + bias_11, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_203 = signal_64 + signal_175
    bias_12 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_203 + signal_182 + bias_12, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_204 = signal_68 + signal_140
    bias_13 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_168 + signal_204 + bias_13, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_205 = signal_7 - signal_37
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_206 = signal_4 - signal_34
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_14 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, -signal_36 + signal_206 + signal_203 + signal_6 - signal_205 - signal_185 - signal_135 - signal_126 + bias_14, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_207 = signal_8 - signal_38
    signal_208 = signal_14 - signal_160
    bias_15 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_204 + signal_186 + signal_207 - signal_135 - signal_208 + bias_15, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_209 = signal_61 - signal_139
    bias_16 = tl.load(
        bias
        + 4 * block_width
        + local_columns,
        mask=element_mask & (4 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_209 + signal_187 + bias_16, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_210 = signal_130 - signal_174
    bias_17 = tl.load(
        bias
        + 4 * block_width
        + local_columns,
        mask=element_mask & (4 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_210 + signal_191 + bias_17, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_211 = signal_31 - signal_163
    bias_18 = tl.load(
        bias
        + 4 * block_width
        + local_columns,
        mask=element_mask & (4 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, signal_211 + signal_206 - signal_193 - signal_209 + signal_150 + bias_18, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_19 = tl.load(
        bias
        + 4 * block_width
        + local_columns,
        mask=element_mask & (4 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 4 * block_width + local_columns, -signal_194 + signal_205 + signal_207 + signal_9 - signal_210 - signal_39 + signal_211 + bias_19, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (4 * block_width + local_columns < output_columns)
    )
    signal_212 = signal_65 - signal_136
    bias_20 = tl.load(
        bias
        + 5 * block_width
        + local_columns,
        mask=element_mask & (5 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, signal_212 + signal_196 + bias_20, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_22 = tl.load(
        bias
        + 5 * block_width
        + local_columns,
        mask=element_mask & (5 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, signal_201 - signal_127 + signal_195 + signal_212 + signal_150 + signal_32 - signal_35 + bias_22, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_213 = signal_70 - signal_178
    bias_21 = tl.load(
        bias
        + 5 * block_width
        + local_columns,
        mask=element_mask & (5 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, -signal_200 + signal_213 + bias_21, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    bias_23 = tl.load(
        bias
        + 5 * block_width
        + local_columns,
        mask=element_mask & (5 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 5 * block_width + local_columns, -signal_40 + signal_213 + signal_198 - signal_202 - signal_208 + signal_33 + bias_23, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (5 * block_width + local_columns < output_columns)
    )
