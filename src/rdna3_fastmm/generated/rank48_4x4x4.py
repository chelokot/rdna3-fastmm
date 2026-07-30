import triton
import triton.language as tl

DIMENSIONS = (4, 4, 4)
RANK = 48
CERTIFICATE_SHA256 = "c7103a1165af22d4e1a417607e9618b0bbae1b69e0c48ed7fa2a5aba26a3bcb7"


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
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_16 = (signal_15 - signal_7)
    signal_17 = (signal_15 + signal_7)
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_11 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_18 = (signal_3 + signal_11)
    signal_19 = (signal_3 - signal_11)
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_13 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_20 = (signal_5 - signal_13)
    signal_21 = (signal_5 + signal_13)
    signal_10 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_22 = (signal_10 - signal_2)
    signal_23 = (signal_10 + signal_2)
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_24 = (signal_1 + signal_2)
    signal_9 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_25 = (signal_9 + signal_10)
    signal_14 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_26 = (signal_13 + signal_14)
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_27 = (signal_5 + signal_6)
    signal_12 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_28 = (signal_12 + signal_4)
    signal_29 = ((signal_12 - signal_4) - signal_20)
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_29, mask=element_mask
    )
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_30 = (signal_8 - signal_0)
    signal_31 = ((signal_8 + signal_0) - signal_23)
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_31, mask=element_mask
    )
    signal_32 = (signal_28 - signal_17)
    signal_33 = (signal_19 + signal_30)
    signal_34 = (signal_30 - signal_22)
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    signal_35 = (signal_18 + signal_23)
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_35, mask=element_mask
    )
    signal_36 = (signal_18 - signal_23)
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_36, mask=element_mask
    )
    signal_37 = (signal_35 - signal_25)
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_37, mask=element_mask
    )
    signal_38 = (signal_24 - signal_37)
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_38, mask=element_mask
    )
    signal_39 = (signal_21 + signal_28)
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_39, mask=element_mask
    )
    signal_40 = (signal_21 - signal_28)
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_40, mask=element_mask
    )
    signal_41 = (signal_20 + signal_16)
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_41, mask=element_mask
    )
    signal_42 = (signal_20 - signal_16)
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_42, mask=element_mask
    )
    signal_43 = (signal_19 + signal_22)
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_43, mask=element_mask
    )
    signal_44 = (signal_21 + signal_17)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_44, mask=element_mask
    )
    signal_45 = (signal_41 - signal_27)
    tl.store(
        output + 38 * plane_elements + element_offsets, signal_45, mask=element_mask
    )
    signal_46 = (signal_40 - signal_41)
    signal_47 = (signal_30 + signal_22)
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_47, mask=element_mask
    )
    signal_48 = (signal_47 - signal_25)
    tl.store(
        output + 37 * plane_elements + element_offsets, signal_48, mask=element_mask
    )
    signal_49 = (signal_34 - signal_39)
    signal_50 = (signal_43 - signal_44)
    signal_51 = (signal_49 - signal_50)
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_51, mask=element_mask
    )
    signal_52 = (signal_49 + signal_50)
    tl.store(
        output + 47 * plane_elements + element_offsets, signal_52, mask=element_mask
    )
    signal_53 = (signal_36 - signal_42)
    signal_54 = (signal_29 - signal_31)
    signal_55 = (signal_29 + signal_42)
    signal_56 = (signal_36 - signal_31)
    signal_57 = (signal_55 + signal_56)
    signal_58 = (signal_55 - signal_56)
    signal_59 = (signal_48 + signal_37)
    signal_60 = (signal_40 - signal_27)
    tl.store(
        output + 41 * plane_elements + element_offsets, signal_60, mask=element_mask
    )
    signal_61 = (signal_26 - signal_60)
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_61, mask=element_mask
    )
    signal_62 = (signal_45 + signal_60)
    signal_63 = (signal_46 + signal_59)
    tl.store(
        output + 36 * plane_elements + element_offsets, signal_63, mask=element_mask
    )
    signal_64 = (signal_46 - signal_59)
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_64, mask=element_mask
    )
    signal_65 = (signal_53 + signal_50)
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_65, mask=element_mask
    )
    signal_66 = (signal_53 - signal_50)
    tl.store(
        output + 43 * plane_elements + element_offsets, signal_66, mask=element_mask
    )
    signal_67 = (signal_47 - signal_35)
    signal_68 = (signal_67 + signal_62)
    tl.store(
        output + 45 * plane_elements + element_offsets, signal_68, mask=element_mask
    )
    signal_69 = (signal_67 - signal_62)
    tl.store(
        output + 42 * plane_elements + element_offsets, signal_69, mask=element_mask
    )
    signal_70 = (signal_49 - signal_54)
    tl.store(
        output + 39 * plane_elements + element_offsets, signal_70, mask=element_mask
    )
    signal_71 = (signal_49 + signal_54)
    tl.store(
        output + 33 * plane_elements + element_offsets, signal_71, mask=element_mask
    )
    signal_72 = (signal_54 - signal_53)
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_72, mask=element_mask
    )
    signal_73 = (signal_54 + signal_53)
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_73, mask=element_mask
    )
    signal_74 = (signal_26 + signal_45)
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_74, mask=element_mask
    )
    signal_75 = (signal_58 - signal_64)
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_75, mask=element_mask
    )
    signal_76 = (signal_58 - signal_69)
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_76, mask=element_mask
    )
    signal_77 = (signal_38 + signal_59)
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_77, mask=element_mask
    )
    signal_78 = (signal_68 + signal_57)
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_78, mask=element_mask
    )
    signal_79 = (signal_63 - signal_57)
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_79, mask=element_mask
    )
    signal_80 = (signal_33 - signal_32)
    signal_81 = (signal_33 + signal_32)
    signal_82 = (signal_80 - signal_68)
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_82, mask=element_mask
    )
    signal_83 = (signal_80 - signal_63)
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_83, mask=element_mask
    )
    signal_84 = (signal_81 + signal_64)
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_84, mask=element_mask
    )
    signal_85 = (signal_81 - signal_69)
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_85, mask=element_mask
    )
    signal_86 = (signal_27 * 2)
    signal_87 = (signal_26 * 2)
    signal_88 = (signal_25 * 2)
    signal_89 = (signal_24 * 2)
    signal_90 = (signal_86 + signal_71)
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_90, mask=element_mask
    )
    signal_91 = (signal_86 + signal_65)
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_91, mask=element_mask
    )
    signal_92 = (signal_89 + signal_65)
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_92, mask=element_mask
    )
    signal_93 = (signal_89 - signal_71)
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_93, mask=element_mask
    )
    signal_94 = (signal_70 + signal_87)
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_94, mask=element_mask
    )
    signal_95 = (signal_66 - signal_87)
    tl.store(
        output + 30 * plane_elements + element_offsets, signal_95, mask=element_mask
    )
    signal_96 = (signal_66 + signal_88)
    tl.store(
        output + 29 * plane_elements + element_offsets, signal_96, mask=element_mask
    )
    signal_97 = (signal_70 + signal_88)
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_97, mask=element_mask
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
    signal_5 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_7 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_16 = (signal_5 - signal_7)
    tl.store(
        output + 19 * plane_elements + element_offsets, signal_16, mask=element_mask
    )
    signal_4 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_17 = (signal_7 - signal_4)
    tl.store(
        output + 12 * plane_elements + element_offsets, signal_17, mask=element_mask
    )
    signal_18 = (signal_5 + signal_7)
    tl.store(
        output + 0 * plane_elements + element_offsets, signal_18, mask=element_mask
    )
    signal_6 = tl.load(
        source + (1 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (1 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_19 = (signal_5 + signal_6)
    tl.store(
        output + 26 * plane_elements + element_offsets, signal_19, mask=element_mask
    )
    signal_15 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_12 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_20 = (signal_15 - signal_12)
    tl.store(
        output + 24 * plane_elements + element_offsets, signal_20, mask=element_mask
    )
    signal_21 = (signal_15 + signal_12)
    tl.store(
        output + 25 * plane_elements + element_offsets, signal_21, mask=element_mask
    )
    signal_1 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_2 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_22 = (signal_1 - signal_2)
    tl.store(
        output + 2 * plane_elements + element_offsets, signal_22, mask=element_mask
    )
    signal_23 = (signal_1 + signal_2)
    tl.store(
        output + 9 * plane_elements + element_offsets, signal_23, mask=element_mask
    )
    signal_3 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_24 = (signal_1 + signal_3)
    tl.store(
        output + 14 * plane_elements + element_offsets, signal_24, mask=element_mask
    )
    signal_8 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_10 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_25 = (signal_8 + signal_10)
    tl.store(
        output + 3 * plane_elements + element_offsets, signal_25, mask=element_mask
    )
    signal_26 = (signal_8 - signal_10)
    tl.store(
        output + 30 * plane_elements + element_offsets, signal_26, mask=element_mask
    )
    signal_9 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_27 = (signal_10 - signal_9)
    tl.store(
        output + 6 * plane_elements + element_offsets, signal_27, mask=element_mask
    )
    signal_11 = tl.load(
        source + (2 * block_height + local_rows) * source_columns + 3 * block_width + local_columns, mask=element_mask & (2 * block_height + local_rows < source_row_count) & (3 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_28 = (signal_8 + signal_11)
    tl.store(
        output + 8 * plane_elements + element_offsets, signal_28, mask=element_mask
    )
    signal_0 = tl.load(
        source + (0 * block_height + local_rows) * source_columns + 0 * block_width + local_columns, mask=element_mask & (0 * block_height + local_rows < source_row_count) & (0 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_29 = (signal_0 - signal_2)
    tl.store(
        output + 10 * plane_elements + element_offsets, signal_29, mask=element_mask
    )
    signal_14 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 2 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (2 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_30 = (signal_12 + signal_14)
    tl.store(
        output + 18 * plane_elements + element_offsets, signal_30, mask=element_mask
    )
    signal_13 = tl.load(
        source + (3 * block_height + local_rows) * source_columns + 1 * block_width + local_columns, mask=element_mask & (3 * block_height + local_rows < source_row_count) & (1 * block_width + local_columns < source_columns), other=0.0
    ).to(tl.float32)
    signal_31 = (signal_15 - signal_13)
    tl.store(
        output + 21 * plane_elements + element_offsets, signal_31, mask=element_mask
    )
    signal_32 = ((((signal_6 - signal_2) - signal_10) - signal_14) / 2)
    signal_33 = (signal_19 - signal_32)
    tl.store(
        output + 34 * plane_elements + element_offsets, signal_33, mask=element_mask
    )
    signal_34 = (signal_30 + signal_32)
    tl.store(
        output + 36 * plane_elements + element_offsets, signal_34, mask=element_mask
    )
    signal_35 = ((((signal_3 + signal_11) + signal_15) - signal_7) / 2)
    signal_36 = (signal_35 - signal_28)
    tl.store(
        output + 41 * plane_elements + element_offsets, signal_36, mask=element_mask
    )
    signal_37 = (signal_24 - signal_35)
    tl.store(
        output + 42 * plane_elements + element_offsets, signal_37, mask=element_mask
    )
    signal_38 = ((((signal_0 + signal_8) + signal_12) - signal_4) / 2)
    signal_39 = (signal_38 - signal_17)
    tl.store(
        output + 37 * plane_elements + element_offsets, signal_39, mask=element_mask
    )
    signal_40 = (signal_35 - signal_39)
    tl.store(
        output + 4 * plane_elements + element_offsets, signal_40, mask=element_mask
    )
    signal_41 = (signal_29 - signal_38)
    tl.store(
        output + 32 * plane_elements + element_offsets, signal_41, mask=element_mask
    )
    signal_42 = (signal_37 - signal_41)
    signal_43 = (signal_42 - signal_23)
    tl.store(
        output + 15 * plane_elements + element_offsets, signal_43, mask=element_mask
    )
    signal_44 = (signal_41 + signal_37)
    signal_45 = (signal_22 - signal_44)
    tl.store(
        output + 5 * plane_elements + element_offsets, signal_45, mask=element_mask
    )
    signal_46 = (signal_32 - signal_41)
    tl.store(
        output + 20 * plane_elements + element_offsets, signal_46, mask=element_mask
    )
    signal_47 = (signal_36 + signal_38)
    tl.store(
        output + 13 * plane_elements + element_offsets, signal_47, mask=element_mask
    )
    signal_48 = (signal_34 - signal_38)
    tl.store(
        output + 31 * plane_elements + element_offsets, signal_48, mask=element_mask
    )
    signal_49 = ((((signal_5 - signal_1) - signal_13) - signal_9) / 2)
    signal_50 = (signal_49 - signal_27)
    tl.store(
        output + 38 * plane_elements + element_offsets, signal_50, mask=element_mask
    )
    signal_51 = (signal_32 - signal_50)
    tl.store(
        output + 23 * plane_elements + element_offsets, signal_51, mask=element_mask
    )
    signal_52 = (signal_36 + signal_50)
    signal_53 = (signal_52 + signal_25)
    tl.store(
        output + 16 * plane_elements + element_offsets, signal_53, mask=element_mask
    )
    signal_54 = (signal_31 - signal_49)
    tl.store(
        output + 45 * plane_elements + element_offsets, signal_54, mask=element_mask
    )
    signal_55 = (signal_54 - signal_35)
    tl.store(
        output + 1 * plane_elements + element_offsets, signal_55, mask=element_mask
    )
    signal_56 = (signal_49 - signal_33)
    tl.store(
        output + 11 * plane_elements + element_offsets, signal_56, mask=element_mask
    )
    signal_57 = (signal_37 + signal_49)
    tl.store(
        output + 27 * plane_elements + element_offsets, signal_57, mask=element_mask
    )
    signal_58 = (signal_54 - signal_34)
    signal_59 = (signal_54 + signal_34)
    signal_60 = (signal_33 - signal_39)
    signal_61 = (signal_33 + signal_39)
    signal_62 = (signal_61 - signal_16)
    tl.store(
        output + 22 * plane_elements + element_offsets, signal_62, mask=element_mask
    )
    signal_63 = (signal_59 - signal_21)
    tl.store(
        output + 28 * plane_elements + element_offsets, signal_63, mask=element_mask
    )
    signal_64 = (signal_18 - signal_60)
    tl.store(
        output + 29 * plane_elements + element_offsets, signal_64, mask=element_mask
    )
    signal_65 = (signal_20 - signal_58)
    tl.store(
        output + 7 * plane_elements + element_offsets, signal_65, mask=element_mask
    )
    signal_66 = (signal_36 - signal_50)
    signal_67 = (signal_66 + signal_26)
    tl.store(
        output + 17 * plane_elements + element_offsets, signal_67, mask=element_mask
    )
    signal_68 = ((signal_66 + signal_60) / 2)
    tl.store(
        output + 43 * plane_elements + element_offsets, signal_68, mask=element_mask
    )
    signal_69 = (signal_60 - signal_68)
    tl.store(
        output + 40 * plane_elements + element_offsets, signal_69, mask=element_mask
    )
    signal_70 = ((signal_52 + signal_61) / 2)
    tl.store(
        output + 33 * plane_elements + element_offsets, signal_70, mask=element_mask
    )
    signal_71 = (signal_70 - signal_61)
    tl.store(
        output + 39 * plane_elements + element_offsets, signal_71, mask=element_mask
    )
    signal_72 = ((signal_44 + signal_59) / 2)
    tl.store(
        output + 35 * plane_elements + element_offsets, signal_72, mask=element_mask
    )
    signal_73 = (signal_72 - signal_59)
    tl.store(
        output + 47 * plane_elements + element_offsets, signal_73, mask=element_mask
    )
    signal_74 = ((signal_58 + signal_42) / 2)
    tl.store(
        output + 46 * plane_elements + element_offsets, signal_74, mask=element_mask
    )
    signal_75 = (signal_74 - signal_42)
    tl.store(
        output + 44 * plane_elements + element_offsets, signal_75, mask=element_mask
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
    signal_5 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_7 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_16 = (signal_5 - signal_7)
    tl.store(
        output + 19 * plane_elements + output_offsets,
        signal_16,
        mask=output_mask,
    )
    signal_4 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_17 = (signal_7 - signal_4)
    tl.store(
        output + 12 * plane_elements + output_offsets,
        signal_17,
        mask=output_mask,
    )
    signal_18 = (signal_5 + signal_7)
    tl.store(
        output + 0 * plane_elements + output_offsets,
        signal_18,
        mask=output_mask,
    )
    signal_6 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 1 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (1 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_19 = (signal_5 + signal_6)
    tl.store(
        output + 26 * plane_elements + output_offsets,
        signal_19,
        mask=output_mask,
    )
    signal_15 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_12 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_20 = (signal_15 - signal_12)
    tl.store(
        output + 24 * plane_elements + output_offsets,
        signal_20,
        mask=output_mask,
    )
    signal_21 = (signal_15 + signal_12)
    tl.store(
        output + 25 * plane_elements + output_offsets,
        signal_21,
        mask=output_mask,
    )
    signal_1 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_2 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_22 = (signal_1 - signal_2)
    tl.store(
        output + 2 * plane_elements + output_offsets,
        signal_22,
        mask=output_mask,
    )
    signal_23 = (signal_1 + signal_2)
    tl.store(
        output + 9 * plane_elements + output_offsets,
        signal_23,
        mask=output_mask,
    )
    signal_3 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_24 = (signal_1 + signal_3)
    tl.store(
        output + 14 * plane_elements + output_offsets,
        signal_24,
        mask=output_mask,
    )
    signal_8 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_10 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_25 = (signal_8 + signal_10)
    tl.store(
        output + 3 * plane_elements + output_offsets,
        signal_25,
        mask=output_mask,
    )
    signal_26 = (signal_8 - signal_10)
    tl.store(
        output + 30 * plane_elements + output_offsets,
        signal_26,
        mask=output_mask,
    )
    signal_9 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_27 = (signal_10 - signal_9)
    tl.store(
        output + 6 * plane_elements + output_offsets,
        signal_27,
        mask=output_mask,
    )
    signal_11 = tl.trans(
        tl.load(
            source + (3 * block_width + local_columns[:, None]) * source_columns + 2 * block_height + local_rows[None, :], mask=(3 * block_width + local_columns[:, None] < source_row_count) & (2 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_28 = (signal_8 + signal_11)
    tl.store(
        output + 8 * plane_elements + output_offsets,
        signal_28,
        mask=output_mask,
    )
    signal_0 = tl.trans(
        tl.load(
            source + (0 * block_width + local_columns[:, None]) * source_columns + 0 * block_height + local_rows[None, :], mask=(0 * block_width + local_columns[:, None] < source_row_count) & (0 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_29 = (signal_0 - signal_2)
    tl.store(
        output + 10 * plane_elements + output_offsets,
        signal_29,
        mask=output_mask,
    )
    signal_14 = tl.trans(
        tl.load(
            source + (2 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(2 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_30 = (signal_12 + signal_14)
    tl.store(
        output + 18 * plane_elements + output_offsets,
        signal_30,
        mask=output_mask,
    )
    signal_13 = tl.trans(
        tl.load(
            source + (1 * block_width + local_columns[:, None]) * source_columns + 3 * block_height + local_rows[None, :], mask=(1 * block_width + local_columns[:, None] < source_row_count) & (3 * block_height + local_rows[None, :] < source_columns), other=0.0
        ).to(tl.float32)
    )
    signal_31 = (signal_15 - signal_13)
    tl.store(
        output + 21 * plane_elements + output_offsets,
        signal_31,
        mask=output_mask,
    )
    signal_32 = ((((signal_6 - signal_2) - signal_10) - signal_14) / 2)
    signal_33 = (signal_19 - signal_32)
    tl.store(
        output + 34 * plane_elements + output_offsets,
        signal_33,
        mask=output_mask,
    )
    signal_34 = (signal_30 + signal_32)
    tl.store(
        output + 36 * plane_elements + output_offsets,
        signal_34,
        mask=output_mask,
    )
    signal_35 = ((((signal_3 + signal_11) + signal_15) - signal_7) / 2)
    signal_36 = (signal_35 - signal_28)
    tl.store(
        output + 41 * plane_elements + output_offsets,
        signal_36,
        mask=output_mask,
    )
    signal_37 = (signal_24 - signal_35)
    tl.store(
        output + 42 * plane_elements + output_offsets,
        signal_37,
        mask=output_mask,
    )
    signal_38 = ((((signal_0 + signal_8) + signal_12) - signal_4) / 2)
    signal_39 = (signal_38 - signal_17)
    tl.store(
        output + 37 * plane_elements + output_offsets,
        signal_39,
        mask=output_mask,
    )
    signal_40 = (signal_35 - signal_39)
    tl.store(
        output + 4 * plane_elements + output_offsets,
        signal_40,
        mask=output_mask,
    )
    signal_41 = (signal_29 - signal_38)
    tl.store(
        output + 32 * plane_elements + output_offsets,
        signal_41,
        mask=output_mask,
    )
    signal_42 = (signal_37 - signal_41)
    signal_43 = (signal_42 - signal_23)
    tl.store(
        output + 15 * plane_elements + output_offsets,
        signal_43,
        mask=output_mask,
    )
    signal_44 = (signal_41 + signal_37)
    signal_45 = (signal_22 - signal_44)
    tl.store(
        output + 5 * plane_elements + output_offsets,
        signal_45,
        mask=output_mask,
    )
    signal_46 = (signal_32 - signal_41)
    tl.store(
        output + 20 * plane_elements + output_offsets,
        signal_46,
        mask=output_mask,
    )
    signal_47 = (signal_36 + signal_38)
    tl.store(
        output + 13 * plane_elements + output_offsets,
        signal_47,
        mask=output_mask,
    )
    signal_48 = (signal_34 - signal_38)
    tl.store(
        output + 31 * plane_elements + output_offsets,
        signal_48,
        mask=output_mask,
    )
    signal_49 = ((((signal_5 - signal_1) - signal_13) - signal_9) / 2)
    signal_50 = (signal_49 - signal_27)
    tl.store(
        output + 38 * plane_elements + output_offsets,
        signal_50,
        mask=output_mask,
    )
    signal_51 = (signal_32 - signal_50)
    tl.store(
        output + 23 * plane_elements + output_offsets,
        signal_51,
        mask=output_mask,
    )
    signal_52 = (signal_36 + signal_50)
    signal_53 = (signal_52 + signal_25)
    tl.store(
        output + 16 * plane_elements + output_offsets,
        signal_53,
        mask=output_mask,
    )
    signal_54 = (signal_31 - signal_49)
    tl.store(
        output + 45 * plane_elements + output_offsets,
        signal_54,
        mask=output_mask,
    )
    signal_55 = (signal_54 - signal_35)
    tl.store(
        output + 1 * plane_elements + output_offsets,
        signal_55,
        mask=output_mask,
    )
    signal_56 = (signal_49 - signal_33)
    tl.store(
        output + 11 * plane_elements + output_offsets,
        signal_56,
        mask=output_mask,
    )
    signal_57 = (signal_37 + signal_49)
    tl.store(
        output + 27 * plane_elements + output_offsets,
        signal_57,
        mask=output_mask,
    )
    signal_58 = (signal_54 - signal_34)
    signal_59 = (signal_54 + signal_34)
    signal_60 = (signal_33 - signal_39)
    signal_61 = (signal_33 + signal_39)
    signal_62 = (signal_61 - signal_16)
    tl.store(
        output + 22 * plane_elements + output_offsets,
        signal_62,
        mask=output_mask,
    )
    signal_63 = (signal_59 - signal_21)
    tl.store(
        output + 28 * plane_elements + output_offsets,
        signal_63,
        mask=output_mask,
    )
    signal_64 = (signal_18 - signal_60)
    tl.store(
        output + 29 * plane_elements + output_offsets,
        signal_64,
        mask=output_mask,
    )
    signal_65 = (signal_20 - signal_58)
    tl.store(
        output + 7 * plane_elements + output_offsets,
        signal_65,
        mask=output_mask,
    )
    signal_66 = (signal_36 - signal_50)
    signal_67 = (signal_66 + signal_26)
    tl.store(
        output + 17 * plane_elements + output_offsets,
        signal_67,
        mask=output_mask,
    )
    signal_68 = ((signal_66 + signal_60) / 2)
    tl.store(
        output + 43 * plane_elements + output_offsets,
        signal_68,
        mask=output_mask,
    )
    signal_69 = (signal_60 - signal_68)
    tl.store(
        output + 40 * plane_elements + output_offsets,
        signal_69,
        mask=output_mask,
    )
    signal_70 = ((signal_52 + signal_61) / 2)
    tl.store(
        output + 33 * plane_elements + output_offsets,
        signal_70,
        mask=output_mask,
    )
    signal_71 = (signal_70 - signal_61)
    tl.store(
        output + 39 * plane_elements + output_offsets,
        signal_71,
        mask=output_mask,
    )
    signal_72 = ((signal_44 + signal_59) / 2)
    tl.store(
        output + 35 * plane_elements + output_offsets,
        signal_72,
        mask=output_mask,
    )
    signal_73 = (signal_72 - signal_59)
    tl.store(
        output + 47 * plane_elements + output_offsets,
        signal_73,
        mask=output_mask,
    )
    signal_74 = ((signal_58 + signal_42) / 2)
    tl.store(
        output + 46 * plane_elements + output_offsets,
        signal_74,
        mask=output_mask,
    )
    signal_75 = (signal_74 - signal_42)
    tl.store(
        output + 44 * plane_elements + output_offsets,
        signal_75,
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
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_48 = (signal_15 - signal_24)
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_49 = ((signal_44 + signal_15) + signal_24)
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_50 = (signal_5 - signal_25)
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_51 = ((signal_47 - signal_25) - signal_5)
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = (signal_9 + signal_7)
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_53 = ((signal_46 + signal_9) - signal_7)
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_54 = (signal_28 + signal_2)
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_55 = ((signal_35 + signal_28) - signal_2)
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = (signal_33 - signal_16)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = (signal_33 + signal_19)
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_58 = (signal_32 + signal_10)
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = (signal_32 + signal_20)
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_60 = (signal_36 + signal_18)
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_61 = (signal_36 - signal_31)
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_62 = (signal_39 + signal_3)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_63 = (signal_39 + signal_22)
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_64 = (signal_40 - signal_0)
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_65 = (signal_40 + signal_17)
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_66 = (signal_42 + signal_14)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_67 = (signal_42 + signal_27)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = (signal_43 + signal_29)
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_69 = (signal_43 + signal_30)
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_70 = (signal_45 - signal_1)
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_71 = (signal_45 + signal_21)
    signal_72 = (signal_56 - signal_65)
    signal_73 = (signal_56 + signal_65)
    signal_74 = (signal_50 + signal_52)
    signal_75 = (signal_50 - signal_52)
    signal_76 = (signal_48 + signal_54)
    signal_77 = (signal_48 - signal_54)
    signal_78 = (signal_55 + signal_49)
    signal_79 = (signal_55 - signal_49)
    signal_80 = (signal_53 + signal_51)
    signal_81 = (signal_53 - signal_51)
    signal_82 = (signal_61 + signal_58)
    signal_83 = (signal_61 - signal_58)
    signal_84 = (signal_57 + signal_64)
    signal_85 = (signal_57 - signal_64)
    signal_86 = (signal_68 + signal_63)
    signal_87 = (signal_68 - signal_63)
    signal_88 = (signal_67 + signal_71)
    signal_89 = (signal_67 - signal_71)
    signal_90 = (signal_62 + signal_69)
    signal_91 = (signal_62 - signal_69)
    signal_92 = (signal_79 + signal_75)
    signal_93 = (signal_79 - signal_75)
    signal_94 = (signal_66 + signal_70)
    signal_95 = (signal_66 - signal_70)
    signal_96 = (signal_76 + signal_80)
    signal_97 = (signal_76 - signal_80)
    signal_98 = (signal_59 + signal_60)
    signal_99 = (signal_59 - signal_60)
    signal_100 = (signal_74 + signal_78)
    signal_101 = (signal_74 - signal_78)
    signal_102 = (signal_81 - signal_77)
    signal_103 = (signal_81 + signal_77)
    signal_104 = (((signal_85 - signal_88) - signal_92) / 4)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_104, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_105 = (((signal_95 - signal_84) - signal_96) / 4)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_105, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_106 = (((signal_89 - signal_96) - signal_73) / 4)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_106, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_107 = (((signal_72 - signal_94) - signal_92) / 4)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_107, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_108 = (((signal_82 - signal_97) + signal_86) / 4)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_108, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_109 = (((signal_93 + signal_99) - signal_87) / 4)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_109, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_110 = (((signal_91 + signal_93) - signal_83) / 4)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_110, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_111 = (((signal_90 + signal_98) - signal_97) / 4)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_111, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_112 = (((((signal_100 - signal_99) - signal_85) / 4) + signal_26) + signal_34)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_112, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_113 = (((((signal_88 + signal_87) + signal_101) / 4) - signal_11) - signal_34)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_113, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_114 = (((((signal_94 - signal_91) - signal_101) / 4) + signal_8) - signal_41)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_114, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_115 = (((((signal_83 - signal_72) - signal_100) / 4) - signal_13) + signal_41)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_115, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_116 = (((((signal_95 + signal_86) - signal_102) / 4) + signal_4) + signal_37)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_116, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_117 = (((((signal_82 + signal_103) - signal_84) / 4) - signal_12) - signal_37)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_117, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_118 = (((((signal_98 - signal_103) - signal_73) / 4) + signal_23) + signal_38)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_118, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_119 = (((((signal_90 + signal_89) + signal_102) / 4) - signal_6) - signal_38)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_119, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
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
    signal_15 = tl.load(
        source + 15 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_24 = tl.load(
        source + 24 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_48 = (signal_15 - signal_24)
    signal_44 = tl.load(
        source + 44 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_49 = ((signal_44 + signal_15) + signal_24)
    signal_5 = tl.load(
        source + 5 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_25 = tl.load(
        source + 25 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_50 = (signal_5 - signal_25)
    signal_47 = tl.load(
        source + 47 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_51 = ((signal_47 - signal_25) - signal_5)
    signal_9 = tl.load(
        source + 9 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_7 = tl.load(
        source + 7 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_52 = (signal_9 + signal_7)
    signal_46 = tl.load(
        source + 46 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_53 = ((signal_46 + signal_9) - signal_7)
    signal_28 = tl.load(
        source + 28 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_2 = tl.load(
        source + 2 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_54 = (signal_28 + signal_2)
    signal_35 = tl.load(
        source + 35 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_55 = ((signal_35 + signal_28) - signal_2)
    signal_33 = tl.load(
        source + 33 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_16 = tl.load(
        source + 16 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_56 = (signal_33 - signal_16)
    signal_19 = tl.load(
        source + 19 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_57 = (signal_33 + signal_19)
    signal_32 = tl.load(
        source + 32 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_10 = tl.load(
        source + 10 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_58 = (signal_32 + signal_10)
    signal_20 = tl.load(
        source + 20 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_59 = (signal_32 + signal_20)
    signal_36 = tl.load(
        source + 36 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_18 = tl.load(
        source + 18 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_60 = (signal_36 + signal_18)
    signal_31 = tl.load(
        source + 31 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_61 = (signal_36 - signal_31)
    signal_39 = tl.load(
        source + 39 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_3 = tl.load(
        source + 3 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_62 = (signal_39 + signal_3)
    signal_22 = tl.load(
        source + 22 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_63 = (signal_39 + signal_22)
    signal_40 = tl.load(
        source + 40 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_0 = tl.load(
        source + 0 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_64 = (signal_40 - signal_0)
    signal_17 = tl.load(
        source + 17 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_65 = (signal_40 + signal_17)
    signal_42 = tl.load(
        source + 42 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_14 = tl.load(
        source + 14 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_66 = (signal_42 + signal_14)
    signal_27 = tl.load(
        source + 27 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_67 = (signal_42 + signal_27)
    signal_43 = tl.load(
        source + 43 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_29 = tl.load(
        source + 29 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_68 = (signal_43 + signal_29)
    signal_30 = tl.load(
        source + 30 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_69 = (signal_43 + signal_30)
    signal_45 = tl.load(
        source + 45 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_1 = tl.load(
        source + 1 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_70 = (signal_45 - signal_1)
    signal_21 = tl.load(
        source + 21 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_71 = (signal_45 + signal_21)
    signal_72 = (signal_56 - signal_65)
    signal_73 = (signal_56 + signal_65)
    signal_74 = (signal_50 + signal_52)
    signal_75 = (signal_50 - signal_52)
    signal_76 = (signal_48 + signal_54)
    signal_77 = (signal_48 - signal_54)
    signal_78 = (signal_55 + signal_49)
    signal_79 = (signal_55 - signal_49)
    signal_80 = (signal_53 + signal_51)
    signal_81 = (signal_53 - signal_51)
    signal_82 = (signal_61 + signal_58)
    signal_83 = (signal_61 - signal_58)
    signal_84 = (signal_57 + signal_64)
    signal_85 = (signal_57 - signal_64)
    signal_86 = (signal_68 + signal_63)
    signal_87 = (signal_68 - signal_63)
    signal_88 = (signal_67 + signal_71)
    signal_89 = (signal_67 - signal_71)
    signal_90 = (signal_62 + signal_69)
    signal_91 = (signal_62 - signal_69)
    signal_92 = (signal_79 + signal_75)
    signal_93 = (signal_79 - signal_75)
    signal_94 = (signal_66 + signal_70)
    signal_95 = (signal_66 - signal_70)
    signal_96 = (signal_76 + signal_80)
    signal_97 = (signal_76 - signal_80)
    signal_98 = (signal_59 + signal_60)
    signal_99 = (signal_59 - signal_60)
    signal_100 = (signal_74 + signal_78)
    signal_101 = (signal_74 - signal_78)
    signal_102 = (signal_81 - signal_77)
    signal_103 = (signal_81 + signal_77)
    signal_104 = (((signal_85 - signal_88) - signal_92) / 4)
    bias_4 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_104 + bias_4, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_105 = (((signal_95 - signal_84) - signal_96) / 4)
    bias_12 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_105 + bias_12, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_106 = (((signal_89 - signal_96) - signal_73) / 4)
    bias_5 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_106 + bias_5, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_107 = (((signal_72 - signal_94) - signal_92) / 4)
    bias_13 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_107 + bias_13, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_108 = (((signal_82 - signal_97) + signal_86) / 4)
    bias_2 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_108 + bias_2, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_109 = (((signal_93 + signal_99) - signal_87) / 4)
    bias_10 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_109 + bias_10, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_110 = (((signal_91 + signal_93) - signal_83) / 4)
    bias_3 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_110 + bias_3, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_111 = (((signal_90 + signal_98) - signal_97) / 4)
    bias_11 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_111 + bias_11, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_26 = tl.load(
        source + 26 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_34 = tl.load(
        source + 34 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_112 = (((((signal_100 - signal_99) - signal_85) / 4) + signal_26) + signal_34)
    bias_8 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_112 + bias_8, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_11 = tl.load(
        source + 11 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_113 = (((((signal_88 + signal_87) + signal_101) / 4) - signal_11) - signal_34)
    bias_6 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_113 + bias_6, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
    signal_8 = tl.load(
        source + 8 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_41 = tl.load(
        source + 41 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_114 = (((((signal_94 - signal_91) - signal_101) / 4) + signal_8) - signal_41)
    bias_15 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_114 + bias_15, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_13 = tl.load(
        source + 13 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_115 = (((((signal_83 - signal_72) - signal_100) / 4) - signal_13) + signal_41)
    bias_1 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_115 + bias_1, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_4 = tl.load(
        source + 4 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_37 = tl.load(
        source + 37 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_116 = (((((signal_95 + signal_86) - signal_102) / 4) + signal_4) + signal_37)
    bias_14 = tl.load(
        bias
        + 3 * block_width
        + local_columns,
        mask=element_mask & (3 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (2 * block_height + local_rows) * output_columns + 3 * block_width + local_columns, signal_116 + bias_14, mask=element_mask & (2 * block_height + local_rows < output_row_count) & (3 * block_width + local_columns < output_columns)
    )
    signal_12 = tl.load(
        source + 12 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_117 = (((((signal_82 + signal_103) - signal_84) / 4) - signal_12) - signal_37)
    bias_0 = tl.load(
        bias
        + 0 * block_width
        + local_columns,
        mask=element_mask & (0 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (0 * block_height + local_rows) * output_columns + 0 * block_width + local_columns, signal_117 + bias_0, mask=element_mask & (0 * block_height + local_rows < output_row_count) & (0 * block_width + local_columns < output_columns)
    )
    signal_23 = tl.load(
        source + 23 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_38 = tl.load(
        source + 38 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_118 = (((((signal_98 - signal_103) - signal_73) / 4) + signal_23) + signal_38)
    bias_9 = tl.load(
        bias
        + 2 * block_width
        + local_columns,
        mask=element_mask & (2 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (1 * block_height + local_rows) * output_columns + 2 * block_width + local_columns, signal_118 + bias_9, mask=element_mask & (1 * block_height + local_rows < output_row_count) & (2 * block_width + local_columns < output_columns)
    )
    signal_6 = tl.load(
        source + 6 * plane_elements + element_offsets, mask=element_mask, other=0.0
    ).to(tl.float32)
    signal_119 = (((((signal_90 + signal_89) + signal_102) / 4) - signal_6) - signal_38)
    bias_7 = tl.load(
        bias
        + 1 * block_width
        + local_columns,
        mask=element_mask & (1 * block_width + local_columns < output_columns),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + (3 * block_height + local_rows) * output_columns + 1 * block_width + local_columns, signal_119 + bias_7, mask=element_mask & (3 * block_height + local_rows < output_row_count) & (1 * block_width + local_columns < output_columns)
    )
