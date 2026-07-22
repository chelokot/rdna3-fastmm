import argparse
import hashlib
from pathlib import Path
from textwrap import indent
from typing import Literal, TypedDict

from rdna3_fastmm.certificate import (
    expand_linear_map,
    LinearTerm,
    load_reduced_scheme,
    verify_reduced_scheme,
)


ExpressionKind = Literal["gate", "output"]
SourceKind = Literal["input", "products"]
OutputMode = Literal["scalar", "mfma", "both"]


class ScheduledExpression(TypedDict):
    kind: ExpressionKind
    index: int
    terms: list[LinearTerm]


def schedule_expressions(
    input_count: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> list[ScheduledExpression]:
    available_count = input_count
    pending_outputs = set(range(len(outputs)))
    schedule: list[ScheduledExpression] = []

    def emit_ready_outputs() -> None:
        ready = [
            output_index
            for output_index in sorted(pending_outputs)
            if all(term["index"] < available_count for term in outputs[output_index])
        ]
        for output_index in ready:
            schedule.append(
                ScheduledExpression(
                    kind="output",
                    index=output_index,
                    terms=outputs[output_index],
                )
            )
            pending_outputs.remove(output_index)

    emit_ready_outputs()
    for gate_index, terms in enumerate(gates):
        schedule.append(ScheduledExpression(kind="gate", index=gate_index, terms=terms))
        available_count += 1
        emit_ready_outputs()
    if pending_outputs:
        raise ValueError("output expressions reference unavailable signals")
    return schedule


def format_expression(terms: list[LinearTerm]) -> str:
    pieces: list[str] = []
    for term_index, term in enumerate(terms):
        signal = f"signal_{term['index']}"
        if term_index == 0:
            pieces.append(signal if term["value"] == 1 else f"-{signal}")
        else:
            operator = "+" if term["value"] == 1 else "-"
            pieces.append(f" {operator} {signal}")
    return "".join(pieces)


def input_load_lines(
    signal_index: int,
    input_count: int,
    block_columns: int,
    source_kind: SourceKind,
) -> list[str]:
    if not 0 <= signal_index < input_count:
        return []
    if source_kind == "input":
        block_row, block_column = divmod(signal_index, block_columns)
        pointer = (
            "source + "
            f"({block_row} * block_height + local_rows) * source_columns + "
            f"{block_column} * block_width + local_columns"
        )
        mask = (
            "element_mask & "
            f"({block_row} * block_height + local_rows < source_row_count) & "
            f"({block_column} * block_width + local_columns < source_columns)"
        )
    else:
        pointer = f"source + {signal_index} * plane_elements + element_offsets"
        mask = "element_mask"
    return [
        f"    signal_{signal_index} = tl.load(",
        f"        {pointer}, mask={mask}, other=0.0",
        "    ).to(tl.float32)",
    ]


def generate_kernel(
    name: str,
    input_count: int,
    block_columns: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
    source_kind: SourceKind,
    output_block_rows: int,
    include_bias: bool = False,
) -> str:
    if include_bias and source_kind != "products":
        raise ValueError("bias is only valid for output reconstruction")
    schedule = schedule_expressions(input_count, gates, outputs)
    lines = [
        "@triton.jit",
        f"def {name}(",
        "    source,",
        "    output,",
    ]
    if include_bias:
        lines.append("    bias,")
    lines.extend(
        [
            "    source_row_count: tl.constexpr,",
            "    source_columns: tl.constexpr,",
            "    output_row_count: tl.constexpr,",
            "    output_columns: tl.constexpr,",
            "    block_height: tl.constexpr,",
            "    block_width: tl.constexpr,",
            "    block_elements: tl.constexpr,",
            "):",
            "    element_offsets = (",
            "        tl.program_id(0) * block_elements + tl.arange(0, block_elements)",
            "    )",
            "    plane_elements = block_height * block_width",
            "    element_mask = element_offsets < plane_elements",
        ]
    )
    lines.extend(
        [
            "    local_rows = element_offsets // block_width",
            "    local_columns = element_offsets % block_width",
        ]
    )
    loaded_inputs: set[int] = set()
    for expression in schedule:
        for term in expression["terms"]:
            signal_index = term["index"]
            if signal_index < input_count and signal_index not in loaded_inputs:
                lines.extend(
                    input_load_lines(
                        signal_index, input_count, block_columns, source_kind
                    )
                )
                loaded_inputs.add(signal_index)
        value = format_expression(expression["terms"])
        if expression["kind"] == "gate":
            signal_index = input_count + expression["index"]
            lines.append(f"    signal_{signal_index} = {value}")
            continue
        output_index = expression["index"]
        if source_kind == "input":
            pointer = f"output + {output_index} * plane_elements + element_offsets"
        else:
            block_row = output_index % output_block_rows
            block_column = output_index // output_block_rows
            pointer = (
                "output + "
                f"({block_row} * block_height + local_rows) * output_columns + "
                f"{block_column} * block_width + local_columns"
            )
            if include_bias:
                bias_mask = (
                    "element_mask & "
                    f"({block_column} * block_width + local_columns "
                    "< output_columns)"
                )
                lines.extend(
                    [
                        f"    bias_{output_index} = tl.load(",
                        "        bias",
                        f"        + {block_column} * block_width",
                        "        + local_columns,",
                        f"        mask={bias_mask},",
                        "        other=0.0,",
                        "    ).to(tl.float32)",
                    ]
                )
                value = f"{value} + bias_{output_index}"
        mask = "element_mask"
        if source_kind == "products":
            mask += (
                " & "
                f"({block_row} * block_height + local_rows < output_row_count) & "
                f"({block_column} * block_width + local_columns < output_columns)"
            )
        lines.extend(
            [
                "    tl.store(",
                f"        {pointer}, {value}, mask={mask}",
                "    )",
            ]
        )
    return "\n".join(lines)


def weight_input_load_lines(
    signal_index: int,
    input_count: int,
    block_columns: int,
) -> list[str]:
    if not 0 <= signal_index < input_count:
        return []
    block_row, block_column = divmod(signal_index, block_columns)
    pointer = (
        "source + "
        f"({block_column} * block_width + local_columns[:, None]) "
        "* source_columns + "
        f"{block_row} * block_height + local_rows[None, :]"
    )
    mask = (
        f"({block_column} * block_width + local_columns[:, None] "
        "< source_row_count) & "
        f"({block_row} * block_height + local_rows[None, :] < source_columns)"
    )
    return [
        f"    signal_{signal_index} = tl.trans(",
        "        tl.load(",
        f"            {pointer}, mask={mask}, other=0.0",
        "        ).to(tl.float32)",
        "    )",
    ]


def generate_weight_input_kernel(
    name: str,
    input_count: int,
    block_columns: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> str:
    schedule = schedule_expressions(input_count, gates, outputs)
    lines = [
        "@triton.jit",
        f"def {name}(",
        "    source,",
        "    output,",
        "    source_row_count: tl.constexpr,",
        "    source_columns: tl.constexpr,",
        "    block_height: tl.constexpr,",
        "    block_width: tl.constexpr,",
        "    block_rows_per_program: tl.constexpr,",
        "    block_columns_per_program: tl.constexpr,",
        "):",
        "    local_rows = (",
        "        tl.program_id(0) * block_rows_per_program",
        "        + tl.arange(0, block_rows_per_program)",
        "    )",
        "    local_columns = (",
        "        tl.program_id(1) * block_columns_per_program",
        "        + tl.arange(0, block_columns_per_program)",
        "    )",
        "    plane_elements = block_height * block_width",
        "    output_offsets = (",
        "        local_rows[:, None] * block_width + local_columns[None, :]",
        "    )",
        "    output_mask = (",
        "        (local_rows[:, None] < block_height)",
        "        & (local_columns[None, :] < block_width)",
        "    )",
    ]
    loaded_inputs: set[int] = set()
    for expression in schedule:
        for term in expression["terms"]:
            signal_index = term["index"]
            if signal_index < input_count and signal_index not in loaded_inputs:
                lines.extend(
                    weight_input_load_lines(signal_index, input_count, block_columns)
                )
                loaded_inputs.add(signal_index)
        value = format_expression(expression["terms"])
        if expression["kind"] == "gate":
            signal_index = input_count + expression["index"]
            lines.append(f"    signal_{signal_index} = {value}")
            continue
        output_index = expression["index"]
        lines.extend(
            [
                "    tl.store(",
                f"        output + {output_index} * plane_elements + output_offsets,",
                f"        {value},",
                "        mask=output_mask,",
                "    )",
            ]
        )
    return "\n".join(lines)


def transposed_input_load_lines(
    signal_index: int,
    input_count: int,
    block_columns: int,
) -> list[str]:
    if not 0 <= signal_index < input_count:
        return []
    block_row, block_column = divmod(signal_index, block_columns)
    pointer = (
        "source + "
        f"({block_row} * block_height + local_rows[:, None]) * source_columns + "
        f"{block_column} * block_width + local_columns[None, :]"
    )
    mask = (
        f"({block_row} * block_height + local_rows[:, None] < source_row_count) & "
        f"({block_column} * block_width + local_columns[None, :] < source_columns)"
    )
    return [
        f"    signal_{signal_index} = tl.load(",
        f"        {pointer}, mask={mask}, other=0.0",
        "    ).to(tl.float32)",
    ]


def generate_transposed_input_kernel(
    name: str,
    input_count: int,
    block_columns: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> str:
    schedule = schedule_expressions(input_count, gates, outputs)
    lines = [
        "@triton.jit",
        f"def {name}(",
        "    source,",
        "    output,",
        "    source_row_count: tl.constexpr,",
        "    source_columns: tl.constexpr,",
        "    output_row_count: tl.constexpr,",
        "    output_columns: tl.constexpr,",
        "    block_height: tl.constexpr,",
        "    block_width: tl.constexpr,",
        "    block_rows_per_program: tl.constexpr,",
        "    block_columns_per_program: tl.constexpr,",
        "):",
        "    local_rows = (",
        "        tl.program_id(0) * block_rows_per_program",
        "        + tl.arange(0, block_rows_per_program)",
        "    )",
        "    local_columns = (",
        "        tl.program_id(1) * block_columns_per_program",
        "        + tl.arange(0, block_columns_per_program)",
        "    )",
        "    plane_elements = block_height * block_width",
        "    output_offsets = (",
        "        local_columns[:, None] * block_height + local_rows[None, :]",
        "    )",
        "    output_mask = (",
        "        (local_columns[:, None] < block_width)",
        "        & (local_rows[None, :] < block_height)",
        "    )",
    ]
    loaded_inputs: set[int] = set()
    for expression in schedule:
        for term in expression["terms"]:
            signal_index = term["index"]
            if signal_index < input_count and signal_index not in loaded_inputs:
                lines.extend(
                    transposed_input_load_lines(
                        signal_index, input_count, block_columns
                    )
                )
                loaded_inputs.add(signal_index)
        value = format_expression(expression["terms"])
        if expression["kind"] == "gate":
            signal_index = input_count + expression["index"]
            lines.append(f"    signal_{signal_index} = {value}")
            continue
        output_index = expression["index"]
        lines.extend(
            [
                "    tl.store(",
                f"        output + {output_index} * plane_elements + output_offsets,",
                f"        tl.trans({value}),",
                "        mask=output_mask,",
                "    )",
            ]
        )
    return "\n".join(lines)


def generate_transposed_output_kernel(
    name: str,
    input_count: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
    output_block_rows: int,
) -> str:
    schedule = schedule_expressions(input_count, gates, outputs)
    lines = [
        "@triton.jit",
        f"def {name}(",
        "    source,",
        "    output,",
        "    output_row_count: tl.constexpr,",
        "    output_columns: tl.constexpr,",
        "    block_height: tl.constexpr,",
        "    block_width: tl.constexpr,",
        "    block_rows_per_program: tl.constexpr,",
        "    block_columns_per_program: tl.constexpr,",
        "):",
        "    local_rows = (",
        "        tl.program_id(0) * block_rows_per_program",
        "        + tl.arange(0, block_rows_per_program)",
        "    )",
        "    local_columns = (",
        "        tl.program_id(1) * block_columns_per_program",
        "        + tl.arange(0, block_columns_per_program)",
        "    )",
        "    plane_elements = block_height * block_width",
        "    source_offsets = (",
        "        local_columns[:, None] * block_height + local_rows[None, :]",
        "    )",
        "    source_mask = (",
        "        (local_columns[:, None] < block_width)",
        "        & (local_rows[None, :] < block_height)",
        "    )",
    ]
    loaded_inputs: set[int] = set()
    for expression in schedule:
        for term in expression["terms"]:
            signal_index = term["index"]
            if signal_index < input_count and signal_index not in loaded_inputs:
                lines.extend(
                    [
                        f"    signal_{signal_index} = tl.trans(",
                        "        tl.load(",
                        "            source",
                        f"            + {signal_index} * plane_elements",
                        "            + source_offsets,",
                        "            mask=source_mask,",
                        "            other=0.0,",
                        "        ).to(tl.float32)",
                        "    )",
                    ]
                )
                loaded_inputs.add(signal_index)
        value = format_expression(expression["terms"])
        if expression["kind"] == "gate":
            signal_index = input_count + expression["index"]
            lines.append(f"    signal_{signal_index} = {value}")
            continue
        output_index = expression["index"]
        block_row = output_index % output_block_rows
        block_column = output_index // output_block_rows
        pointer = (
            "output + "
            f"({block_row} * block_height + local_rows[:, None]) "
            "* output_columns + "
            f"{block_column} * block_width + local_columns[None, :]"
        )
        mask = (
            "(local_rows[:, None] < block_height) & "
            "(local_columns[None, :] < block_width) & "
            f"({block_row} * block_height + local_rows[:, None] "
            "< output_row_count) & "
            f"({block_column} * block_width + local_columns[None, :] "
            "< output_columns)"
        )
        lines.extend(
            [
                "    tl.store(",
                f"        {pointer},",
                f"        {value},",
                f"        mask={mask},",
                "    )",
            ]
        )
    return "\n".join(lines)


def rank_major_output_coefficients(
    rank: int,
    first_size: int,
    second_size: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> tuple[tuple[int, ...], ...]:
    output_columns = expand_linear_map(rank, gates, outputs)
    row_major_outputs = [
        output_columns[column * first_size + row]
        for row in range(first_size)
        for column in range(second_size)
    ]
    return tuple(
        tuple(output[term] for output in row_major_outputs) for term in range(rank)
    )


def format_coefficient_constant(
    name: str, coefficients: tuple[tuple[int, ...], ...]
) -> str:
    coefficient_lines = [f"{name} = ("]
    coefficient_lines.extend(f"    {row}," for row in coefficients)
    coefficient_lines.append(")")
    return "\n".join(coefficient_lines)


def generate_mfma_output(
    rank: int,
    first_size: int,
    second_size: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> tuple[str, str]:
    output_count = first_size * second_size
    if output_count & (output_count - 1) or output_count > 64:
        raise ValueError(
            "MFMA reconstruction requires a power-of-two output count ≤ 64"
        )
    coefficients = rank_major_output_coefficients(
        rank, first_size, second_size, gates, outputs
    )
    padded_rank = ((rank + 15) // 16) * 16
    kernel = f"""@triton.jit
def output_transform_mfma_kernel(
    source,
    coefficients,
    output,
    output_row_count: tl.constexpr,
    output_columns: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
    block_positions: tl.constexpr,
    rank_chunk: tl.constexpr,
):
    positions = (
        tl.program_id(0) * block_positions + tl.arange(0, block_positions)
    )
    output_offsets = tl.arange(0, {output_count})
    plane_elements = block_height * block_width
    position_mask = positions < plane_elements
    accumulator = tl.zeros((block_positions, {output_count}), tl.float32)
    for rank_start in range(0, {padded_rank}, rank_chunk):
        rank_offsets = rank_start + tl.arange(0, rank_chunk)
        product_values = tl.load(
            source
            + rank_offsets[None, :] * plane_elements
            + positions[:, None],
            mask=position_mask[:, None] & (rank_offsets[None, :] < {rank}),
            other=0.0,
        ).to(tl.float16)
        coefficient_values = tl.load(
            coefficients
            + rank_offsets[:, None] * {output_count}
            + output_offsets[None, :],
            mask=rank_offsets[:, None] < {rank},
            other=0.0,
        ).to(tl.float16)
        accumulator = tl.dot(
            product_values,
            coefficient_values,
            acc=accumulator,
            out_dtype=tl.float32,
        )
    local_rows = positions // block_width
    local_columns = positions % block_width
    output_block_rows = output_offsets // {second_size}
    output_block_columns = output_offsets % {second_size}
    output_rows = (
        output_block_rows[None, :] * block_height + local_rows[:, None]
    )
    output_column_indices = (
        output_block_columns[None, :] * block_width + local_columns[:, None]
    )
    output_mask = (
        position_mask[:, None]
        & (output_rows < output_row_count)
        & (output_column_indices < output_columns)
    )
    tl.store(
        output + output_rows * output_columns + output_column_indices,
        accumulator,
        mask=output_mask,
    )"""
    return format_coefficient_constant("MFMA_OUTPUT_COEFFICIENTS", coefficients), kernel


def validate_fused_output_count(first_size: int, second_size: int) -> int:
    output_count = first_size * second_size
    if output_count & (output_count - 1) or output_count > 16:
        raise ValueError(
            "fused product reconstruction requires a power-of-two output count ≤ 16"
        )
    return output_count


def transformed_product_source(rank_expression: str) -> str:
    return f"""product = tl.zeros((block_m, block_n), tl.float32)
for inner_start in tl.range(
    0, block_inner, block_k, num_stages=1, loop_unroll_factor=1
):
    inner_offsets = inner_start + tl.arange(0, block_k)
    left_values = tl.load(
        left_transformed
        + {rank_expression} * left_plane_elements
        + local_rows[:, None] * block_inner
        + inner_offsets[None, :],
        mask=(local_rows[:, None] < block_rows)
        & (inner_offsets[None, :] < block_inner),
        other=0.0,
    ).to(tl.float16)
    right_values = tl.load(
        right_transformed
        + {rank_expression} * right_plane_elements
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
    )"""


def output_mapping_source(second_size: int) -> str:
    return f"""output_block_rows = output_blocks // {second_size}
output_block_columns = output_blocks % {second_size}
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
)"""


def generate_serial_product_output(
    rank: int,
    first_size: int,
    second_size: int,
) -> str:
    output_count = validate_fused_output_count(first_size, second_size)
    product_source = indent(transformed_product_source("rank_index"), " " * 8)
    mapping_source = indent(output_mapping_source(second_size), " " * 4)
    kernel = f"""@triton.jit
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
    output_blocks = tl.arange(0, {output_count})
    left_plane_elements = block_rows * block_inner
    right_plane_elements = block_inner * block_columns
    accumulator = tl.zeros((block_m, {output_count}, block_n), tl.float32)
    for rank_index in tl.range(0, {rank}, loop_unroll_factor=1):
{product_source}
        coefficient_values = tl.load(coefficients + rank_index * {output_count} + output_blocks).to(
            tl.float32
        )
        accumulator += product[:, None, :] * coefficient_values[None, :, None]
{mapping_source}
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
    )"""
    return kernel


def generate_sparse_serial_product_output(
    rank: int,
    first_size: int,
    second_size: int,
    coefficients: tuple[tuple[int, ...], ...],
) -> str:
    output_count = validate_fused_output_count(first_size, second_size)
    if len(coefficients) != rank or any(
        len(row) != output_count for row in coefficients
    ):
        raise ValueError("fused output coefficient dimensions do not match the scheme")
    if any(value not in {-1, 0, 1} for row in coefficients for value in row):
        raise ValueError("sparse serial fusion requires coefficients in {-1, 0, 1}")
    lines = [
        "@triton.jit",
        "def sparse_fused_product_output_kernel(",
        "    left_transformed,",
        "    right_transformed,",
        "    output,",
        "    bias,",
        "    block_rows: tl.constexpr,",
        "    block_inner: tl.constexpr,",
        "    block_columns: tl.constexpr,",
        "    output_row_count: tl.constexpr,",
        "    output_columns: tl.constexpr,",
        "    block_m: tl.constexpr,",
        "    block_n: tl.constexpr,",
        "    block_k: tl.constexpr,",
        "    has_bias: tl.constexpr,",
        "):",
        "    local_rows = tl.program_id(0) * block_m + tl.arange(0, block_m)",
        "    local_columns = tl.program_id(1) * block_n + tl.arange(0, block_n)",
        "    left_plane_elements = block_rows * block_inner",
        "    right_plane_elements = block_inner * block_columns",
    ]
    lines.extend(
        f"    accumulator_{output_index} = tl.zeros((block_m, block_n), tl.float32)"
        for output_index in range(output_count)
    )
    for rank_index, coefficient_row in enumerate(coefficients):
        lines.extend(
            indent(transformed_product_source(str(rank_index)), " " * 4).splitlines()
        )
        for output_index, coefficient in enumerate(coefficient_row):
            if coefficient == 1:
                lines.append(f"    accumulator_{output_index} += product")
            elif coefficient == -1:
                lines.append(f"    accumulator_{output_index} -= product")
    lines.append("    if has_bias:")
    for block_column in range(second_size):
        lines.extend(
            [
                f"        bias_{block_column} = tl.load(",
                f"            bias + {block_column} * block_columns + local_columns,",
                "            mask=(local_columns < block_columns)",
                f"            & ({block_column} * block_columns + local_columns < output_columns),",
                "            other=0.0,",
                "        ).to(tl.float32)",
            ]
        )
    for output_index in range(output_count):
        block_column = output_index % second_size
        lines.append(f"        accumulator_{output_index} += bias_{block_column}")
    for output_index in range(output_count):
        block_row = output_index // second_size
        block_column = output_index % second_size
        lines.extend(
            [
                "    tl.store(",
                "        output",
                f"        + ({block_row} * block_rows + local_rows[:, None]) * output_columns",
                f"        + {block_column} * block_columns",
                "        + local_columns[None, :],",
                f"        accumulator_{output_index},",
                "        mask=(local_rows[:, None] < block_rows)",
                "        & (local_columns[None, :] < block_columns)",
                f"        & ({block_row} * block_rows + local_rows[:, None] < output_row_count)",
                f"        & ({block_column} * block_columns + local_columns[None, :] < output_columns),",
                "    )",
            ]
        )
    return "\n".join(lines)


def generate_atomic_product_output(
    rank: int, first_size: int, second_size: int
) -> tuple[str, str]:
    output_count = validate_fused_output_count(first_size, second_size)
    product_source = indent(transformed_product_source("rank_index"), " " * 4)
    product_kernel = f"""@triton.jit
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
{product_source}
    for output_block in tl.range(0, {output_count}, loop_unroll_factor=1):
        coefficient_value = tl.load(coefficients + rank_index * {output_count} + output_block).to(
            tl.float32
        )
        output_block_row = output_block // {second_size}
        output_block_column = output_block % {second_size}
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
        )"""
    finalizer_kernel = """@triton.jit
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
        tl.store(accumulator + offsets, 0.0, mask=output_mask)"""
    return product_kernel, finalizer_kernel


def module_header(
    source_path: Path, dimensions: tuple[int, int, int], rank: int
) -> str:
    certificate_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return "\n".join(
        [
            "import triton",
            "import triton.language as tl",
            "",
            f"DIMENSIONS = {dimensions}",
            f"RANK = {rank}",
            f'CERTIFICATE_SHA256 = "{certificate_sha256}"',
        ]
    )


def generate_research_output_fusion_module(source_path: Path) -> str:
    source = load_reduced_scheme(source_path)
    summary = verify_reduced_scheme(source)
    first_size, _, second_size = summary.dimensions
    validate_fused_output_count(first_size, second_size)
    coefficients = rank_major_output_coefficients(
        summary.rank,
        first_size,
        second_size,
        source["w_fresh"],
        source["w"],
    )
    serial_kernel = generate_serial_product_output(
        summary.rank, first_size, second_size
    )
    atomic_kernel, finalizer_kernel = generate_atomic_product_output(
        summary.rank, first_size, second_size
    )
    header = module_header(source_path, summary.dimensions, summary.rank)
    coefficient_constant = format_coefficient_constant(
        "FUSED_OUTPUT_COEFFICIENTS", coefficients
    )
    return (
        header
        + "\n\n"
        + coefficient_constant
        + "\n\n\n"
        + "\n\n\n".join((serial_kernel, atomic_kernel, finalizer_kernel))
        + "\n"
    )


def generate_sparse_research_output_fusion_module(source_path: Path) -> str:
    source = load_reduced_scheme(source_path)
    summary = verify_reduced_scheme(source)
    first_size, _, second_size = summary.dimensions
    coefficients = rank_major_output_coefficients(
        summary.rank,
        first_size,
        second_size,
        source["w_fresh"],
        source["w"],
    )
    kernel = generate_sparse_serial_product_output(
        summary.rank,
        first_size,
        second_size,
        coefficients,
    )
    return (
        module_header(source_path, summary.dimensions, summary.rank)
        + "\n\n\n"
        + kernel
        + "\n"
    )


def generate_module(
    source_path: Path,
    include_transposed: bool = False,
    output_mode: OutputMode = "scalar",
) -> str:
    source = load_reduced_scheme(source_path)
    summary = verify_reduced_scheme(source)
    first_size, shared_size, second_size = summary.dimensions
    rank = summary.rank
    header = module_header(source_path, summary.dimensions, rank)
    kernels = [
        generate_kernel(
            "left_transform_kernel",
            first_size * shared_size,
            shared_size,
            source["u_fresh"],
            source["u"],
            "input",
            first_size,
        ),
        generate_kernel(
            "right_transform_kernel",
            shared_size * second_size,
            second_size,
            source["v_fresh"],
            source["v"],
            "input",
            shared_size,
        ),
    ]
    if output_mode in ("scalar", "both"):
        kernels.extend(
            [
                generate_weight_input_kernel(
                    "right_transform_weight_kernel",
                    shared_size * second_size,
                    second_size,
                    source["v_fresh"],
                    source["v"],
                ),
                generate_kernel(
                    "output_transform_kernel",
                    rank,
                    second_size,
                    source["w_fresh"],
                    source["w"],
                    "products",
                    first_size,
                ),
                generate_kernel(
                    "output_transform_bias_kernel",
                    rank,
                    second_size,
                    source["w_fresh"],
                    source["w"],
                    "products",
                    first_size,
                    include_bias=True,
                ),
            ]
        )
    if output_mode in ("mfma", "both"):
        coefficients, mfma_kernel = generate_mfma_output(
            rank,
            first_size,
            second_size,
            source["w_fresh"],
            source["w"],
        )
        header += "\n\n" + coefficients
        kernels.append(mfma_kernel)
    if include_transposed:
        kernels.extend(
            [
                generate_transposed_input_kernel(
                    "left_transform_transposed_kernel",
                    first_size * shared_size,
                    shared_size,
                    source["u_fresh"],
                    source["u"],
                ),
                generate_transposed_input_kernel(
                    "right_transform_transposed_kernel",
                    shared_size * second_size,
                    second_size,
                    source["v_fresh"],
                    source["v"],
                ),
                generate_transposed_output_kernel(
                    "output_transform_transposed_products_kernel",
                    rank,
                    source["w_fresh"],
                    source["w"],
                    first_size,
                ),
            ]
        )
    return header + "\n\n\n" + "\n\n\n".join(kernels) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include-transposed", action="store_true")
    parser.add_argument("--research-output-fusion", action="store_true")
    parser.add_argument("--sparse-research-output-fusion", action="store_true")
    parser.add_argument(
        "--output-mode", choices=("scalar", "mfma", "both"), default="scalar"
    )
    arguments = parser.parse_args()
    if arguments.research_output_fusion or arguments.sparse_research_output_fusion:
        if arguments.include_transposed or arguments.output_mode != "scalar":
            parser.error(
                "research output fusion cannot be combined with production modes"
            )
        if arguments.research_output_fusion and arguments.sparse_research_output_fusion:
            parser.error("select only one research output fusion mode")
        if arguments.sparse_research_output_fusion:
            generated = generate_sparse_research_output_fusion_module(
                arguments.certificate
            )
        else:
            generated = generate_research_output_fusion_module(arguments.certificate)
    else:
        generated = generate_module(
            arguments.certificate, arguments.include_transposed, arguments.output_mode
        )
    arguments.output.write_text(generated)


if __name__ == "__main__":
    main()
