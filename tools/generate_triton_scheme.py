import argparse
import hashlib
from pathlib import Path
from typing import Literal, TypedDict

from tools.export_fmm_git_scheme import LinearTerm
from tools.verify_reduced_scheme import load_reduced_scheme, verify_reduced_scheme


ExpressionKind = Literal["gate", "output"]
SourceKind = Literal["input", "products"]


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
        "    block_elements: tl.constexpr,",
        "):",
        "    element_offsets = (",
        "        tl.program_id(0) * block_elements + tl.arange(0, block_elements)",
        "    )",
        "    plane_elements = block_height * block_width",
        "    element_mask = element_offsets < plane_elements",
    ]
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


def generate_module(source_path: Path, include_transposed: bool = False) -> str:
    source = load_reduced_scheme(source_path)
    summary = verify_reduced_scheme(source)
    first_size, shared_size, second_size = summary.dimensions
    rank = summary.rank
    certificate_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    header = "\n".join(
        [
            "import triton",
            "import triton.language as tl",
            "",
            f"DIMENSIONS = ({first_size}, {shared_size}, {second_size})",
            f"RANK = {rank}",
            f'CERTIFICATE_SHA256 = "{certificate_sha256}"',
        ]
    )
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
        generate_kernel(
            "output_transform_kernel",
            rank,
            second_size,
            source["w_fresh"],
            source["w"],
            "products",
            first_size,
        ),
    ]
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
    arguments = parser.parse_args()
    arguments.output.write_text(
        generate_module(arguments.certificate, arguments.include_transposed)
    )


if __name__ == "__main__":
    main()
