import argparse
from pathlib import Path

from rdna3_fastmm.slp import (
    load_and_verify_slp_certificate,
    SlpProgram,
)
from tools.generate_triton_scheme import (
    generate_scheduled_kernel,
    generate_scheduled_weight_input_kernel,
    ScheduledExpression,
)


def schedule_program(
    program: SlpProgram,
    output_indices: dict[int, int],
) -> list[ScheduledExpression]:
    schedule: list[ScheduledExpression] = []
    emitted_outputs: set[int] = set()
    for gate_index, assignment in enumerate(program.assignments):
        if assignment.signal_index != program.input_count + gate_index:
            raise ValueError("SLP signal numbering is not sequential")
        schedule.append(
            ScheduledExpression(
                kind="gate",
                index=gate_index,
                dependencies=list(assignment.dependencies),
                source=assignment.source,
            )
        )
        if not assignment.target.startswith("o"):
            continue
        source_output_index = int(assignment.target[1:])
        output_index = output_indices[source_output_index]
        schedule.append(
            ScheduledExpression(
                kind="output",
                index=output_index,
                dependencies=[assignment.signal_index],
                source=f"signal_{assignment.signal_index}",
            )
        )
        emitted_outputs.add(output_index)
    if emitted_outputs != set(range(program.output_count)):
        raise ValueError("SLP schedule did not emit every output")
    return schedule


def identity_output_indices(output_count: int) -> dict[int, int]:
    return {index: index for index in range(output_count)}


def column_major_output_indices(
    first_size: int,
    second_size: int,
) -> dict[int, int]:
    return {
        row * second_size + column: column * first_size + row
        for row in range(first_size)
        for column in range(second_size)
    }


def generate_module(certificate_path: Path) -> str:
    certificate, summary = load_and_verify_slp_certificate(certificate_path)
    first_size, shared_size, second_size = summary.dimensions
    rank = summary.rank
    left_schedule = schedule_program(
        certificate.left,
        identity_output_indices(rank),
    )
    right_schedule = schedule_program(
        certificate.right,
        identity_output_indices(rank),
    )
    output_schedule = schedule_program(
        certificate.output,
        column_major_output_indices(first_size, second_size),
    )
    header = "\n".join(
        [
            "import triton",
            "import triton.language as tl",
            "",
            f"DIMENSIONS = {summary.dimensions}",
            f"RANK = {rank}",
            f'CERTIFICATE_SHA256 = "{summary.sha256}"',
        ]
    )
    kernels = [
        generate_scheduled_kernel(
            "left_transform_kernel",
            first_size * shared_size,
            shared_size,
            left_schedule,
            "input",
            first_size,
        ),
        generate_scheduled_kernel(
            "right_transform_kernel",
            shared_size * second_size,
            second_size,
            right_schedule,
            "input",
            shared_size,
        ),
        generate_scheduled_weight_input_kernel(
            "right_transform_weight_kernel",
            shared_size * second_size,
            second_size,
            right_schedule,
        ),
        generate_scheduled_kernel(
            "output_transform_kernel",
            rank,
            second_size,
            output_schedule,
            "products",
            first_size,
        ),
        generate_scheduled_kernel(
            "output_transform_bias_kernel",
            rank,
            second_size,
            output_schedule,
            "products",
            first_size,
            include_bias=True,
        ),
    ]
    return header + "\n\n\n" + "\n\n\n".join(kernels) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.write_text(generate_module(arguments.certificate))


if __name__ == "__main__":
    main()
