import argparse
import json
from pathlib import Path
from typing import cast

from rdna3_fastmm.certificate import (
    ComplexityData,
    LinearTerm,
    ReducedSchemeData,
    verify_reduced_scheme,
)


def require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], value)


def require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[object], value)


def require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def require_sign(value: object, name: str) -> int:
    sign = require_integer(value, name)
    if sign not in (-1, 1):
        raise ValueError(f"{name} must be -1 or 1")
    return sign


def convert_gate(value: object, name: str) -> list[LinearTerm]:
    gate = require_list(value, name)
    if len(gate) != 4:
        raise ValueError(f"{name} must contain two signed signal references")
    return [
        LinearTerm(
            index=require_integer(gate[0], f"{name} left index"),
            value=require_sign(gate[1], f"{name} left sign"),
        ),
        LinearTerm(
            index=require_integer(gate[2], f"{name} right index"),
            value=require_sign(gate[3], f"{name} right sign"),
        ),
    ]


def convert_output(value: object, name: str) -> list[LinearTerm]:
    output = require_list(value, name)
    if len(output) != 2:
        raise ValueError(f"{name} must contain one signed signal reference")
    return [
        LinearTerm(
            index=require_integer(output[0], f"{name} index"),
            value=require_sign(output[1], f"{name} sign"),
        )
    ]


def convert_circuit(
    value: object,
    name: str,
    expected_input_count: int,
    expected_output_count: int,
) -> tuple[list[list[LinearTerm]], list[list[LinearTerm]]]:
    circuit = require_mapping(value, f"{name} circuit")
    input_count = require_integer(
        circuit.get("input_count"), f"{name} circuit input count"
    )
    if input_count != expected_input_count:
        raise ValueError(f"{name} circuit input count is incorrect")
    raw_gates = require_list(circuit.get("gates"), f"{name} circuit gates")
    raw_outputs = require_list(circuit.get("outputs"), f"{name} circuit outputs")
    if len(raw_outputs) != expected_output_count:
        raise ValueError(f"{name} circuit output count is incorrect")
    gates = [
        convert_gate(gate, f"{name} gate {index}")
        for index, gate in enumerate(raw_gates)
    ]
    outputs = [
        convert_output(output, f"{name} output {index}")
        for index, output in enumerate(raw_outputs)
    ]
    return gates, outputs


def convert_linear_circuit(value: object) -> ReducedSchemeData:
    source = require_mapping(value, "source")
    if source.get("format") != "matmul-linear-circuit-v1":
        raise ValueError("unsupported linear circuit format")
    raw_dimensions = require_list(source.get("dimensions"), "dimensions")
    if len(raw_dimensions) != 3:
        raise ValueError("dimensions must contain three values")
    dimensions = [
        require_integer(dimension, f"dimension {index}")
        for index, dimension in enumerate(raw_dimensions)
    ]
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError("dimensions must be positive")
    rank = require_integer(source.get("rank"), "rank")
    if rank < 1:
        raise ValueError("rank must be positive")
    coordinate_order = require_mapping(
        source.get("coordinate_order"), "coordinate order"
    )
    if coordinate_order != {
        "left": "row-major",
        "right": "row-major",
        "output": "column-major",
    }:
        raise ValueError("unsupported coordinate order")
    circuits = require_mapping(source.get("circuits"), "circuits")
    first_size, shared_size, second_size = dimensions
    u_fresh, u = convert_circuit(
        circuits.get("left"),
        "left",
        first_size * shared_size,
        rank,
    )
    v_fresh, v = convert_circuit(
        circuits.get("right"),
        "right",
        shared_size * second_size,
        rank,
    )
    w_fresh, w = convert_circuit(
        circuits.get("output"),
        "output",
        rank,
        first_size * second_size,
    )
    scheme = ReducedSchemeData(
        n=dimensions,
        m=rank,
        z2=False,
        u_fresh=u_fresh,
        v_fresh=v_fresh,
        w_fresh=w_fresh,
        u=u,
        v=v,
        w=w,
    )
    summary = verify_reduced_scheme(scheme)
    scheme["complexity"] = ComplexityData(
        naive=summary.naive_additions,
        reduced=summary.additions,
    )
    return scheme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    raw = cast(object, json.loads(arguments.source.read_text()))
    converted = convert_linear_circuit(raw)
    arguments.output.write_text(json.dumps(converted, indent=4) + "\n")


if __name__ == "__main__":
    main()
