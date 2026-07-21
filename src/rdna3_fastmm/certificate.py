from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast, NotRequired, TypedDict


class LinearTerm(TypedDict):
    index: int
    value: int


class ComplexityData(TypedDict):
    naive: int
    reduced: int


class SchemeData(TypedDict):
    n: list[int]
    m: int
    z2: bool
    u: list[list[int]]
    v: list[list[int]]
    w: list[list[int]]


class ReducedSchemeData(TypedDict):
    n: list[int]
    m: int
    z2: NotRequired[bool]
    u_fresh: list[list[LinearTerm]]
    v_fresh: list[list[LinearTerm]]
    w_fresh: list[list[LinearTerm]]
    u: list[list[LinearTerm]]
    v: list[list[LinearTerm]]
    w: list[list[LinearTerm]]
    complexity: NotRequired[ComplexityData]


@dataclass(frozen=True)
class VerificationSummary:
    dimensions: tuple[int, int, int]
    rank: int
    additions: int
    side_additions: tuple[int, int, int]
    naive_additions: int
    tensor_entries: int


@dataclass(frozen=True)
class ExpandedScheme:
    dimensions: tuple[int, int, int]
    rank: int
    left_coefficients: tuple[tuple[int, ...], ...]
    right_coefficients: tuple[tuple[int, ...], ...]
    output_coefficients: tuple[tuple[int, ...], ...]


def expand_linear_map(
    input_count: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> list[list[int]]:
    signals = [
        [int(row == column) for column in range(input_count)]
        for row in range(input_count)
    ]
    expanded_outputs: list[list[int]] = []
    for expression_index, expression in enumerate(gates + outputs):
        expanded = [0] * input_count
        for term in expression:
            source = signals[term["index"]]
            for coordinate, coefficient in enumerate(source):
                expanded[coordinate] += term["value"] * coefficient
        if expression_index < len(gates):
            signals.append(expanded)
        else:
            expanded_outputs.append(expanded)
    return expanded_outputs


def circuit_cost(gates: list[list[LinearTerm]], outputs: list[list[LinearTerm]]) -> int:
    return sum(max(len(expression) - 1, 0) for expression in gates + outputs)


def naive_addition_cost(scheme: SchemeData) -> int:
    output_count = scheme["n"][0] * scheme["n"][2]
    nonzero_coefficients = sum(
        coefficient != 0
        for factor in (scheme["u"], scheme["v"], scheme["w"])
        for row in factor
        for coefficient in row
    )
    return nonzero_coefficients - 2 * scheme["m"] - output_count


def validate_expressions(
    name: str,
    expressions: list[list[LinearTerm]],
    input_count: int,
    gate_count: int,
) -> None:
    if not isinstance(expressions, list):
        raise ValueError(f"{name} expressions must be a list")
    for expression_index, expression in enumerate(expressions):
        if not isinstance(expression, list):
            raise ValueError(f"{name} expression {expression_index} must be a list")
        if not expression:
            raise ValueError(f"{name} expression {expression_index} is empty")
        available_count = input_count + min(expression_index, gate_count)
        for term in expression:
            if not isinstance(term, dict):
                raise ValueError(
                    f"{name} expression {expression_index} has invalid term"
                )
            index = term.get("index")
            value = term.get("value")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    f"{name} expression {expression_index} has invalid index"
                )
            if not 0 <= index < available_count:
                raise ValueError(
                    f"{name} expression {expression_index} references unavailable signal {index}"
                )
            if isinstance(value, bool) or value not in (-1, 1):
                raise ValueError(
                    f"{name} expression {expression_index} has non-signed value"
                )


def validate_scheme_header(source: ReducedSchemeData) -> tuple[int, int, int, int]:
    dimensions = source["n"]
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            for dimension in dimensions
        )
    ):
        raise ValueError("scheme dimensions must be three positive integers")
    rank = source["m"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("scheme rank must be a positive integer")
    if source.get("z2") is not False:
        raise ValueError(
            "certificate must explicitly define an integer, non-Z2 circuit"
        )
    return dimensions[0], dimensions[1], dimensions[2], rank


def expand_and_verify_circuit(
    name: str,
    input_count: int,
    gates: list[list[LinearTerm]],
    outputs: list[list[LinearTerm]],
) -> tuple[list[list[int]], int]:
    validate_expressions(name, gates + outputs, input_count, len(gates))
    return expand_linear_map(input_count, gates, outputs), circuit_cost(gates, outputs)


def tensor_entries(scheme: SchemeData) -> dict[tuple[int, int, int], int]:
    entries: dict[tuple[int, int, int], int] = {}
    for term in range(scheme["m"]):
        u_nonzero = [
            (index, value) for index, value in enumerate(scheme["u"][term]) if value
        ]
        v_nonzero = [
            (index, value) for index, value in enumerate(scheme["v"][term]) if value
        ]
        w_nonzero = [
            (index, value) for index, value in enumerate(scheme["w"][term]) if value
        ]
        for u_index, u_value in u_nonzero:
            for v_index, v_value in v_nonzero:
                for w_index, w_value in w_nonzero:
                    coordinate = (u_index, v_index, w_index)
                    entries[coordinate] = (
                        entries.get(coordinate, 0) + u_value * v_value * w_value
                    )
    return {coordinate: value for coordinate, value in entries.items() if value}


def expected_tensor_entries(
    first_size: int, shared_size: int, second_size: int
) -> dict[tuple[int, int, int], int]:
    return {
        (
            row * shared_size + inner,
            inner * second_size + column,
            column * first_size + row,
        ): 1
        for row in range(first_size)
        for inner in range(shared_size)
        for column in range(second_size)
    }


def verify_reduced_scheme(source: ReducedSchemeData) -> VerificationSummary:
    first_size, shared_size, second_size, rank = validate_scheme_header(source)
    u, u_cost = expand_and_verify_circuit(
        "u", first_size * shared_size, source["u_fresh"], source["u"]
    )
    v, v_cost = expand_and_verify_circuit(
        "v", shared_size * second_size, source["v_fresh"], source["v"]
    )
    output_columns, w_cost = expand_and_verify_circuit(
        "w", rank, source["w_fresh"], source["w"]
    )
    if len(u) != rank or len(v) != rank:
        raise ValueError("input factor row count does not match rank")
    if len(output_columns) != first_size * second_size:
        raise ValueError("output circuit row count does not match output size")
    w = [
        [output_columns[column][term] for column in range(len(output_columns))]
        for term in range(rank)
    ]
    expanded = SchemeData(
        n=[first_size, shared_size, second_size],
        m=rank,
        z2=source.get("z2", False),
        u=u,
        v=v,
        w=w,
    )
    actual_tensor = tensor_entries(expanded)
    expected_tensor = expected_tensor_entries(first_size, shared_size, second_size)
    if actual_tensor != expected_tensor:
        coordinates = set(actual_tensor) | set(expected_tensor)
        mismatches = sum(
            actual_tensor.get(coordinate, 0) != expected_tensor.get(coordinate, 0)
            for coordinate in coordinates
        )
        raise ValueError(f"Brent tensor has {mismatches} mismatched entries")
    side_costs = (u_cost, v_cost, w_cost)
    additions = sum(side_costs)
    naive_additions = naive_addition_cost(expanded)
    raw_complexity = cast(dict[str, object] | None, source.get("complexity"))
    if raw_complexity is not None:
        if not isinstance(raw_complexity, dict):
            raise ValueError("declared complexity must be an object")
        if raw_complexity.get("reduced") != additions:
            raise ValueError("declared reduced complexity is incorrect")
        if raw_complexity.get("naive") != naive_additions:
            raise ValueError("declared naive complexity is incorrect")
    return VerificationSummary(
        dimensions=(first_size, shared_size, second_size),
        rank=rank,
        additions=additions,
        side_additions=side_costs,
        naive_additions=naive_additions,
        tensor_entries=len(actual_tensor),
    )


def load_reduced_scheme(path: Path) -> ReducedSchemeData:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("certificate must be a JSON object")
    parsed = cast(dict[str, object], raw)
    required_fields = ("n", "m", "u_fresh", "v_fresh", "w_fresh", "u", "v", "w")
    if any(field not in parsed for field in required_fields):
        raise ValueError("input is not a reduced scheme")
    return cast(ReducedSchemeData, parsed)


def load_expanded_scheme(path: Path) -> ExpandedScheme:
    source = load_reduced_scheme(path)
    summary = verify_reduced_scheme(source)
    first_size, shared_size, second_size = summary.dimensions
    rank = summary.rank
    left_coefficients = expand_linear_map(
        first_size * shared_size, source["u_fresh"], source["u"]
    )
    right_coefficients = expand_linear_map(
        shared_size * second_size, source["v_fresh"], source["v"]
    )
    output_columns = expand_linear_map(rank, source["w_fresh"], source["w"])
    output_coefficients = [
        output_columns[column * first_size + row]
        for row in range(first_size)
        for column in range(second_size)
    ]
    return ExpandedScheme(
        dimensions=(first_size, shared_size, second_size),
        rank=rank,
        left_coefficients=tuple(map(tuple, left_coefficients)),
        right_coefficients=tuple(map(tuple, right_coefficients)),
        output_coefficients=tuple(map(tuple, output_coefficients)),
    )
