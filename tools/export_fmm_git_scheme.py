import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import cast

from rdna3_fastmm.certificate import (
    ComplexityData,
    expand_linear_map,
    LinearTerm,
    ReducedSchemeData,
    SchemeData,
)

__all__ = [
    "ComplexityData",
    "LinearTerm",
    "ReducedSchemeData",
    "SchemeData",
    "expand_linear_map",
]


def expand_reduced_scheme(scheme: ReducedSchemeData) -> SchemeData:
    first_size, shared_size, second_size = scheme["n"]
    rank = scheme["m"]
    u = expand_linear_map(first_size * shared_size, scheme["u_fresh"], scheme["u"])
    v = expand_linear_map(shared_size * second_size, scheme["v_fresh"], scheme["v"])
    output_columns = expand_linear_map(rank, scheme["w_fresh"], scheme["w"])
    w = [
        [output_columns[column][row] for column in range(len(output_columns))]
        for row in range(rank)
    ]
    return SchemeData(
        n=scheme["n"],
        m=rank,
        z2=scheme.get("z2", False),
        u=u,
        v=v,
        w=w,
    )


def reduce_coefficient(value: object, modulus: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"unsupported coefficient: {value!r}")
    coefficient = Fraction(value)
    if modulus is None:
        if coefficient.denominator != 1:
            raise ValueError("fractional coefficients require --modulus")
        return coefficient.numerator
    if modulus < 2:
        raise ValueError("modulus must be at least two")
    try:
        denominator_inverse = pow(coefficient.denominator, -1, modulus)
    except ValueError as error:
        raise ValueError(
            f"coefficient denominator is not invertible modulo {modulus}"
        ) from error
    residue = coefficient.numerator * denominator_inverse % modulus
    return residue - modulus if residue > modulus // 2 else residue


def normalize_factor(value: object, modulus: int | None) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("scheme factor must be a list")
    factor: list[list[int]] = []
    for raw_row in value:
        if not isinstance(raw_row, list):
            raise ValueError("scheme factor row must be a list")
        factor.append(
            [reduce_coefficient(coefficient, modulus) for coefficient in raw_row]
        )
    return factor


def load_git_scheme(
    repository: Path,
    object_name: str,
    modulus: int | None = None,
) -> SchemeData:
    raw = subprocess.run(
        ["git", "-C", str(repository), "show", object_name],
        check=True,
        capture_output=True,
    ).stdout
    parsed = cast(dict[str, object], json.loads(raw))
    if "u_fresh" in parsed:
        scheme = expand_reduced_scheme(cast(ReducedSchemeData, parsed))
        if modulus is None:
            return scheme
        return SchemeData(
            n=scheme["n"],
            m=scheme["m"],
            z2=scheme["z2"],
            u=normalize_factor(scheme["u"], modulus),
            v=normalize_factor(scheme["v"], modulus),
            w=normalize_factor(scheme["w"], modulus),
        )
    dimensions = parsed.get("n")
    rank = parsed.get("m")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 3
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in dimensions
        )
        or isinstance(rank, bool)
        or not isinstance(rank, int)
    ):
        raise ValueError("invalid scheme dimensions or rank")
    return SchemeData(
        n=cast(list[int], dimensions),
        m=rank,
        z2=parsed.get("z2") is True,
        u=normalize_factor(parsed.get("u"), modulus),
        v=normalize_factor(parsed.get("v"), modulus),
        w=normalize_factor(parsed.get("w"), modulus),
    )


def serialize_plain_text(scheme: SchemeData) -> str:
    first_size, shared_size, second_size = scheme["n"]
    rank = scheme["m"]
    dimensions = (
        first_size * shared_size,
        shared_size * second_size,
        second_size * first_size,
    )
    factors = (scheme["u"], scheme["v"], scheme["w"])
    if any(len(factor) != rank for factor in factors):
        raise ValueError("factor row count does not match rank")
    if any(
        len(row) != dimension
        for factor, dimension in zip(factors, dimensions, strict=True)
        for row in factor
    ):
        raise ValueError("factor row width does not match matrix dimensions")
    if any(
        coefficient not in (-1, 0, 1)
        for factor in factors
        for row in factor
        for coefficient in row
    ):
        raise ValueError("plain-text flip search requires ternary coefficients")
    lines = [f"{first_size} {shared_size} {second_size} {rank}"]
    lines.extend(
        " ".join(str(coefficient) for coefficient in row)
        for factor in factors
        for row in factor
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("object_name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--modulus", type=int)
    arguments = parser.parse_args()
    scheme = load_git_scheme(
        arguments.repository,
        arguments.object_name,
        arguments.modulus,
    )
    arguments.output.write_text(serialize_plain_text(scheme))


if __name__ == "__main__":
    main()
