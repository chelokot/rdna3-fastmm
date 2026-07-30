from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
from typing import cast


TARGET_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@dataclass(frozen=True)
class SlpAssignment:
    target: str
    signal_index: int
    dependencies: tuple[int, ...]
    source: str
    coefficients: tuple[Fraction, ...]
    additions: int
    scalings: int


@dataclass(frozen=True)
class SlpProgram:
    input_count: int
    output_count: int
    assignments: tuple[SlpAssignment, ...]
    output_signals: tuple[int, ...]
    output_coefficients: tuple[tuple[Fraction, ...], ...]
    additions: int
    scalings: int

    @property
    def operations(self) -> int:
        return self.additions + self.scalings


@dataclass(frozen=True)
class SlpCertificate:
    dimensions: tuple[int, int, int]
    rank: int
    left: SlpProgram
    right: SlpProgram
    output: SlpProgram
    sha256: str


@dataclass(frozen=True)
class SlpVerificationSummary:
    dimensions: tuple[int, int, int]
    rank: int
    additions: tuple[int, int, int]
    scalings: tuple[int, int, int]
    operations: tuple[int, int, int]
    tensor_entries: int
    sha256: str


@dataclass(frozen=True)
class _Signal:
    index: int
    coefficients: tuple[Fraction, ...]


@dataclass(frozen=True)
class _ExpressionValue:
    scalar: Fraction | None
    coefficients: tuple[Fraction, ...] | None
    dependencies: tuple[int, ...]
    source: str
    additions: int
    scalings: int


def _merge_dependencies(*groups: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(index for group in groups for index in group))


def _combine_coefficients(
    first: tuple[Fraction, ...],
    second: tuple[Fraction, ...],
    sign: int,
) -> tuple[Fraction, ...]:
    return tuple(
        first_value + sign * second_value
        for first_value, second_value in zip(first, second, strict=True)
    )


def _scale_coefficients(
    coefficients: tuple[Fraction, ...], scale: Fraction
) -> tuple[Fraction, ...]:
    return tuple(scale * coefficient for coefficient in coefficients)


def _is_dyadic(value: Fraction) -> bool:
    denominator = value.denominator
    return denominator > 0 and denominator & (denominator - 1) == 0


def _evaluate_expression(
    node: ast.expr,
    signals: dict[str, _Signal],
) -> _ExpressionValue:
    if isinstance(node, ast.Name):
        signal = signals.get(node.id)
        if signal is None:
            raise ValueError(f"SLP references unavailable signal {node.id}")
        return _ExpressionValue(
            scalar=None,
            coefficients=signal.coefficients,
            dependencies=(signal.index,),
            source=f"signal_{signal.index}",
            additions=0,
            scalings=0,
        )
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("SLP constants must be integers")
        return _ExpressionValue(
            scalar=Fraction(node.value),
            coefficients=None,
            dependencies=(),
            source=str(node.value),
            additions=0,
            scalings=0,
        )
    if not isinstance(node, ast.BinOp):
        raise ValueError(f"unsupported SLP expression {type(node).__name__}")
    left = _evaluate_expression(node.left, signals)
    right = _evaluate_expression(node.right, signals)
    dependencies = _merge_dependencies(left.dependencies, right.dependencies)
    additions = left.additions + right.additions
    scalings = left.scalings + right.scalings
    if isinstance(node.op, (ast.Add, ast.Sub)):
        left_coefficients = left.coefficients
        right_coefficients = right.coefficients
        if left_coefficients is None or right_coefficients is None:
            raise ValueError("SLP additions require two linear expressions")
        sign = 1 if isinstance(node.op, ast.Add) else -1
        operator = "+" if sign == 1 else "-"
        return _ExpressionValue(
            scalar=None,
            coefficients=_combine_coefficients(
                left_coefficients,
                right_coefficients,
                sign,
            ),
            dependencies=dependencies,
            source=f"({left.source} {operator} {right.source})",
            additions=additions + 1,
            scalings=scalings,
        )
    if isinstance(node.op, ast.Mult):
        if left.scalar is not None and right.coefficients is not None:
            scale = left.scalar
            coefficients = right.coefficients
        elif right.scalar is not None and left.coefficients is not None:
            scale = right.scalar
            coefficients = left.coefficients
        else:
            raise ValueError("SLP multiplication requires one scalar")
        if not _is_dyadic(scale):
            raise ValueError("SLP scaling must be dyadic")
        return _ExpressionValue(
            scalar=None,
            coefficients=_scale_coefficients(coefficients, scale),
            dependencies=dependencies,
            source=f"({left.source} * {right.source})",
            additions=additions,
            scalings=scalings + int(abs(scale) != 1),
        )
    if isinstance(node.op, ast.Div):
        left_coefficients = left.coefficients
        divisor = right.scalar
        if left_coefficients is None or divisor is None or divisor == 0:
            raise ValueError("SLP division requires a linear expression and scalar")
        scale = Fraction(1, 1) / divisor
        if not _is_dyadic(scale):
            raise ValueError("SLP scaling must be dyadic")
        return _ExpressionValue(
            scalar=None,
            coefficients=_scale_coefficients(left_coefficients, scale),
            dependencies=dependencies,
            source=f"({left.source} / {right.source})",
            additions=additions,
            scalings=scalings + int(abs(scale) != 1),
        )
    raise ValueError(f"unsupported SLP operator {type(node.op).__name__}")


def load_slp_program(
    path: Path,
    input_count: int,
    output_count: int,
) -> SlpProgram:
    if input_count < 1 or output_count < 1:
        raise ValueError("SLP input and output counts must be positive")
    signals = {
        f"i{index}": _Signal(
            index,
            tuple(Fraction(coordinate == index) for coordinate in range(input_count)),
        )
        for index in range(input_count)
    }
    assignments: list[SlpAssignment] = []
    outputs: dict[int, _Signal] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if not line.endswith(";") or line.count(":=") != 1:
            raise ValueError(f"{path}:{line_number}: invalid SLP assignment")
        target, expression_source = line[:-1].split(":=", 1)
        if TARGET_PATTERN.fullmatch(target) is None:
            raise ValueError(f"{path}:{line_number}: invalid SLP target")
        if target in signals:
            raise ValueError(f"{path}:{line_number}: duplicate SLP target {target}")
        try:
            expression = ast.parse(expression_source, mode="eval").body
        except SyntaxError as error:
            raise ValueError(f"{path}:{line_number}: invalid SLP expression") from error
        value = _evaluate_expression(expression, signals)
        coefficients = value.coefficients
        if coefficients is None:
            raise ValueError(f"{path}:{line_number}: SLP result must be linear")
        signal = _Signal(input_count + len(assignments), coefficients)
        assignment = SlpAssignment(
            target=target,
            signal_index=signal.index,
            dependencies=value.dependencies,
            source=value.source,
            coefficients=coefficients,
            additions=value.additions,
            scalings=value.scalings,
        )
        assignments.append(assignment)
        signals[target] = signal
        if target.startswith("o") and target[1:].isdigit():
            output_index = int(target[1:])
            if not 0 <= output_index < output_count:
                raise ValueError(
                    f"{path}:{line_number}: output index {output_index} is out of range"
                )
            outputs[output_index] = signal
    if len(outputs) != output_count:
        missing = sorted(set(range(output_count)) - outputs.keys())
        raise ValueError(f"{path}: missing SLP outputs {missing}")
    ordered_outputs = tuple(outputs[index] for index in range(output_count))
    return SlpProgram(
        input_count=input_count,
        output_count=output_count,
        assignments=tuple(assignments),
        output_signals=tuple(signal.index for signal in ordered_outputs),
        output_coefficients=tuple(signal.coefficients for signal in ordered_outputs),
        additions=sum(assignment.additions for assignment in assignments),
        scalings=sum(assignment.scalings for assignment in assignments),
    )


def _string_field(source: dict[str, object], name: str) -> str:
    value = source.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"certificate {name} must be a nonempty string")
    return value


def _certificate_sha256(
    manifest_path: Path,
    program_paths: tuple[tuple[str, Path], ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(manifest_path.read_bytes())
    for name, path in program_paths:
        digest.update(b"\0")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_slp_certificate(manifest_path: Path) -> SlpCertificate:
    parsed = json.loads(manifest_path.read_text())
    if not isinstance(parsed, dict):
        raise ValueError("SLP certificate must be an object")
    source = cast(dict[str, object], parsed)
    if source.get("schema_version") != 1:
        raise ValueError("unsupported SLP certificate schema")
    dimensions = source.get("n")
    rank = source.get("m")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            for dimension in dimensions
        )
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 1
    ):
        raise ValueError("invalid SLP dimensions or rank")
    if source.get("domain") != "dyadic-rational":
        raise ValueError("SLP certificate must use the dyadic-rational domain")
    if source.get("input_order") != "row-major":
        raise ValueError("SLP certificate inputs must be row-major")
    if source.get("output_order") != "row-major":
        raise ValueError("SLP certificate outputs must be row-major")
    raw_programs = source.get("programs")
    raw_complexity = source.get("complexity")
    if not isinstance(raw_programs, dict) or not isinstance(raw_complexity, dict):
        raise ValueError("SLP certificate programs and complexity must be objects")
    programs = cast(dict[str, object], raw_programs)
    complexity = cast(dict[str, object], raw_complexity)
    manifest_directory = manifest_path.parent.resolve()
    resolved_paths: list[tuple[str, Path]] = []
    for name in ("u", "v", "w"):
        relative_path = Path(_string_field(programs, name))
        path = (manifest_directory / relative_path).resolve()
        if path.parent != manifest_directory:
            raise ValueError("SLP program paths must stay inside the certificate")
        resolved_paths.append((name, path))
    first_size, shared_size, second_size = cast(list[int], dimensions)
    left = load_slp_program(
        resolved_paths[0][1],
        first_size * shared_size,
        rank,
    )
    right = load_slp_program(
        resolved_paths[1][1],
        shared_size * second_size,
        rank,
    )
    output = load_slp_program(
        resolved_paths[2][1],
        rank,
        first_size * second_size,
    )
    for name, program in (("u", left), ("v", right), ("w", output)):
        expected_operations = complexity.get(name)
        if (
            isinstance(expected_operations, bool)
            or not isinstance(expected_operations, int)
            or expected_operations != program.operations
        ):
            raise ValueError(f"SLP {name} complexity does not match the manifest")
    return SlpCertificate(
        dimensions=(first_size, shared_size, second_size),
        rank=rank,
        left=left,
        right=right,
        output=output,
        sha256=_certificate_sha256(manifest_path, tuple(resolved_paths)),
    )


def verify_slp_certificate(
    certificate: SlpCertificate,
) -> SlpVerificationSummary:
    first_size, shared_size, second_size = certificate.dimensions
    entries: dict[tuple[int, int, int], Fraction] = {}
    for term in range(certificate.rank):
        left_nonzero = [
            (index, value)
            for index, value in enumerate(certificate.left.output_coefficients[term])
            if value
        ]
        right_nonzero = [
            (index, value)
            for index, value in enumerate(certificate.right.output_coefficients[term])
            if value
        ]
        output_nonzero = [
            (index, certificate.output.output_coefficients[index][term])
            for index in range(first_size * second_size)
            if certificate.output.output_coefficients[index][term]
        ]
        for left_index, left_value in left_nonzero:
            for right_index, right_value in right_nonzero:
                for output_index, output_value in output_nonzero:
                    coordinate = (left_index, right_index, output_index)
                    entries[coordinate] = (
                        entries.get(coordinate, Fraction(0))
                        + left_value * right_value * output_value
                    )
    entries = {coordinate: value for coordinate, value in entries.items() if value}
    expected = {
        (
            row * shared_size + inner,
            inner * second_size + column,
            row * second_size + column,
        ): Fraction(1)
        for row in range(first_size)
        for inner in range(shared_size)
        for column in range(second_size)
    }
    if entries != expected:
        mismatches = sum(
            entries.get(coordinate, Fraction(0))
            != expected.get(coordinate, Fraction(0))
            for coordinate in entries.keys() | expected.keys()
        )
        raise ValueError(f"SLP Brent tensor has {mismatches} mismatches")
    programs = (certificate.left, certificate.right, certificate.output)
    return SlpVerificationSummary(
        dimensions=certificate.dimensions,
        rank=certificate.rank,
        additions=(
            programs[0].additions,
            programs[1].additions,
            programs[2].additions,
        ),
        scalings=(
            programs[0].scalings,
            programs[1].scalings,
            programs[2].scalings,
        ),
        operations=(
            programs[0].operations,
            programs[1].operations,
            programs[2].operations,
        ),
        tensor_entries=len(entries),
        sha256=certificate.sha256,
    )


def load_and_verify_slp_certificate(
    manifest_path: Path,
) -> tuple[SlpCertificate, SlpVerificationSummary]:
    certificate = load_slp_certificate(manifest_path)
    return certificate, verify_slp_certificate(certificate)
