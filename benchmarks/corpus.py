from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, cast


DTypeName = Literal["float16", "bfloat16"]
WeightLayout = Literal["out_in"]
EvidenceKind = Literal["architecture-derived", "locally-profiled"]
DEFAULT_CORPUS_PATH = Path(__file__).parent / "corpora" / "comfyui-sota-2026.json"


@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    note: str
    references: tuple[str, ...]


@dataclass(frozen=True)
class GemmCase:
    id: str
    model: str
    workload: str
    projection: str
    shape: tuple[int, int, int]
    input_output_dtype: DTypeName
    compute_dtype: DTypeName
    weight_layout: WeightLayout
    bias: bool
    priority: int
    evidence: Evidence

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def inner(self) -> int:
        return self.shape[1]

    @property
    def columns(self) -> int:
        return self.shape[2]


@dataclass(frozen=True)
class GemmCorpus:
    schema_version: int
    name: str
    operator: str
    cases: tuple[GemmCase, ...]

    def case(self, case_id: str) -> GemmCase:
        return next(case for case in self.cases if case.id == case_id)


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact_keys(value: dict[str, object], expected: set[str], context: str) -> None:
    if value.keys() != expected:
        missing = sorted(expected - value.keys())
        unexpected = sorted(value.keys() - expected)
        raise ValueError(
            f"{context} has invalid keys; missing={missing}, unexpected={unexpected}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _literal[LiteralValue: str](
    value: object, allowed: set[LiteralValue], context: str
) -> LiteralValue:
    parsed = _string(value, context)
    for allowed_value in allowed:
        if parsed == allowed_value:
            return allowed_value
    raise ValueError(f"{context} must be one of {sorted(allowed)}")


def _string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    parsed = tuple(_string(item, f"{context}[]") for item in value)
    if not parsed:
        raise ValueError(f"{context} must not be empty")
    return parsed


def _shape(value: object, context: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must contain three dimensions")
    dimensions = tuple(_integer(dimension, f"{context}[]") for dimension in value)
    if min(dimensions) < 1:
        raise ValueError(f"{context} dimensions must be positive")
    return cast(tuple[int, int, int], dimensions)


def _load_evidence(value: object, context: str) -> Evidence:
    raw = _mapping(value, context)
    _exact_keys(raw, {"kind", "note", "references"}, context)
    return Evidence(
        kind=_literal(
            raw["kind"], {"architecture-derived", "locally-profiled"}, f"{context}.kind"
        ),
        note=_string(raw["note"], f"{context}.note"),
        references=_string_tuple(raw["references"], f"{context}.references"),
    )


def _load_case(value: object, index: int) -> GemmCase:
    context = f"cases[{index}]"
    raw = _mapping(value, context)
    _exact_keys(
        raw,
        {
            "id",
            "model",
            "workload",
            "projection",
            "shape",
            "input_output_dtype",
            "compute_dtype",
            "weight_layout",
            "bias",
            "priority",
            "evidence",
        },
        context,
    )
    priority = _integer(raw["priority"], f"{context}.priority")
    if priority not in {1, 2, 3}:
        raise ValueError(f"{context}.priority must be 1, 2, or 3")
    return GemmCase(
        id=_string(raw["id"], f"{context}.id"),
        model=_string(raw["model"], f"{context}.model"),
        workload=_string(raw["workload"], f"{context}.workload"),
        projection=_string(raw["projection"], f"{context}.projection"),
        shape=_shape(raw["shape"], f"{context}.shape"),
        input_output_dtype=_literal(
            raw["input_output_dtype"],
            {"float16", "bfloat16"},
            f"{context}.input_output_dtype",
        ),
        compute_dtype=_literal(
            raw["compute_dtype"],
            {"float16", "bfloat16"},
            f"{context}.compute_dtype",
        ),
        weight_layout=_literal(
            raw["weight_layout"], {"out_in"}, f"{context}.weight_layout"
        ),
        bias=_boolean(raw["bias"], f"{context}.bias"),
        priority=priority,
        evidence=_load_evidence(raw["evidence"], f"{context}.evidence"),
    )


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> GemmCorpus:
    raw = _mapping(json.loads(path.read_text()), "corpus")
    _exact_keys(raw, {"schema_version", "name", "operator", "cases"}, "corpus")
    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported corpus schema version {schema_version}")
    operator = _string(raw["operator"], "operator")
    if operator != "torch.nn.functional.linear":
        raise ValueError(f"unsupported corpus operator {operator}")
    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty array")
    cases = tuple(_load_case(case, index) for index, case in enumerate(raw_cases))
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return GemmCorpus(
        schema_version=schema_version,
        name=_string(raw["name"], "name"),
        operator=operator,
        cases=cases,
    )
