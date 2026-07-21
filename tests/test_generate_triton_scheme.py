import ast
import hashlib
from pathlib import Path

from tools.generate_triton_scheme import generate_module, schedule_expressions
from tools.verify_reduced_scheme import load_reduced_scheme


def test_schedule_places_every_gate_and_output_once() -> None:
    source = load_reduced_scheme(
        Path("certificates/4x4x4_rank49_159add/certificate.json")
    )
    schedule = schedule_expressions(16, source["u_fresh"], source["u"])

    gates = [
        expression["index"] for expression in schedule if expression["kind"] == "gate"
    ]
    outputs = [
        expression["index"] for expression in schedule if expression["kind"] == "output"
    ]
    assert sorted(gates) == list(range(len(source["u_fresh"])))
    assert sorted(outputs) == list(range(len(source["u"])))


def test_generate_rank_49_module() -> None:
    module = generate_module(Path("certificates/4x4x4_rank49_159add/certificate.json"))

    assert "DIMENSIONS = (4, 4, 4)" in module
    assert "RANK = 49" in module
    assert "def left_transform_kernel(" in module
    assert "def right_transform_kernel(" in module
    assert "def output_transform_kernel(" in module


def test_checked_in_rank_49_module_matches_generator() -> None:
    certificate = Path("certificates/4x4x4_rank49_159add/certificate.json")
    generated = generate_module(certificate)

    assert generated == Path("src/rdna3_fastmm/generated/rank49_4x4x4.py").read_text()
    assert ast.parse(generated)
    certificate_sha256 = hashlib.sha256(certificate.read_bytes()).hexdigest()
    assert f'CERTIFICATE_SHA256 = "{certificate_sha256}"' in generated
    assert "transposed_kernel" not in generated


def test_transposed_kernels_are_opt_in() -> None:
    module = generate_module(
        Path("certificates/4x4x4_rank49_159add/certificate.json"),
        include_transposed=True,
    )

    assert "def left_transform_transposed_kernel(" in module
    assert "def right_transform_transposed_kernel(" in module
    assert "def output_transform_transposed_products_kernel(" in module
