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


def test_generate_rank_343_mfma_reconstruction() -> None:
    certificate = Path("certificates/8x8x8_rank343_1661add/certificate.json")
    module = generate_module(certificate, output_mode="mfma")
    tree = ast.parse(module)
    coefficient_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "MFMA_OUTPUT_COEFFICIENTS"
    )
    coefficients = ast.literal_eval(coefficient_assignment.value)

    assert "DIMENSIONS = (8, 8, 8)" in module
    assert "RANK = 343" in module
    assert "def output_transform_mfma_kernel(" in module
    assert "def output_transform_kernel(" not in module
    assert len(coefficients) == 343
    assert all(len(row) == 64 for row in coefficients)
    assert {value for row in coefficients for value in row} == {-1, 0, 1}
