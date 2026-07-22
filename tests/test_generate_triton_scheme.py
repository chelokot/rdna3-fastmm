import ast
import hashlib
from pathlib import Path

from rdna3_fastmm.certificate import expand_linear_map
from tools.generate_triton_scheme import (
    generate_module,
    generate_research_output_fusion_module,
    schedule_expressions,
)
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
    assert "def right_transform_weight_kernel(" in module
    assert "def output_transform_kernel(" in module
    assert "def output_transform_bias_kernel(" in module
    assert "FUSED_OUTPUT_COEFFICIENTS" not in module
    assert "def fused_product_output_kernel(" not in module
    assert "def atomic_product_output_kernel(" not in module


def test_generate_rank_49_research_coefficients_and_serial_fusion() -> None:
    certificate = Path("certificates/4x4x4_rank49_159add/certificate.json")
    source = load_reduced_scheme(certificate)
    module = generate_research_output_fusion_module(certificate)
    tree = ast.parse(module)
    coefficient_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "FUSED_OUTPUT_COEFFICIENTS"
    )
    coefficients = ast.literal_eval(coefficient_assignment.value)
    output_columns = expand_linear_map(49, source["w_fresh"], source["w"])
    expected = tuple(
        tuple(
            output_columns[column * 4 + row][rank]
            for row in range(4)
            for column in range(4)
        )
        for rank in range(49)
    )
    products = tuple(range(-24, 25))
    signals = list(products)
    for expression in source["w_fresh"]:
        signals.append(
            sum(term["value"] * signals[term["index"]] for term in expression)
        )
    circuit_outputs = tuple(
        sum(term["value"] * signals[term["index"]] for term in expression)
        for expression in source["w"]
    )
    row_major_circuit_outputs = tuple(
        circuit_outputs[column * 4 + row] for row in range(4) for column in range(4)
    )
    coefficient_outputs = tuple(
        sum(coefficients[rank][output] * products[rank] for rank in range(49))
        for output in range(16)
    )
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "fused_product_output_kernel"
    )

    assert coefficients == expected
    assert len(coefficients) == 49
    assert all(len(row) == 16 for row in coefficients)
    assert {value for row in coefficients for value in row} == {-1, 0, 1}
    assert sum(value != 0 for row in coefficients for value in row) == 196
    assert coefficient_outputs == row_major_circuit_outputs
    assert [argument.arg for argument in kernel.args.args] == [
        "left_transformed",
        "right_transformed",
        "coefficients",
        "output",
        "bias",
        "block_rows",
        "block_inner",
        "block_columns",
        "output_row_count",
        "output_columns",
        "block_m",
        "block_n",
        "block_k",
        "has_bias",
    ]
    assert "for rank_index in tl.range(0, 49, loop_unroll_factor=1):" in module
    assert module.count("product = tl.dot(") == 2


def test_generate_rank_49_atomic_product_output() -> None:
    module = generate_research_output_fusion_module(
        Path("certificates/4x4x4_rank49_159add/certificate.json")
    )
    tree = ast.parse(module)
    atomic_kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "atomic_product_output_kernel"
    )
    finalizer_kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "atomic_output_finalize_kernel"
    )

    assert [argument.arg for argument in atomic_kernel.args.args] == [
        "left_transformed",
        "right_transformed",
        "coefficients",
        "accumulator",
        "block_rows",
        "block_inner",
        "block_columns",
        "output_row_count",
        "output_columns",
        "block_m",
        "block_n",
        "block_k",
    ]
    assert [argument.arg for argument in finalizer_kernel.args.args] == [
        "accumulator",
        "output",
        "bias",
        "output_elements",
        "output_columns",
        "block_elements",
        "has_bias",
        "clear_accumulator",
    ]
    assert "rank_index = tl.program_id(2)" in module
    assert "for output_block in tl.range(0, 16, loop_unroll_factor=1):" in module
    assert "tl.atomic_add(" in module
    assert "coefficient_value != 0.0" in module
    assert 'sem="relaxed"' in module
    assert "if clear_accumulator:" in module


def test_checked_in_rank_49_module_matches_generator() -> None:
    certificate = Path("certificates/4x4x4_rank49_159add/certificate.json")
    generated = generate_module(certificate)

    assert generated == Path("src/rdna3_fastmm/generated/rank49_4x4x4.py").read_text()
    assert ast.parse(generated)
    certificate_sha256 = hashlib.sha256(certificate.read_bytes()).hexdigest()
    assert f'CERTIFICATE_SHA256 = "{certificate_sha256}"' in generated
    assert "transposed_kernel" not in generated
    assert "fused_product_output_kernel" not in generated
    assert "atomic_product_output_kernel" not in generated


def test_checked_in_rank_49_research_module_matches_generator() -> None:
    certificate = Path("certificates/4x4x4_rank49_159add/certificate.json")
    generated = generate_research_output_fusion_module(certificate)

    assert (
        generated
        == Path("research/prototypes/generated_rank49_output_fusion.py").read_text()
    )
    assert ast.parse(generated)


def test_checked_in_rank_7_research_module_matches_generator() -> None:
    certificate = Path("certificates/research/2x2x2_rank7_15add/certificate.json")
    generated = generate_module(certificate)

    assert generated == Path("research/generated/rank7_2x2x2.py").read_text()
    assert ast.parse(generated)


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


def test_generate_rank_343_scalar_without_fused_reconstruction() -> None:
    module = generate_module(
        Path("certificates/8x8x8_rank343_1661add/certificate.json")
    )

    assert "def output_transform_kernel(" in module
    assert "FUSED_OUTPUT_COEFFICIENTS" not in module
    assert "def fused_product_output_kernel(" not in module
