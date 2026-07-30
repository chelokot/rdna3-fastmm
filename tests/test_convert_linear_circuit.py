import json
from pathlib import Path

import pytest

from rdna3_fastmm.certificate import load_expanded_scheme, load_reduced_scheme
from tools.convert_linear_circuit import convert_linear_circuit


SOURCE_PATH = Path("certificates/research/3x3x3_rank23_55add/source_certificate.json")
CONVERTED_PATH = Path("certificates/research/3x3x3_rank23_55add/certificate.json")


def test_checked_in_rank_23_55_addition_certificate_matches_converter() -> None:
    converted = convert_linear_circuit(json.loads(SOURCE_PATH.read_text()))

    assert converted == load_reduced_scheme(CONVERTED_PATH)


def test_rank_23_55_addition_factor_maps_match_58_addition_circuit() -> None:
    reduced = load_expanded_scheme(CONVERTED_PATH)
    original = load_expanded_scheme(
        Path("certificates/research/3x3x3_rank23_58add/certificate.json")
    )

    assert reduced == original


def test_converter_rejects_incompatible_coordinate_order() -> None:
    source = json.loads(SOURCE_PATH.read_text())
    source["coordinate_order"]["output"] = "row-major"

    with pytest.raises(ValueError, match="coordinate order"):
        convert_linear_circuit(source)
