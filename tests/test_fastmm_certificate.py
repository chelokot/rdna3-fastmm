import json
from pathlib import Path

import pytest

from rdna3_fastmm.certificate import load_expanded_scheme
from tools.verify_reduced_scheme import load_reduced_scheme, verify_reduced_scheme


def test_load_expanded_rank_49_scheme() -> None:
    scheme = load_expanded_scheme(
        Path("certificates/4x4x4_rank49_159add/certificate.json")
    )

    assert scheme.dimensions == (4, 4, 4)
    assert scheme.rank == 49
    assert len(scheme.left_coefficients) == 49
    assert all(len(row) == 16 for row in scheme.left_coefficients)
    assert len(scheme.right_coefficients) == 49
    assert all(len(row) == 16 for row in scheme.right_coefficients)
    assert len(scheme.output_coefficients) == 16
    assert all(len(row) == 49 for row in scheme.output_coefficients)


def test_rank_49_certificate_is_exact() -> None:
    path = Path("certificates/4x4x4_rank49_159add/certificate.json")
    summary = verify_reduced_scheme(load_reduced_scheme(path))

    assert summary.dimensions == (4, 4, 4)
    assert summary.rank == 49
    assert summary.side_additions == (42, 42, 75)
    assert summary.additions == 159
    assert summary.tensor_entries == 64


def test_rank_23_56_addition_certificate_is_exact() -> None:
    path = Path("certificates/research/3x3x3_rank23_56add/certificate.json")
    summary = verify_reduced_scheme(load_reduced_scheme(path))

    assert summary.dimensions == (3, 3, 3)
    assert summary.rank == 23
    assert summary.side_additions == (13, 13, 30)
    assert summary.additions == 56
    assert summary.tensor_entries == 27


def test_rank_23_55_addition_certificate_is_exact() -> None:
    path = Path("certificates/research/3x3x3_rank23_55add/certificate.json")
    summary = verify_reduced_scheme(load_reduced_scheme(path))

    assert summary.dimensions == (3, 3, 3)
    assert summary.rank == 23
    assert summary.side_additions == (13, 14, 28)
    assert summary.additions == 55
    assert summary.tensor_entries == 27


def test_rank_29_rectangular_certificate_is_exact() -> None:
    path = Path("certificates/research/3x3x4_rank29_92add/certificate.json")
    summary = verify_reduced_scheme(load_reduced_scheme(path))

    assert summary.dimensions == (3, 3, 4)
    assert summary.rank == 29
    assert summary.side_additions == (21, 26, 45)
    assert summary.additions == 92
    assert summary.tensor_entries == 36


def test_rank_105_rectangular_certificate_is_exact() -> None:
    path = Path("certificates/research/4x6x6_rank105_430add/certificate.json")
    summary = verify_reduced_scheme(load_reduced_scheme(path))

    assert summary.dimensions == (4, 6, 6)
    assert summary.rank == 105
    assert summary.side_additions == (112, 131, 187)
    assert summary.additions == 430
    assert summary.tensor_entries == 144


def test_expanded_loader_rejects_invalid_brent_tensor(tmp_path: Path) -> None:
    source_path = Path("certificates/4x4x4_rank49_159add/certificate.json")
    source = json.loads(source_path.read_text())
    source["u"][0][0]["value"] *= -1
    invalid_path = tmp_path / "invalid-certificate.json"
    invalid_path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="Brent tensor"):
        load_expanded_scheme(invalid_path)
