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


def test_expanded_loader_rejects_invalid_brent_tensor(tmp_path: Path) -> None:
    source_path = Path("certificates/4x4x4_rank49_159add/certificate.json")
    source = json.loads(source_path.read_text())
    source["u"][0][0]["value"] *= -1
    invalid_path = tmp_path / "invalid-certificate.json"
    invalid_path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="Brent tensor"):
        load_expanded_scheme(invalid_path)
