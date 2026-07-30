import hashlib
from pathlib import Path

import pytest

from rdna3_fastmm.slp import (
    load_and_verify_slp_certificate,
    load_slp_program,
)
from tools.generate_slp_triton_scheme import generate_module


CERTIFICATE = Path("certificates/4x4x4_rank48_accurate/certificate.json")
GENERATED = Path("src/rdna3_fastmm/generated/rank48_4x4x4.py")


def test_rank48_slp_certificate_is_exact() -> None:
    certificate, summary = load_and_verify_slp_certificate(CERTIFICATE)

    assert summary.dimensions == (4, 4, 4)
    assert summary.rank == 48
    assert summary.additions == (80, 68, 108)
    assert summary.scalings == (4, 8, 16)
    assert summary.operations == (84, 76, 124)
    assert summary.tensor_entries == 64
    assert summary.sha256 == (
        "c7103a1165af22d4e1a417607e9618b0bbae1b69e0c48ed7fa2a5aba26a3bcb7"
    )
    assert tuple(
        len(program.assignments)
        for program in (certificate.left, certificate.right, certificate.output)
    ) == (82, 60, 72)


@pytest.mark.parametrize(
    ("name", "sha256"),
    (
        (
            "left.slp",
            "af7719f288b346ae0dde8b8bf4a4a7bcf8d160ec8387c4a72d560a4bdb65f19b",
        ),
        (
            "right.slp",
            "c09dda6d06633b9b438683819be49151c2ff24d26bca1097c8023ad534d390fa",
        ),
        (
            "output.slp",
            "d441067f8e8d5d9c374ab2f290763ecc5f4738f989e1e3570d844d8207ed042f",
        ),
    ),
)
def test_rank48_vendored_slp_matches_upstream(name: str, sha256: str) -> None:
    path = CERTIFICATE.parent / name

    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256


def test_rank48_generated_module_is_current() -> None:
    assert generate_module(CERTIFICATE) == GENERATED.read_text()


def test_slp_rejects_unavailable_signal(tmp_path: Path) -> None:
    path = tmp_path / "invalid.slp"
    path.write_text("o0:=i0+missing;\n")

    with pytest.raises(ValueError, match="unavailable signal missing"):
        load_slp_program(path, 1, 1)


def test_slp_rejects_nondyadic_scale(tmp_path: Path) -> None:
    path = tmp_path / "invalid.slp"
    path.write_text("o0:=i0/3;\n")

    with pytest.raises(ValueError, match="dyadic"):
        load_slp_program(path, 1, 1)


def test_slp_rejects_missing_output(tmp_path: Path) -> None:
    path = tmp_path / "invalid.slp"
    path.write_text("x0:=i0+i0;\n")

    with pytest.raises(ValueError, match=r"missing SLP outputs \[0\]"):
        load_slp_program(path, 1, 1)
