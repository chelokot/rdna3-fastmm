from copy import deepcopy

import pytest

from tools.export_fmm_git_scheme import ReducedSchemeData
from tools.verify_reduced_scheme import verify_reduced_scheme


def identity_panel_scheme() -> ReducedSchemeData:
    return ReducedSchemeData(
        n=[1, 1, 2],
        m=2,
        z2=False,
        complexity={"naive": 0, "reduced": 0},
        u_fresh=[],
        v_fresh=[],
        w_fresh=[],
        u=[[{"index": 0, "value": 1}], [{"index": 0, "value": 1}]],
        v=[[{"index": 0, "value": 1}], [{"index": 1, "value": 1}]],
        w=[[{"index": 0, "value": 1}], [{"index": 1, "value": 1}]],
    )


def test_verify_reduced_scheme_accepts_exact_tensor() -> None:
    summary = verify_reduced_scheme(identity_panel_scheme())
    assert summary.dimensions == (1, 1, 2)
    assert summary.rank == 2
    assert summary.additions == 0
    assert summary.tensor_entries == 2


def test_verify_reduced_scheme_rejects_wrong_tensor() -> None:
    scheme = deepcopy(identity_panel_scheme())
    scheme["w"][1] = [{"index": 0, "value": 1}]
    with pytest.raises(ValueError, match="Brent tensor"):
        verify_reduced_scheme(scheme)


def test_verify_reduced_scheme_rejects_wrong_declared_cost() -> None:
    scheme = deepcopy(identity_panel_scheme())
    scheme["complexity"]["reduced"] = 1
    with pytest.raises(ValueError, match="declared reduced"):
        verify_reduced_scheme(scheme)


def test_verify_reduced_scheme_rejects_non_signed_edges() -> None:
    scheme = deepcopy(identity_panel_scheme())
    scheme["u"][0] = [{"index": 0, "value": 2}]
    with pytest.raises(ValueError, match="non-signed"):
        verify_reduced_scheme(scheme)


def test_verify_reduced_scheme_rejects_binary_field_certificate() -> None:
    scheme = deepcopy(identity_panel_scheme())
    scheme["z2"] = True
    with pytest.raises(ValueError, match="non-Z2"):
        verify_reduced_scheme(scheme)


def test_verify_reduced_scheme_rejects_nonpositive_dimensions() -> None:
    scheme = deepcopy(identity_panel_scheme())
    scheme["n"] = [0, 1, 2]
    with pytest.raises(ValueError, match="positive integers"):
        verify_reduced_scheme(scheme)
