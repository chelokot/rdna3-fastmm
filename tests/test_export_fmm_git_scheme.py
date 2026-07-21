import pytest

from tools.export_fmm_git_scheme import (
    expand_reduced_scheme,
    reduce_coefficient,
    ReducedSchemeData,
    SchemeData,
    serialize_plain_text,
)


def test_serialize_plain_text() -> None:
    scheme = SchemeData(
        n=[1, 1, 2],
        m=2,
        z2=False,
        u=[[1], [1]],
        v=[[1, 0], [0, 1]],
        w=[[1, 0], [0, 1]],
    )
    assert serialize_plain_text(scheme) == ("1 1 2 2\n1\n1\n1 0\n0 1\n1 0\n0 1\n")


def test_serialize_plain_text_rejects_nonternary_coefficients() -> None:
    scheme = SchemeData(
        n=[1, 1, 1],
        m=1,
        z2=False,
        u=[[2]],
        v=[[1]],
        w=[[1]],
    )
    with pytest.raises(ValueError, match="ternary"):
        serialize_plain_text(scheme)


@pytest.mark.parametrize(
    ("coefficient", "expected"),
    [(1, 1), (-1, -1), ("1/8", -1), ("-1/8", 1)],
)
def test_reduce_coefficient_modulo_three(
    coefficient: int | str,
    expected: int,
) -> None:
    assert reduce_coefficient(coefficient, 3) == expected


def test_reduce_coefficient_requires_modulus_for_fraction() -> None:
    with pytest.raises(ValueError, match="fractional"):
        reduce_coefficient("1/8", None)


def test_expand_reduced_scheme() -> None:
    reduced = ReducedSchemeData(
        n=[1, 1, 2],
        m=2,
        z2=False,
        u_fresh=[],
        v_fresh=[[{"index": 0, "value": 1}, {"index": 1, "value": 1}]],
        w_fresh=[],
        u=[[{"index": 0, "value": 1}], [{"index": 0, "value": 1}]],
        v=[[{"index": 2, "value": 1}], [{"index": 1, "value": 1}]],
        w=[[{"index": 0, "value": 1}], [{"index": 1, "value": 1}]],
    )
    assert expand_reduced_scheme(reduced) == SchemeData(
        n=[1, 1, 2],
        m=2,
        z2=False,
        u=[[1], [1]],
        v=[[1, 1], [0, 1]],
        w=[[1, 0], [0, 1]],
    )


def test_expand_reduced_scheme_defaults_to_nonbinary() -> None:
    reduced = ReducedSchemeData(
        n=[1, 1, 1],
        m=1,
        u_fresh=[],
        v_fresh=[],
        w_fresh=[],
        u=[[{"index": 0, "value": 1}]],
        v=[[{"index": 0, "value": 1}]],
        w=[[{"index": 0, "value": 1}]],
    )
    assert expand_reduced_scheme(reduced)["z2"] is False
