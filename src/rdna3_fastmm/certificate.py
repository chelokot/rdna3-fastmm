from dataclasses import dataclass
from pathlib import Path

from tools.export_fmm_git_scheme import expand_linear_map
from tools.verify_reduced_scheme import load_reduced_scheme, verify_reduced_scheme


@dataclass(frozen=True)
class ExpandedScheme:
    dimensions: tuple[int, int, int]
    rank: int
    left_coefficients: tuple[tuple[int, ...], ...]
    right_coefficients: tuple[tuple[int, ...], ...]
    output_coefficients: tuple[tuple[int, ...], ...]


def load_expanded_scheme(path: Path) -> ExpandedScheme:
    source = load_reduced_scheme(path)
    summary = verify_reduced_scheme(source)
    first_size, shared_size, second_size = summary.dimensions
    rank = summary.rank
    left_coefficients = expand_linear_map(
        first_size * shared_size, source["u_fresh"], source["u"]
    )
    right_coefficients = expand_linear_map(
        shared_size * second_size, source["v_fresh"], source["v"]
    )
    output_columns = expand_linear_map(rank, source["w_fresh"], source["w"])
    output_coefficients = [
        output_columns[column * first_size + row]
        for row in range(first_size)
        for column in range(second_size)
    ]
    return ExpandedScheme(
        dimensions=(first_size, shared_size, second_size),
        rank=rank,
        left_coefficients=tuple(map(tuple, left_coefficients)),
        right_coefficients=tuple(map(tuple, right_coefficients)),
        output_coefficients=tuple(map(tuple, output_coefficients)),
    )
