import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rdna3_fastmm.certificate import (
    load_reduced_scheme,
    ReducedSchemeData,
    VerificationSummary,
    verify_reduced_scheme,
)

__all__ = [
    "ReducedSchemeData",
    "VerificationSummary",
    "load_reduced_scheme",
    "verify_reduced_scheme",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scheme", type=Path)
    arguments = parser.parse_args()
    summary = verify_reduced_scheme(load_reduced_scheme(arguments.scheme))
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
