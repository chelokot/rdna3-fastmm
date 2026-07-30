import argparse
from dataclasses import asdict
import json
from pathlib import Path

from rdna3_fastmm.slp import (
    load_and_verify_slp_certificate,
    SlpCertificate,
    SlpVerificationSummary,
)

__all__ = [
    "SlpCertificate",
    "SlpVerificationSummary",
    "load_and_verify_slp_certificate",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    arguments = parser.parse_args()
    _, summary = load_and_verify_slp_certificate(arguments.certificate)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
