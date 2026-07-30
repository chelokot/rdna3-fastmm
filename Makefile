PYTHON ?= python3
MYPY ?= mypy
PYTEST ?= pytest
RUFF ?= ruff

.PHONY: verify lint typecheck test compile certificates hashes

verify: lint typecheck test compile hashes certificates

lint:
	$(RUFF) check __init__.py src tools tests benchmarks research/prototypes
	$(RUFF) format --check __init__.py src tools tests benchmarks research/prototypes --exclude src/rdna3_fastmm/generated --exclude 'research/prototypes/generated_*.py'

typecheck:
	$(MYPY) __init__.py src/rdna3_fastmm/certificate.py benchmarks/corpus.py benchmarks/run_corpus.py tools tests

test:
	$(PYTEST)

compile:
	$(PYTHON) -m compileall -q __init__.py src benchmarks research/prototypes

hashes:
	sha256sum --check certificates/SHA256SUMS

certificates:
	$(PYTHON) -m tools.verify_reduced_scheme certificates/2x2x2_rank7_15add/certificate.json
	$(PYTHON) -m tools.verify_reduced_scheme certificates/4x4x4_rank49_159add/certificate.json
	$(PYTHON) -m tools.verify_reduced_scheme certificates/8x8x8_rank343_1661add/certificate.json
	$(PYTHON) -m tools.verify_reduced_scheme certificates/research/3x3x3_rank23_58add/certificate.json
