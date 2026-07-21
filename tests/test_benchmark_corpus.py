import json
from pathlib import Path

import pytest

from benchmarks.corpus import DEFAULT_CORPUS_PATH, load_corpus


def test_real_model_corpus_has_unique_bfloat16_linear_cases() -> None:
    corpus = load_corpus()

    assert corpus.schema_version == 1
    assert corpus.operator == "torch.nn.functional.linear"
    assert len(corpus.cases) >= 10
    assert len({case.id for case in corpus.cases}) == len(corpus.cases)
    assert {case.model for case in corpus.cases} >= {
        "HiDream-O1",
        "Ideogram 4",
        "LTX-2.3",
        "Qwen-Image-Edit-2511",
    }
    assert all(case.input_output_dtype == "bfloat16" for case in corpus.cases)
    assert all(case.compute_dtype == "float16" for case in corpus.cases)
    assert all(case.weight_layout == "out_in" for case in corpus.cases)
    assert all(min(case.shape) > 0 for case in corpus.cases)


def test_real_model_corpus_preserves_orientation() -> None:
    corpus = load_corpus()

    up = corpus.case("ltx-2.3-blueprint-4992-mlp-up")
    down = corpus.case("ltx-2.3-blueprint-4992-mlp-down")

    assert up.shape == (4992, 4096, 16384)
    assert down.shape == (4992, 16384, 4096)


def test_corpus_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CORPUS_PATH.read_text())
    raw["cases"].append(raw["cases"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="unique"):
        load_corpus(path)
