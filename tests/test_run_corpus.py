from pathlib import Path

from benchmarks.corpus import load_corpus
from benchmarks.run_corpus import benchmark_command, RunnerConfig


def test_benchmark_command_preserves_linear_contract() -> None:
    case = load_corpus().case("ideogram4-local-8214-mlp-up")
    config = RunnerConfig(
        output_directory=Path("results"),
        warmups=3,
        rounds=9,
        tile_size=8,
        max_memory_fraction=0.3,
        timeout_seconds=180,
        allow_dirty=False,
    )

    command = benchmark_command(case, config, Path("results/case.json"))

    assert command[1] == "benchmarks/benchmark.py"
    assert command[command.index("--shape") + 1] == "8214,4608,12288"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--compute-dtype") + 1] == "float16"
    assert "--no-bias" in command
    assert "--allow-dirty" not in command


def test_benchmark_command_includes_real_bias() -> None:
    case = load_corpus().case("ltx-2.3-blueprint-4992-mlp-up")
    config = RunnerConfig(
        output_directory=Path("results"),
        warmups=1,
        rounds=3,
        tile_size=8,
        max_memory_fraction=0.3,
        timeout_seconds=180,
        allow_dirty=True,
    )

    command = benchmark_command(case, config, Path("results/case.json"))

    assert "--no-bias" not in command
    assert "--allow-dirty" in command
