import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import cast

from benchmarks.corpus import DEFAULT_CORPUS_PATH, GemmCase, load_corpus


@dataclass(frozen=True)
class RunnerConfig:
    output_directory: Path
    warmups: int
    rounds: int
    tile_size: int
    max_memory_fraction: float
    timeout_seconds: int
    allow_dirty: bool


def benchmark_command(
    case: GemmCase, config: RunnerConfig, output_path: Path
) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/benchmark.py",
        "--algorithm",
        "rank49",
        "--operator",
        "linear",
        "--linear-implementation",
        "triton-op",
        "--shape",
        ",".join(str(dimension) for dimension in case.shape),
        "--dtype",
        case.input_output_dtype,
        "--compute-dtype",
        case.compute_dtype,
        "--mode",
        "dynamic",
        "--warmups",
        str(config.warmups),
        "--rounds",
        str(config.rounds),
        "--tile-size",
        str(config.tile_size),
        "--max-memory-fraction",
        str(config.max_memory_fraction),
        "--allow-unrecommended",
        "--output",
        str(output_path),
    ]
    if not case.bias:
        command.append("--no-bias")
    if config.allow_dirty:
        command.append("--allow-dirty")
    return command


def parse_arguments() -> tuple[Path, tuple[str, ...] | None, RunnerConfig | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true")
    selection.add_argument("--case", action="append")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.30)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    corpus_path = cast(Path, arguments.corpus)
    if arguments.list:
        return corpus_path, None, None
    output_directory = cast(Path | None, arguments.output_directory)
    if output_directory is None:
        parser.error("--output-directory is required when running cases")
    if arguments.warmups < 1 or arguments.rounds < 3:
        parser.error("warmups must be positive and rounds must be at least three")
    if arguments.tile_size < 1 or arguments.timeout_seconds < 1:
        parser.error("tile size and timeout must be positive")
    if not 0 < arguments.max_memory_fraction <= 1:
        parser.error("max memory fraction must be in the interval (0, 1]")
    case_ids = tuple(cast(list[str], arguments.case))
    if len(case_ids) != len(set(case_ids)):
        parser.error("case ids must not be repeated")
    return (
        corpus_path,
        case_ids,
        RunnerConfig(
            output_directory=output_directory,
            warmups=arguments.warmups,
            rounds=arguments.rounds,
            tile_size=arguments.tile_size,
            max_memory_fraction=arguments.max_memory_fraction,
            timeout_seconds=arguments.timeout_seconds,
            allow_dirty=arguments.allow_dirty,
        ),
    )


def print_cases(cases: tuple[GemmCase, ...]) -> None:
    for case in cases:
        shape = "x".join(str(dimension) for dimension in case.shape)
        bias = "bias" if case.bias else "no-bias"
        print(f"{case.id}\tpriority={case.priority}\t{shape}\t{bias}")


def main() -> None:
    corpus_path, case_ids, config = parse_arguments()
    corpus = load_corpus(corpus_path)
    if case_ids is None or config is None:
        print_cases(corpus.cases)
        return
    selected = tuple(corpus.case(case_id) for case_id in case_ids)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(selected, start=1):
        output_path = config.output_directory / f"{case.id}.json"
        print(f"[{index}/{len(selected)}] {case.id}", flush=True)
        subprocess.run(
            benchmark_command(case, config, output_path),
            check=True,
            timeout=config.timeout_seconds,
        )


if __name__ == "__main__":
    main()
