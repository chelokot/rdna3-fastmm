# RDNA3 FastMM

RDNA3 FastMM is an experimental FP16 matrix-multiplication backend for AMD
RDNA3 GPUs. It turns exact low-rank matrix-multiplication circuits into Triton
kernels, dispatches only on measured wins, and leaves every other operation to
PyTorch Inductor and the platform GEMM libraries.

The current backend is deliberately narrow: inference-only, contiguous FP16,
`gfx1100`, and a small whitelist of large shapes measured on an RX 7900 XTX.
It is not a numerically identical replacement for `torch.mm`, and it does not
claim a universal GEMM win.

## Current result

The first productionized candidate uses an exact `⟨4,4,4⟩` rank-49 circuit.
Each large multiplication is reduced from 64 to 49 leaf GEMMs, surrounded by
two input transforms and one output reconstruction.

| Shape | Mode | PyTorch `mm` | RDNA3 FastMM | Speedup |
|---|---|---:|---:|---:|
| `12288³` | dynamic | 38.15 ms | 33.82 ms | 1.128× |
| `12288³` | prepacked right | 38.15 ms | 31.15 ms | 1.225× |
| `16384³` | dynamic | 91.19 ms | 74.96 ms | 1.216× |
| `16384³` | prepacked right | 91.19 ms | 70.58 ms | 1.292× |

These are medians from the development machine with PyTorch 2.9.1+ROCm 6.4,
Triton 3.5.1, and an RX 7900 XTX. The `16384³` prepacked result corresponds to
124.6 classical-equivalent TFLOP/s, not 124.6 physically executed TFLOP/s.
The rank-49 leaf work runs at about 95.4 TFLOP/s; transforms account for the
rest of the latency. Sampled relative L2 error against CPU FP32 was 0.00250,
versus 0.000214 for `torch.mm`.

The backend rejects common LLM shapes where it does not win. For example,
prepacked `8192×4096×11008` measured 6.96 ms against 6.83 ms for PyTorch.

## Direct runtime

```python
import torch

from rdna3_fastmm.runtime import Rank49Plan

left = torch.randn((16384, 16384), device="cuda", dtype=torch.float16)
right = torch.randn_like(left)
plan = Rank49Plan(16384, 16384, 16384, left.device)
workspace = plan.allocate_workspace(prepacked_right=True)
packed_right = plan.pack_right(right)
output = plan.run_packed(left, packed_right, workspace)
```

`Rank49Plan.is_recommended()` is the performance gate. Callers should fall back
to `torch.mm` whenever it returns false. Packing a `16384²` right operand costs
about 5.7 ms and breaks even after two uses.

## PyTorch integration

The intended user surface is an opt-in compile backend:

```python
compiled = torch.compile(model, backend="rdna3-fastmm")
```

Its graph pass replaces only proven-profitable `aten.mm` nodes and compiles the
remaining graph with standard Inductor. This is an out-of-tree backend, not yet
an entry in Inductor's internal GEMM autotuner. See [the integration design](docs/torch-compile.md)
for the boundary and upstream path.

## Reproducing the baseline

Install a matching ROCm PyTorch build and Triton first, then install this
package without replacing them:

```bash
python -m pip install -e . --no-deps
python benchmarks/benchmark_rank49.py \
  --shape 16384,16384,16384 \
  --output benchmarks/results/rank49-16384.json
```

The benchmark refuses a dirty tree or an unmeasured runtime by default. It
rotates execution order, retains raw timings, accounts for packing and memory,
and validates a tile in every output macroblock against CPU FP32.

Run all CPU-safe checks with:

```bash
python -m pip install -e '.[dev]' --no-deps
make verify
```

Every accepted and rejected experiment is recorded in
[`research/EXPERIMENTS.md`](research/EXPERIMENTS.md). Exact public circuit
certificates remain machine-verifiable and retain their upstream provenance.
