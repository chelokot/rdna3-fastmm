# RDNA3 FastMM

RDNA3 FastMM is an experimental BF16/FP16 matrix-multiplication backend for AMD
RDNA3 GPUs. It turns exact low-rank matrix-multiplication circuits into Triton
kernels, dispatches only on measured wins, and leaves every other operation to
PyTorch and the platform GEMM libraries.

The current backend is deliberately narrow: inference-only, contiguous tensors,
`gfx1100`, and measured large-shape families on an RX 7900 XTX. The practical
path accepts the native `torch.nn.functional.linear` contract: BF16 input,
contiguous `weight[N,K]`, optional bias, and FP16 rank-7 or rank-49 leaf
products. It is not a numerically identical replacement for PyTorch GEMM and
does not claim a win outside its measured dispatch policy.

## Real-model linear result

The ComfyUI-oriented corpus covers exact or evidence-derived shapes from
HiDream-O1, Ideogram 4, LTX-2.3, and Qwen-Image-Edit-2511. Clean results at
commit `45ad03bc89139cca2905375737f773ddf694d6b5` compare the allocating
`torch.nn.functional.linear` API against the public allocating rank-7 Triton
operator used by the compile integration.

| Workload | Shape `M×K×N` | PyTorch `F.linear` | RDNA3 FastMM | Speedup |
|---|---:|---:|---:|---:|
| HiDream-O1, MLP up | `3600×4096×12288` | 5.14 ms | 4.14 ms | **1.241×** |
| HiDream-O1, MLP up | `4096×4096×12288` | 5.69 ms | 4.41 ms | **1.289×** |
| HiDream-O1, MLP down | `4096×12288×4096` | 7.06 ms | 5.07 ms | **1.391×** |
| Ideogram 4, MLP up | `4704×4608×12288` | 6.69 ms | 5.75 ms | **1.163×** |
| Ideogram 4, MLP down | `4704×12288×4608` | 8.65 ms | 6.23 ms | **1.388×** |
| Ideogram 4 high-res, MLP up | `8214×4608×12288` | 12.17 ms | 9.52 ms | **1.278×** |
| Ideogram 4 high-res, MLP down | `8214×12288×4608` | 14.90 ms | 10.37 ms | **1.437×** |
| LTX-2.3 first stage, MLP up | `4992×4096×16384` | 7.77 ms | 6.80 ms | **1.142×** |
| Qwen Edit, MLP up | `8192×3072×12288` | 7.31 ms | 6.28 ms | **1.163×** |
| Qwen Edit, MLP up | `16384×3072×12288` | 14.55 ms | 12.41 ms | **1.173×** |
| Qwen Edit, MLP down | `16384×12288×3072` | 14.58 ms | 13.31 ms | **1.095×** |

These are individual operator medians, not end-to-end model or image-generation
speedups. Rank-7 process peak allocation was 0.70–2.34 GiB in these clean runs.
Its sampled relative L2 error was 1.04–1.15 times PyTorch BF16's error, with no
non-finite values.

The canonical reports used PyTorch's default hipBLAS preference. A later
process-isolated backend control found that hipBLASLt improves native BF16
Linear while slowing FastMM's FP16 batched leaves. Comparing the fastest backend
for each implementation narrows the Ideogram `M=8214` gains to `1.171×` up and
`1.120×` down. FastMM does not mutate PyTorch's process-global BLAS preference;
the complete control is recorded as E025 in `research/EXPERIMENTS.md`.

The measured rank-7 gate covers no-bias Ideogram up at `M=3328..9216` and down
at `M=2048..9216`, HiDream up at `M=3600..4096`, exact HiDream down at
`M=4096`, exact biased LTX up at `M=4992`, and the clean Qwen points above.
Exact biased LTX `M=19968` remains on rank-49, which was still slightly faster
than rank-7 there. LTX `M=720`, LTX `M=4992` down, and unmeasured shapes remain
ordinary PyTorch/Inductor operations. The selector also refuses a rewrite when
the complete candidate allocation exceeds half of currently free VRAM.

## Original square result

The first productionized candidate uses an exact `⟨4,4,4⟩` rank-49 circuit.
Each large multiplication is reduced from 64 to 49 leaf GEMMs, surrounded by
two input transforms and one output reconstruction.

| Shape | Mode | PyTorch `mm` | RDNA3 FastMM | Speedup |
|---|---|---:|---:|---:|
| `12288³` | dynamic | 38.15 ms | 33.82 ms | 1.128× |
| `12288³` | prepacked right | 38.15 ms | 31.15 ms | 1.225× |
| `16384³` | dynamic | 90.79 ms | 74.04 ms | 1.226× |
| `16384³` | Inductor external callable | 93.40 ms | 74.57 ms | 1.253× |
| `16384³` | prepacked right | 90.79 ms | 70.37 ms | 1.290× |
| `16384³` | rank-343 prepacked right | 91.36 ms | **68.60 ms** | **1.332×** |

These are medians from the development machine with PyTorch 2.9.1+ROCm 6.4,
Triton 3.5.1, and an RX 7900 XTX. The `16384³` prepacked result corresponds to
125.0 classical-equivalent TFLOP/s, not 125.0 physically executed TFLOP/s.
The rank-49 leaf work runs at about 95.7 TFLOP/s; transforms account for the
rest of the latency. Sampled relative L2 error against CPU FP32 was 0.00250,
versus 0.000214 for `torch.mm`.

The backend rejects common LLM shapes where it does not win. For example,
prepacked `8192×4096×11008` measured 6.96 ms against 6.83 ms for PyTorch.

The rank-343 candidate uses an MFMA output reconstruction. Its clean-tree
`16384³` report measured 128.22 classical-equivalent TFLOP/s and 85.89 executed
leaf TFLOP/s. Packing takes 10.01 ms, so packing plus the first multiply still
finishes in 78.61 ms. Its sampled relative L2 error against CPU FP32 is 0.00291.

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
about 5.8 ms and breaks even after two uses.

`Rank343Plan` exposes the faster `16384³` prepacked candidate with the same API.

The real-model path consumes a native row-major Linear weight without creating
`weight.T.contiguous()` and can fuse bias into output reconstruction. Rank-7 is
the lower-memory default for its measured families:

```python
import torch

from rdna3_fastmm.runtime import Rank7Plan

input_tensor = torch.randn((8214, 4608), device="cuda", dtype=torch.bfloat16)
weight = torch.randn((12288, 4608), device="cuda", dtype=torch.bfloat16)
plan = Rank7Plan(
    8214,
    4608,
    12288,
    input_tensor.device,
    dtype=torch.bfloat16,
    compute_dtype=torch.float16,
)
workspace = plan.allocate_workspace()
output = plan.run_linear(input_tensor, weight, workspace)
```

Call `plan.is_linear_recommended(has_bias=False)` before selecting this path.
The gate includes dtype, runtime, bias semantics, orientation, and the measured
row interval. Use `Rank49Plan` for the retained exact LTX `M=19968` path.

## PyTorch integration

The intended user surface is an opt-in compile backend:

```python
compiled = torch.compile(model, backend="rdna3-fastmm")
```

On the pinned PyTorch runtime, the backend rewrites eligible BF16 `F.linear`
nodes before AOT lowering. Measured rank-7 families use the public
`rdna3_fastmm::linear_rank7` Triton operator; the exact large LTX pair uses the
rank-49 `rdna3_fastmm::linear` operator. Their implementations expose the
transforms, `bmm`, allocations, and reconstruction to PyTorch's functional/AOT
machinery. Unsupported dtypes, layouts, bias settings, shapes, and
memory-constrained calls remain ordinary Inductor operations.

The backend also retains the legacy FP16 rank-49 callable through Inductor's
private `external_matmul` hook for the original square `aten.mm` result. See
[the integration design](docs/torch-compile.md) for the remaining private-API
boundary and upstream path.

PyTorch 2.9.1 disables `torch.compile` on Python 3.14. Use Python 3.12 or 3.13
for the compile backend; the direct runtime and benchmark harness work on the
development machine's Python 3.14 installation.

The clean legacy external-callable benchmark includes Python plan construction
and workspace allocation. It measured 74.57 ms against 93.40 ms for `torch.mm`,
only about 0.53 ms slower than the separately measured direct dynamic path.
This validates the callable's overhead, not end-to-end Inductor selection under
the unsupported Python 3.14 runtime. Both BF16 Triton operators and the FX
rewrite are GPU-tested here, but the same Python restriction also blocks final
`torch.compile` execution on this workstation.

## Reproducing the baseline

Install a matching ROCm PyTorch build and Triton first, then install this
package without replacing them:

```bash
python -m pip install -e . --no-deps
python benchmarks/benchmark.py \
  --algorithm rank49 \
  --shape 16384,16384,16384 \
  --output benchmarks/results/rank49-16384.json
```

List and run bounded real-model cases explicitly:

```bash
python -m benchmarks.run_corpus --list
python -m benchmarks.run_corpus \
  --case ideogram4-local-8214-mlp-up \
  --algorithm rank7 \
  --output-directory /tmp/rdna3-fastmm-results
```

The benchmark refuses a dirty tree or an unmeasured runtime by default. It
rotates execution order, retains raw timings, accounts for memory, and validates
a tile in every output macroblock against CPU FP32. The corpus runner requires
explicit case IDs, caps each case at 30% of free GPU memory by default, and
terminates an individual subprocess after 180 seconds.

Run all CPU-safe checks with:

```bash
python -m pip install -e '.[dev]' --no-deps
make verify
```

Every accepted and rejected experiment is recorded in
[`research/EXPERIMENTS.md`](research/EXPERIMENTS.md). Exact public circuit
certificates remain machine-verifiable and retain their upstream provenance.
