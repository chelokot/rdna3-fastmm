# Experiment log

This file is append-only. Each experiment records the hypothesis, relevant
configuration, measured outcome, and dispatch decision. Raw reproducible runs
belong in `benchmarks/results/`.

## 2026-07-21

### E001 — Rank-49 square baseline: accepted narrowly

- Scheme: exact `⟨4,4,4⟩`, rank 49, 159 additions, directional costs
  `(42, 42, 75)`.
- Runtime: RX 7900 XTX `gfx1100`, PyTorch 2.9.1+ROCm 6.4, Triton 3.5.1,
  FP16 leaf products.
- Kernel configuration: 256 positions, 2 warps, 1 stage for each scalar
  transform.
- `12288³`: dynamic 33.82 ms versus 38.15 ms (`1.128×`); prepacked 31.15 ms
  (`1.225×`).
- The clean report at commit `83dbaf778d596f31f253bd69b73bad1a8d1002b3`
  measured `16384³` dynamic at 74.04 ms versus 90.79 ms (`1.226×`) and
  prepacked at 70.37 ms (`1.290×`).
- All 16 output macroblocks were sampled against CPU FP32. Candidate relative
  L2 error was 0.00250 at `16384³`; no non-finite values were observed.
- Decision: whitelist only the measured square shapes and tested runtime. Raw
  report: [`rx7900xtx-rank49-both-16384-83dbaf7.json`](../benchmarks/results/rx7900xtx-rank49-both-16384-83dbaf7.json).

### E002 — Rank-7 Strassen family: rejected for real shapes

- `8192³`: dynamic `1.046×`, prepacked `1.068×`.
- `8192×4096×11008`: dynamic `0.971×`, prepacked `1.020×`.
- `8192×11008×4096`: dynamic `0.953×`, prepacked `1.011×`.
- Taller variants reached only `1.010–1.020×` prepacked.
- Decision: margins are too small and do not survive realistic aspect ratios.

### E003 — Recursive rank-49 at a 4096 leaf size: rejected

- PyTorch: 1.348 ms; dynamic: 2.083 ms (`0.647×`); prepacked: 1.828 ms
  (`0.738×`).
- Decision: transform and launch costs dominate; do not recurse.

### E004 — Parallel U/V on two ROCm streams: rejected

- Sequential transforms: 8.364 ms; concurrent: 8.639 ms.
- Full execution: 73.053 ms sequential; 73.944 ms concurrent.
- Decision: stream overhead and resource contention outweigh overlap.

### E005 — Two-dimensional tiled transforms: rejected

- Reconstruction alone was effectively tied: 6.7940 versus 6.7979 ms.
- Full dynamic path regressed from 7.583 to 8.659 ms on the test shape.
- Decision: restore the one-dimensional scalar layout.

### E006 — MFMA reconstruction for rank 49: rejected

- The implementation was correct, but the best measured reconstruction was
  0.460 ms versus 0.303 ms for the generated scalar DAG at the 4096 test size.
- Decision: the 16-output transform is too small to amortize MFMA tiling.

### E007 — Transposed products and transform layouts: rejected

- Transposing the leaf-product layout did not recover its conversion cost and
  worsened full rank-49 execution.
- Decision: retain rank-major contiguous product planes.

### E008 — Skew rank-15, rank-45, and rank-47 schemes: rejected

- Multiple realistic `4096↔11008` orientations stayed near parity or lost to
  PyTorch after transforms.
- Decision: no dispatch entries.

### E009 — TunableOp search for 343 batched 2048³ leaves: stopped

- A rocBLAS-only TunableOp run took more than 3.5 minutes and selected the
  default implementation at approximately 57–59 ms.
- Decision: do not repeat exhaustive library tuning on this workstation; tune
  transform kernels separately and preserve desktop responsiveness.

### E010 — Rank-343 MFMA reconstruction: accepted as prepacked candidate

- Scheme: exact `⟨8,8,8⟩`, rank 343, 1661 additions.
- Replacing the scalar 343-to-64 output DAG with a padded `352×64` MFMA
  reconstruction reduced that stage from roughly 14 ms to about 5 ms.
- The clean report at commit `f5f3e889cef2b50b6de149b3e3b3f5126cb0b42e`
  measured the productionized prepacked `16384³` path at 68.60 ms against
  91.36 ms for PyTorch (`1.332×`).
- Classical-equivalent throughput was 128.22 TFLOP/s; executed leaf throughput
  was 85.89 TFLOP/s.
- CPU FP32 sampling covered all 64 output forms: relative L2 was `2.91e-3`,
  maximum absolute error 2.32, and no non-finite values were observed.
- Right packing took 10.01 ms and packing plus first execution took 78.61 ms.
- Decision: whitelist only prepacked `16384³`. Raw report:
  [`rx7900xtx-rank343-prepacked-16384-f5f3e88.json`](../benchmarks/results/rx7900xtx-rank343-prepacked-16384-f5f3e88.json).

### E011 — Rank-343 right-transform launch sweep: 512/8 selected

- `(block positions, warps)` medians at the `16384³` packing shape included:
  128/2 9.31 ms, 256/4 9.19 ms, 512/8 8.99 ms, and 1024/8 9.91 ms.
- 1024/4 regressed to 18.08 ms; smaller 64–256 position variants remained near
  9.17–9.55 ms.
- Decision: use 512 positions and 8 warps for rank-343 right prepacking.

### E012 — Inductor external callable overhead: accepted

- Clean report at commit `afdacb46e3a511443da53b05c82808e755b111a2`.
- The exact module-level callable registered through `external_matmul`, including
  plan construction and workspace allocation, measured 74.57 ms versus
  93.40 ms for PyTorch at `16384³` (`1.253×`).
- Its median was about 0.53 ms above the clean direct dynamic runtime median.
- Correctness matched the direct rank-49 path: relative L2 `2.50e-3`, maximum
  absolute error 1.27, and no non-finite values across all 16 output forms.
- Decision: integration overhead preserves the material win. End-to-end
  `torch.compile` selection remains pending a Python 3.12 or 3.13 ROCm runtime.
  Raw report: [`rx7900xtx-rank49-external-16384-afdacb4.json`](../benchmarks/results/rx7900xtx-rank49-external-16384-afdacb4.json).

### E013 — BF16 input/output with FP16 leaves: accepted for research

- Native BF16 rank-49 leaves at `257×263×269` produced relative L2 error
  `2.02e-2`, versus `1.66e-3` for PyTorch BF16.
- Converting transformed planes and leaf products to FP16 reduced candidate
  error to `2.76e-3` on the same case.
- At `4097×4096×12288`, the mixed path measured `4.72 ms` versus `3.85 ms`
  for a contiguous-right `torch.mm`; native BF16 leaves were both slower and
  approximately 12.5 times less accurate than the baseline.
- Decision: retain only BF16 input/output with FP16 transformed planes and leaf
  products. Keep it outside dispatch until tested through the real Linear
  contract.

### E014 — Contiguous-right real-shape screen: rejected as a dispatch contract

- LTX-like `3510×4096×16384` reached `1.101×` with a prepacked right operand,
  but the dynamic path reached only `0.864×`.
- Ideogram-like `16824×4608×12288` reached only `1.031×` dynamically.
- Rank-49 expands a packed right operand by `49/16 = 3.0625×`; one
  `4096×16384` BF16 weight needs approximately 392 MiB after packing.
- Real `F.linear` stores contiguous `weight[N,K]` and presents `weight.T` to
  GEMM. Materializing a contiguous transpose or caching every expanded model
  weight is not practical on a 24 GB card.
- Decision: no dispatch from these numbers. Implement a native weight-layout
  transform and benchmark the allocating `F.linear` API.

### E015 — Native Linear weight transform: 2×1024/4 selected

- The first coalesced two-dimensional kernel used a `32×32` tile and took
  approximately 3.75 ms to transform an Ideogram `weight[12288,4608]`; the full
  `8214×4608×12288` candidate reached only `0.892×`.
- Sweeps over both tile axes found `2×1024` with four warps at approximately
  1.41 ms. Large contiguous output spans matter because the transform writes
  49 planes for 16 source blocks.
- The generated kernel reads native row-major `weight[N,K]`, writes the logical
  right-transform planes directly, and never materializes `weight.T`.
- Bias is fused into the certificate-derived output reconstruction kernel.
- Decision: select `2×1024`, four warps, one stage for the measured model-width
  families.

### E016 — BF16 real-model Linear corpus: accepted selectively

- Runtime: RX 7900 XTX `gfx1100`, PyTorch 2.9.1+ROCm 6.4, Triton 3.5.1,
  BF16 inputs/outputs, FP16 leaves.
- Baseline: allocating `torch.nn.functional.linear` with native row-major
  weight and the model's real bias setting.
- Clean commit: `feadc08fd5442ac7aa1436556416579d86322f46`.

| Model path | Shape | PyTorch | Rank-49 | Speedup | Decision |
|---|---:|---:|---:|---:|---|
| HiDream-O1 MLP up | `4096×4096×12288` | 5.48 ms | 5.16 ms | `1.062×` | exact shape |
| Ideogram 4 MLP up | `8214×4608×12288` | 12.17 ms | 10.16 ms | `1.198×` | accept family |
| Ideogram 4 MLP down | `8214×12288×4608` | 14.97 ms | 12.63 ms | `1.185×` | accept family |
| LTX-2.3 second-stage up | `19968×4096×16384` | 33.46 ms | 26.70 ms | `1.253×` | exact shape |
| LTX-2.3 second-stage down | `19968×16384×4096` | 34.32 ms | 27.54 ms | `1.246×` | exact shape |
| Qwen Edit three-ref up | `16384×3072×12288` | 14.68 ms | 13.56 ms | `1.082×` | exact shape |
| Qwen Edit three-ref down | `16384×12288×3072` | 15.29 ms | 16.05 ms | `0.953×` | reject |

- Ideogram boundary sweeps accepted no-bias up projections for
  `M=5120..9216` and down projections for `M=5632..9216`. Every tested point in
  those intervals exceeded `1.09×`; `M=4704` reached only `1.03–1.04×` and is
  not dispatched.
- LTX at `M=4992` reached `0.954×` up and `0.746×` down. Qwen at approximately
  `M=8192` reached only `1.013×`. HiDream's reverse projection reached
  `0.978×`. All are rejected.
- Across accepted clean reports, sampled candidate relative L2 was
  `2.62e-3..2.95e-3`, at most 1.82 times the corresponding PyTorch error, with
  no non-finite values.
- The LTX up path peaked at approximately 5.29 GB of process GPU allocation.
  End-to-end model benefit remains unmeasured.
- Decision: expose a separate BF16 Linear recommendation gate containing only
  these measured shapes and intervals.

### E017 — Rank-343 mixed-precision real shape: rejected

- At `16824×4608×12288`, BF16 input/output with FP16 leaves measured 20.91 ms
  against 17.99 ms for PyTorch (`0.860×`).
- Process peak allocation was approximately 4.75 GB and candidate relative L2
  was `3.07e-3`.
- Decision: rank-343 remains restricted to its original FP16 prepacked
  `16384³` result.

### E018 — Shape-tuned public Triton Linear operator: accepted

- The native weight transform needs different launch geometry by `(K,N)`:
  - `(4608,12288)` and `(3072,12288)`: `2×1024`, four warps;
  - `(12288,4608)` and `(4096,12288)`: `8×512`, eight warps;
  - `(4096,16384)` and `(16384,4096)`: `4×512`, eight warps.
- On the Ideogram down orientation, isolated weight transformation fell from
  approximately 3.35 ms to 1.47 ms.
- The backend now registers `rdna3_fastmm::linear` through
  `torch.library.triton_op`, uses `wrap_triton` for generated transforms and
  reconstruction, and exposes the leaf `torch.bmm` to AOT lowering.
- A pre-AOT FX pass replaces only static, contiguous BF16 Linear nodes whose
  flattened shape, orientation, and bias semantics match the measured gate.
- Clean operator-level results at commit
  `399360532cabf86e286a3366a95224ab3f9902ec`:

| Model path | Shape | PyTorch | Triton op | Speedup |
|---|---:|---:|---:|---:|
| Ideogram up | `8214×4608×12288` | 12.28 ms | 10.18 ms | `1.207×` |
| Ideogram down | `8214×12288×4608` | 15.36 ms | 11.29 ms | `1.359×` |
| LTX-2.3 second-stage up | `19968×4096×16384` | 33.24 ms | 25.61 ms | `1.298×` |
| LTX-2.3 second-stage down | `19968×16384×4096` | 34.10 ms | 27.44 ms | `1.242×` |

- Clean Triton-op boundary results remained positive: Ideogram up measured
  `1.109×` at `M=5120` and `1.234×` at `M=9216`; down measured `1.288×` at
  `M=5632` and `1.354×` at `M=9216`.
- HiDream and Qwen had repeat minima around `1.04–1.06×`. They remain in the
  corpus and transform tuning table but were removed from the recommendation
  gate.
- Direct Inductor validation is blocked by the installed PyTorch 2.9.1 runtime
  on Python 3.14: Dynamo rejects Python 3.14, while importing the direct
  Inductor compiler reaches a Python-3.14-incompatible `typing.Union` mutation
  in PyTorch quantization code. Installing a second ROCm stack solely for this
  check would consume approximately 15 GB and was deferred.
- Decision: accept the public operator, strict FX rewrite, Ideogram intervals,
  and exact LTX shapes. Preserve ordinary Inductor fallback everywhere else.
