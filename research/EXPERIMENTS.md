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

### E019 — Rank-49 product/reconstruction fusion: rejected

- A generator opt-in emits two research-only alternatives while leaving the
  production rank-49 module byte-for-byte unchanged:
  - one program serially computes all 49 leaf products and reconstructs 16
    output blocks;
  - 49 independent product programs atomically accumulate directly into an
    FP32 output buffer, followed by a bias/cast finalizer.
- At `5120×4608×12288`, the existing `torch.bmm` plus reconstruction stage
  took approximately 4.71–4.95 ms.

| Alternative | Best measured configuration | Candidate stage | Full Linear | PyTorch Linear |
|---|---|---:|---:|---:|
| Serial fusion | `64×128×32`, eight warps | 18.62 ms | 20.45 ms | 7.85 ms |
| Atomic fusion | `128×128×32`, eight warps | 15.29 ms | 16.39 ms | 8.67 ms |

- The best serial kernel used 256 VGPRs and spilled 776 registers, including
  1664 bytes of scratch per thread. Smaller non-spilling tiles were much
  slower because each program still executes all 49 matrix products.
- Atomic fusion performs 196 coefficient-weighted updates across 16 outputs,
  or 12.25 FP32 atomics per output element on average. At this shape that is
  approximately 2.53 billion contended atomic updates.
- Serial fusion reduced estimated candidate workspace from 876,675,072 bytes
  to 491,323,392 bytes. Atomic fusion used approximately 742,981,632 bytes.
  Both retained bounded small-shape error, but the memory saving did not
  compensate for the latency regression.
- Decision: keep the generated alternatives as reproducible research
  artifacts and do not expose either through runtime dispatch.

### E020 — Rank-7 `2×2×2` Linear prototype: accepted for public-op validation

- A verified 15-addition rank-7 certificate reduces the transformed batch from
  49 quarter-size products to seven half-size products. Relative to ordinary
  GEMM this executes `7/8 = 87.5%` of the scalar multiplications and expands a
  transformed weight by `7/4 = 1.75×`, versus `49/16 = 3.0625×` for rank-49.
- The benchmark uses native contiguous BF16 `weight[N,K]`, FP16 transformed
  planes and leaf products, BF16 output, optional fused bias, sampled FP32
  correctness, and explicit ordinary-FP16 controls with matching conversions.
- Current numbers are plan-level: rank-7 reuses preallocated transformed,
  product, and output buffers while native and FP16 controls allocate outputs.
  They therefore justify an allocation-fair public operator benchmark, not
  dispatch by themselves.

| Model path | Shape | PyTorch BF16 | FP16 control | Rank-7 dynamic | Rank-7 prepacked | Dynamic speedup |
|---|---:|---:|---:|---:|---:|---:|
| Ideogram up | `5120×4608×12288` | 7.25 ms | 8.58 ms | 5.87 ms | 5.22 ms | `1.235×` |
| Ideogram up | `8214×4608×12288` | 12.75 ms | 12.89 ms | 9.59 ms | 8.66 ms | `1.329×` |
| Ideogram down | `8214×12288×4608` | 15.54 ms | 16.95 ms | 10.08 ms | 9.46 ms | `1.542×` |
| Ideogram up | `4704×4608×12288` | 7.07 ms | — | 5.79 ms | 4.85 ms | `1.220×` |
| Ideogram up | `4056×4608×12288` | 6.07 ms | — | 4.92 ms | 4.07 ms | `1.233×` |
| LTX-2.3 up | `19968×4096×16384` | 33.83 ms | 37.86 ms | 25.70 ms | — | `1.316×` |
| LTX-2.3 down | `19968×16384×4096` | 35.26 ms | 37.91 ms | 27.57 ms | — | `1.279×` |

- The tuned Ideogram transform configurations were `512` elements/four warps
  with an `8×512`/eight-warp weight kernel for the up projection, and `1024`
  elements/four warps with an `8×256`/eight-warp weight kernel for down.
- Sampled rank-7 relative L2 error was only `1.05–1.09×` the corresponding
  native BF16 error, materially better than the accepted rank-49 path. No
  non-finite values were observed.
- On Ideogram, rank-7 was approximately 6% faster than the clean public
  rank-49 up result and 12% faster down before allocation-fair validation. On
  the exact LTX shapes it was effectively tied with rank-49, so rank-49 remains
  the latency candidate there.
- Decision: implement rank-7 as an allocating `torch.library.triton_op`, repeat
  clean alternating benchmarks across dense family boundaries, and change the
  FX recommendation gate only for shapes that retain a repeated measured win.

### E021 — Allocating rank-7 Triton operator: accepted selectively

- Clean commit: `45ad03bc89139cca2905375737f773ddf694d6b5`.
- Baseline and candidate both use their public allocating APIs. The benchmark
  alternates execution order, records nine timed samples, validates a tile in
  every output macroblock against CPU FP32, and includes the replaced output in
  its peak-memory estimate.
- PyTorch 2.9.1's `triton_op` source scanner recognizes only direct
  `wrap_triton(simple_name)` calls. Both rank-7 and rank-49 operators now expose
  all four possible inner kernels to AOT cache hashing instead of hiding them
  behind a helper or module-qualified attribute.

| Model path | Shape | Bias | PyTorch | Rank-7 op | Speedup | Error ratio |
|---|---:|:---:|---:|---:|---:|---:|
| HiDream up | `3600×4096×12288` | no | 5.14 ms | 4.14 ms | `1.241×` | `1.088×` |
| HiDream up | `4096×4096×12288` | no | 5.69 ms | 4.41 ms | `1.289×` | `1.053×` |
| HiDream down | `4096×12288×4096` | no | 7.06 ms | 5.07 ms | `1.391×` | `1.094×` |
| Ideogram up | `4704×4608×12288` | no | 6.69 ms | 5.75 ms | `1.163×` | `1.066×` |
| Ideogram down | `4704×12288×4608` | no | 8.65 ms | 6.23 ms | `1.388×` | `1.107×` |
| Ideogram up | `8214×4608×12288` | no | 12.17 ms | 9.52 ms | `1.278×` | `1.137×` |
| Ideogram down | `8214×12288×4608` | no | 14.90 ms | 10.37 ms | `1.437×` | `1.055×` |
| LTX first-stage up | `4992×4096×16384` | yes | 7.77 ms | 6.80 ms | `1.142×` | `1.150×` |
| Qwen up | `8192×3072×12288` | yes | 7.31 ms | 6.28 ms | `1.163×` | `1.100×` |
| Qwen up | `16384×3072×12288` | yes | 14.55 ms | 12.41 ms | `1.173×` | `1.044×` |
| Qwen down | `16384×12288×3072` | yes | 14.58 ms | 13.31 ms | `1.095×` | `1.104×` |

- Dense Ideogram screens retained no-bias up for `M=3328..9216` and down for
  `M=2048..9216`. The accepted HiDream interval is `M=3600..4096` up plus exact
  `M=4096` down. LTX and Qwen gates are exact biased shapes from the table.
- LTX `M=720`, LTX `M=4992` down, and rank-7 LTX `M=19968` were rejected. The
  last remained slightly slower than the existing rank-49 operator, so the
  selector retains rank-49 for the exact second-stage LTX pair.
- Candidate-to-baseline sampled error ratios were `1.044×..1.150×`; no
  non-finite values were observed. Rank-7 process peak allocation was
  0.70–2.34 GiB across the accepted clean reports, compared with approximately
  4.93 GiB for the retained rank-49 LTX up control in the current tree.
- Automatic selection prefers rank-7 for its measured families, then rank-49
  for the remaining exact LTX pair, then ordinary Inductor. A conservative
  runtime guard refuses the rewrite when candidate workspace plus output would
  consume more than half of currently free device memory.
- Decision: accept the public `rdna3_fastmm::linear_rank7` operator and only the
  measured shape families above. Preserve rank-49 and native fallbacks. Raw
  reports are indexed in [`benchmarks/results/README.md`](../benchmarks/results/README.md).

### E022 — Sparse rank-7 product/reconstruction fusion: rejected

- The certificate-derived research kernel hard-codes the 14 nonzero rank-7
  reconstruction updates. It computes seven FP16 `tl.dot` products serially,
  retains four output tiles in FP32, fuses optional bias, and stores BF16
  without materializing the rank-major product tensor.
- Full odd-shape correctness passed at `257×263×269` with and without bias.
  Avoiding the intermediate FP16 product rounding slightly improved sampled
  error versus the current rank-7 operator.
- At Ideogram `8214×4608×12288`, fusion removed a 336.9 MiB product tensor
  and reduced candidate workspace from 652.2 MiB to 315.3 MiB.

| Tile `BM×BN×BK` | Warps | Fused stage | Current stage | Stage ratio |
|---:|---:|---:|---:|---:|
| `16×64×32` | 4 | 93.18 ms | 8.68 ms | `0.093×` |
| `32×64×32` | 4 | 42.61 ms | 8.51 ms | `0.200×` |
| `32×64×64` | 4 | 51.10 ms | 8.38 ms | `0.164×` |
| `32×128×32` | 8 | 46.80 ms | 8.77 ms | `0.187×` |
| `64×64×32` | 8 | 30.06 ms | 8.42 ms | `0.280×` |
| `64×128×32` | 8 | 26.09 ms | 8.50 ms | `0.326×` |

- The best full candidate took 23.93 ms, versus 9.76 ms for the current rank-7
  operator and 12.27 ms for PyTorch. Its sampled relative L2 was `1.81e-3`,
  versus `1.95e-3` for current rank-7 and `1.76e-3` for PyTorch.
- Register spilling was not the primary failure. `64×128×32` used 256 VGPRs
  and 204 bytes of private storage, but non-spilling `64×64×32` still took
  30.06 ms with 175 VGPRs, and `16×64×32` used only 104 VGPRs while taking
  93.18 ms. Serializing seven GEMMs inside one Triton program loses too much
  leaf throughput relative to the tuned batched library GEMM.
- Decision: retain the generator and benchmark as reproducible negative
  evidence, but do not test more shapes or expose the kernel to dispatch.
