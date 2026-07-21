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
- `16384³`: dynamic 74.96 ms versus 91.19 ms (`1.216×`); prepacked 70.58 ms
  (`1.292×`).
- All 16 output macroblocks were sampled against CPU FP32. Candidate relative
  L2 error was 0.00250 at `16384³`; no non-finite values were observed.
- Decision: whitelist only the measured square shapes and tested runtime.

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
