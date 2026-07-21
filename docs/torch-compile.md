# Torch compile integration

The package registers an out-of-tree backend named `rdna3-fastmm` through the
`torch_dynamo_backends` package entry-point group.

```python
compiled = torch.compile(model, backend="rdna3-fastmm")
```

On the pinned PyTorch 2.9.1+ROCm 6.4 runtime, `aten.mm` lowering reads the
private `torch._inductor.config.external_matmul` list. RDNA3 FastMM appends a
module-level, versioned out-callable to that list and invokes ordinary Inductor
with GEMM autotuning enabled. Inductor wraps it as an `ExternKernelChoice` and
times it beside the available ATen, Triton, CK, and other matrix-multiplication
choices.

The external callable itself has a strict contract. Only inference-mode,
contiguous FP16 `gfx1100` inputs with a measured rank-49 shape enter FastMM.
Every other invocation calls `torch.mm(..., out=out)`. This makes an external
choice safe even though Inductor constructs it for more than the whitelisted
shapes.

## Current limits

- `external_matmul` and `torch._inductor.compile` are private PyTorch APIs. The
  backend pins and tests the exact supported runtime rather than pretending the
  hook is stable.
- The hook covers `aten.mm`, not fused `aten.addmm`.
- The external call is opaque to fusion and Inductor's memory planner. Its
  temporary workspace is allocated through PyTorch's caching allocator.
- CUDA graphs are disabled while the callable performs Python-side plan and
  workspace construction.
- Rank-343 currently wins only with an explicitly prepacked right operand; its
  lifetime and mutation semantics do not fit this dynamic `mm` hook yet.
- PyTorch 2.9.1 rejects `torch.compile` on Python 3.14. End-to-end selection must
  run under Python 3.12 or 3.13, although registry, option-merging, fallback,
  runtime, and GPU kernel tests remain testable here.

## Upstream path

A stable contribution would expose a supported external GEMM choice API or add
an RDNA3 lowering inside Inductor. That path is also required for `addmm`
epilogues, planned workspaces, transparent packed weights, CUDA-graph capture,
and broader autotune integration. The out-of-tree backend is the evidence
generator for deciding whether that engineering is justified.
