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

The measured ComfyUI path has a different operator contract: BF16
`torch.nn.functional.linear(input, weight, bias)` with native contiguous
`weight[N,K]`, optional bias, and FP16 circuit leaves. `Rank49Plan.run_linear`
implements that contract directly and `is_linear_recommended` carries its
measured dispatch policy. It cannot use `external_matmul`: biased Linear lowers
through `addmm`, and the ordinary right operand is the non-contiguous
`weight.T` view.

The compile integration performs a pre-AOT rewrite of eligible Linear nodes to
the public `rdna3_fastmm::linear` operator registered with
`torch.library.triton_op`. That preserves row-major weights, keeps unsupported
nodes in ordinary Inductor, and gives the custom operator an automatically
derived fake/meta contract. The operator body uses `wrap_triton` for all three
generated kernels and an ordinary `torch.bmm` for leaf products, so AOT and
Inductor can see its allocations and internal operations.

The FX rewrite, eager Triton operator, leading-dimension semantics, bias path,
fallback policy, and clean operator-level performance are GPU-tested. Full
Inductor execution remains pending a Python 3.12 or 3.13 ROCm environment:
PyTorch 2.9.1 rejects Dynamo on Python 3.14, and direct Inductor import also
fails in PyTorch's quantization package on Python 3.14.

## Current limits

- `external_matmul` and `torch._inductor.compile` are private PyTorch APIs. The
  backend pins and tests the exact supported runtime rather than pretending the
  hook is stable.
- The private legacy hook covers only `aten.mm`; BF16 Linear uses the separate
  public custom-operator rewrite.
- The legacy external call is opaque to fusion and Inductor's memory planner. Its
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
