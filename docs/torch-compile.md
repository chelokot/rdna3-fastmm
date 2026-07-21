# Torch compile integration

The first integration target is an out-of-tree backend registered under the
name `rdna3-fastmm`.

When `torch.compile(model, backend="rdna3-fastmm")` is invoked, the backend
receives an FX graph and example inputs. It identifies eligible inference-only
`aten.mm` nodes, replaces them with an RDNA3 FastMM custom operator, and passes
the rewritten graph to Inductor. Ineligible nodes and whole ineligible graphs
remain unchanged.

This provides one opt-in compile call and preserves Inductor fusion around the
custom operation. It does not make RDNA3 FastMM a candidate in Inductor's own
GEMM autotuning table. True competition beside ATen, Triton, CK, and other
internal choices requires an upstream or version-specific Inductor lowering.

## Planned stages

1. Register a custom operator with a fake implementation for graph tracing and
   strict runtime validation for `gfx1100` FP16 inference.
2. Register the named backend and rewrite only the dynamic rank-49 shapes that
   already beat `torch.mm`.
3. Add an explicit packed-weight API. Constant-weight graph recognition follows
   only after lifetime, mutation, stream, and concurrency semantics are proven.
4. Feed clean benchmark artifacts into the dispatch table rather than using a
   broad size heuristic.
5. Prototype an Inductor lowering and upstream it only if the out-of-tree data
   shows a useful workload family rather than isolated square sizes.

Autograd, arbitrary strides, dynamic dimensions, mixed dtypes, and silent
repacking are intentionally outside the initial contract.
