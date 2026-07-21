# Architecture

RDNA3 FastMM separates mathematical schemes, generated kernels, runtime policy,
PyTorch graph integration, and evidence. A faster prototype is not dispatchable
until it has passed every layer.

## Layers

1. `certificates/` contains exact reduced straight-line programs and upstream
   provenance. `tools/verify_reduced_scheme.py` expands them over the integers
   and checks the Brent tensor identity.
2. `tools/generate_triton_scheme.py` deterministically emits the transform
   kernels checked into `src/rdna3_fastmm/generated/`.
3. `src/rdna3_fastmm/runtime.py` owns device guards, shape policy, workspace
   allocation, prepacking, alias checks, and kernel launch configuration.
4. The compile backend rewrites eligible graph nodes to an opaque custom
   operator and sends the rest of the graph to standard Inductor.
5. `benchmarks/` produces machine-readable reports. A dispatch whitelist is
   changed only from clean-tree reports on the tested runtime.
6. `research/EXPERIMENTS.md` is append-only evidence for successful and failed
   hypotheses.

## Execution path

For a rank-49 `⟨4,4,4⟩` plan, the input matrices are divided into four blocks
along each dimension. Generated linear transforms produce 49 left and 49 right
planes. `torch.bmm` computes the 49 leaf products, and a generated transform
reconstructs the 16 output blocks directly into row-major output storage.

Prepacked execution materializes the right transform once. Dynamic execution
recomputes both transforms for each call. Workspaces and packed operands are
explicit so allocation is never included in kernel timings.

## Dispatch contract

A candidate must match all of the following before automatic selection:

- tested GPU architecture and runtime versions;
- exact dtype, layout, dimensions, and inference-only semantics;
- a clean benchmark report with a material speedup after all transforms;
- bounded error against FP32 across every circuit output form;
- an explicit memory budget;
- no storage overlap between inputs, outputs, workspaces, or packed operands.

Everything else uses the unmodified PyTorch/Inductor path. This conservative
contract is part of the optimization, not temporary scaffolding.
