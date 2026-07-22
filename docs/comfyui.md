# ComfyUI integration

## Workflow contract

`RDNA3 FastMM Compile` is a model-patch node with one `MODEL` input and one
`MODEL` output. Place it after every node that changes model weights or objects
and before the sampler:

```text
Load Diffusion Model → Apply LoRA / model patches → RDNA3 FastMM Compile → KSampler
```

The node uses ComfyUI's public `set_torch_compile_wrapper` helper. It clones the
input with `disable_dynamic=True`, compiles only the `diffusion_model`, and
installs the result through ComfyUI's keyed `APPLY_MODEL` wrapper. ComfyUI keeps
ownership of model loading and offloading.

The compile configuration is deliberately fixed for the first release:

```python
set_torch_compile_wrapper(
    model=model_clone,
    backend=compile_backend,
    options={"guard_filter_fn": skip_comfy_transformer_option_guards},
    dynamic=False,
    keys=["diffusion_model"],
)
```

Static graphs are required because FastMM's dispatch policy is built from
measured concrete matrix dimensions. A symbolic row dimension cannot safely
select a measured family. Different resolutions may therefore produce distinct
compiled graphs.

## Dispatch and fallback

On the first model execution, Dynamo captures the diffusion graph and invokes
the RDNA3 FastMM backend. The backend rewrites eligible BF16 Linear calls to the
public rank-7 or rank-49 Triton operators before ordinary Inductor lowering.
Unmeasured shapes, unsupported bias and layout combinations, other dtypes, and
calls that exceed the free-memory policy remain ordinary Inductor operations.

The node never changes PyTorch's process-global BLAS preference. It also does
not prepack or retain model weights. This avoids stale data when ComfyUI clones,
patches, unloads, or moves a model. A future packed-weight cache must be owned by
the `ModelPatcher` lifecycle and keyed by its patch identity, device, dtype, and
shape.

Only one ComfyUI compile wrapper can be active on a model. Applying the generic
`TorchCompileModel` node after RDNA3 FastMM, or applying RDNA3 FastMM after that
node, replaces the earlier compile wrapper. Other non-compile ModelPatcher
wrappers remain intact.

## Compatibility

- ComfyUI 0.19.0 or newer
- Python 3.12 or 3.13
- PyTorch 2.9.1 for ROCm 6.4
- Triton 3.5.1
- `gfx1100` with 48 compute units, currently the measured RX 7900 XTX target

The custom-node entrypoint loads the repository's local `src/` package and
passes its compiler callable directly. Normal Python package installations can
also select the identical backend by its registered `rdna3-fastmm` name. The
entrypoint refuses a previously imported conflicting `rdna3_fastmm` package.

FP8 or quantized weights do not satisfy the current BF16 selector and therefore
receive no FastMM rewrite. Compilation can also change host and device memory
behavior for workflows that rely on aggressive offloading; those configurations
need end-to-end validation before being advertised as supported. In particular,
`clone(disable_dynamic=True)` converts ComfyUI's dynamic patcher to a
non-dynamic delegate, and a custom loader without the required cached factory
can reject that conversion.

## Validation status

The node contract, clone semantics, compile configuration, guard filter,
runtime rejection, registry metadata, and repository entrypoint are covered by
CPU-safe tests. The custom Linear operators and FX rewrite are separately tested
on the development RX 7900 XTX.

The development ComfyUI environment currently uses Python 3.14, on which
PyTorch 2.9.1 disables `torch.compile`. Full ComfyUI sampling validation remains
pending a Python 3.12 or 3.13 environment; the repository does not claim an
end-to-end image-generation speedup yet.
