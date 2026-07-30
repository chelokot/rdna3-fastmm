from collections.abc import Callable, Sequence
from importlib import import_module
import logging
import sys
from typing import Protocol, Self, cast


BackendCompiler = Callable[..., object]
logger = logging.getLogger(__name__)


class ComfyModel(Protocol):
    def clone(self, disable_dynamic: bool = False) -> Self: ...


class GuardEntry(Protocol):
    name: str


class TorchCompileSetter(Protocol):
    def __call__(
        self,
        *,
        model: ComfyModel,
        backend: BackendCompiler,
        options: dict[str, object],
        dynamic: bool,
        keys: list[str],
    ) -> None: ...


class RuntimeSupportCheck(Protocol):
    def __call__(self) -> str | None: ...


def skip_comfy_transformer_option_guards(
    guard_entries: Sequence[GuardEntry],
) -> list[bool]:
    return ["transformer_options" not in entry.name for entry in guard_entries]


def _unsupported_reason() -> str | None:
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        return "RDNA3 FastMM compilation requires Python 3.12 or 3.13"
    try:
        runtime_module = import_module("rdna3_fastmm.runtime")
    except ModuleNotFoundError as error:
        return f"missing Python dependency {error.name}"
    check = cast(
        RuntimeSupportCheck,
        getattr(runtime_module, "unsupported_runtime_reason"),
    )
    return check()


def _load_compile_backend() -> BackendCompiler:
    backend_module = import_module("rdna3_fastmm.inductor")
    return cast(BackendCompiler, getattr(backend_module, "compile_backend"))


def _load_compile_setter() -> TorchCompileSetter:
    try:
        helper_module = import_module("comfy_api.torch_helpers")
    except ModuleNotFoundError as error:
        if error.name is not None and error.name.startswith("comfy_api"):
            raise RuntimeError(
                "RDNA3 FastMM requires ComfyUI 0.19.0 or newer"
            ) from error
        raise
    return cast(
        TorchCompileSetter,
        getattr(helper_module, "set_torch_compile_wrapper"),
    )


class RDNA3FastMMCompile:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "compile_model"
    CATEGORY = "model_patches/compile"
    DESCRIPTION = (
        "Compiles the diffusion model with RDNA3 FastMM. Measured BF16 Linear "
        "shapes use reduced-multiplication kernels; all other operations remain "
        "ordinary PyTorch Inductor operations."
    )
    EXPERIMENTAL = True

    def compile_model(self, model: ComfyModel) -> tuple[ComfyModel]:
        reason = _unsupported_reason()
        if reason is not None:
            raise RuntimeError(f"Cannot enable RDNA3 FastMM: {reason}")

        compile_setter = _load_compile_setter()
        compile_backend = _load_compile_backend()
        model_clone = model.clone(disable_dynamic=True)
        compile_setter(
            model=model_clone,
            backend=compile_backend,
            options={
                "guard_filter_fn": skip_comfy_transformer_option_guards,
            },
            dynamic=False,
            keys=["diffusion_model"],
        )
        logger.info("Enabled RDNA3 FastMM for the ComfyUI diffusion model")
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "RDNA3FastMMCompile": RDNA3FastMMCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RDNA3FastMMCompile": "RDNA3 FastMM Compile",
}
