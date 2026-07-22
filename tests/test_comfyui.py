import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Self
import tomllib

import pytest

from rdna3_fastmm import comfyui
from rdna3_fastmm.comfyui import BackendCompiler, ComfyModel


class FakeGuardEntry:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeModel:
    def __init__(self) -> None:
        self.clone_arguments: list[bool] = []

    def clone(self, disable_dynamic: bool = False) -> Self:
        self.clone_arguments.append(disable_dynamic)
        return type(self)()


def test_guard_filter_skips_only_comfy_transformer_options() -> None:
    entries = [
        FakeGuardEntry("model.model_options['transformer_options']"),
        FakeGuardEntry("model.model_options['other']"),
        FakeGuardEntry("input_tensor"),
    ]

    assert comfyui.skip_comfy_transformer_option_guards(entries) == [
        False,
        True,
        True,
    ]


def test_compile_node_clones_and_wraps_only_diffusion_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    source_model = FakeModel()

    def backend(*args: object, **kwargs: object) -> object:
        return object()

    def set_compile_wrapper(
        *,
        model: ComfyModel,
        backend: BackendCompiler,
        options: dict[str, object],
        dynamic: bool,
        keys: list[str],
    ) -> None:
        captured.update(
            model=model,
            backend=backend,
            options=options,
            dynamic=dynamic,
            keys=keys,
        )

    monkeypatch.setattr(comfyui, "_unsupported_reason", lambda: None)
    monkeypatch.setattr(
        comfyui,
        "_load_compile_backend",
        lambda: backend,
    )
    monkeypatch.setattr(
        comfyui,
        "_load_compile_setter",
        lambda: set_compile_wrapper,
    )

    (compiled_model,) = comfyui.RDNA3FastMMCompile().compile_model(source_model)

    assert compiled_model is captured["model"]
    assert compiled_model is not source_model
    assert source_model.clone_arguments == [True]
    assert captured["backend"] is backend
    assert captured["keys"] == ["diffusion_model"]
    assert captured["options"] == {
        "guard_filter_fn": comfyui.skip_comfy_transformer_option_guards,
    }
    assert captured["dynamic"] is False


def test_compile_node_rejects_unsupported_runtime_before_cloning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = FakeModel()
    monkeypatch.setattr(
        comfyui,
        "_unsupported_reason",
        lambda: "unsupported test runtime",
    )

    with pytest.raises(
        RuntimeError,
        match="Cannot enable RDNA3 FastMM: unsupported test runtime",
    ):
        comfyui.RDNA3FastMMCompile().compile_model(source_model)

    assert source_model.clone_arguments == []


def test_compile_setter_requires_current_comfyui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_comfy_module(name: str) -> ModuleType:
        raise ModuleNotFoundError(
            f"No module named {name!r}",
            name=name,
        )

    monkeypatch.setattr(comfyui, "import_module", missing_comfy_module)

    with pytest.raises(
        RuntimeError,
        match="RDNA3 FastMM requires ComfyUI 0.19.0 or newer",
    ):
        comfyui._load_compile_setter()


@pytest.mark.parametrize("version", ((3, 11, 9), (3, 14, 0)))
def test_adapter_rejects_unsupported_python_before_importing_runtime(
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int, int],
) -> None:
    monkeypatch.setattr(sys, "version_info", version)
    monkeypatch.setattr(
        comfyui,
        "import_module",
        lambda name: pytest.fail("runtime must not load on unsupported Python"),
    )

    assert comfyui._unsupported_reason() == (
        "RDNA3 FastMM compilation requires Python 3.12 or 3.13"
    )


def test_comfyui_node_contract_is_exported() -> None:
    assert comfyui.RDNA3FastMMCompile.INPUT_TYPES() == {
        "required": {"model": ("MODEL",)},
    }
    assert comfyui.RDNA3FastMMCompile.RETURN_TYPES == ("MODEL",)
    assert comfyui.RDNA3FastMMCompile.FUNCTION == "compile_model"
    assert comfyui.RDNA3FastMMCompile.EXPERIMENTAL is True
    assert comfyui.NODE_CLASS_MAPPINGS == {
        "RDNA3FastMMCompile": comfyui.RDNA3FastMMCompile,
    }
    assert comfyui.NODE_DISPLAY_NAME_MAPPINGS == {
        "RDNA3FastMMCompile": "RDNA3 FastMM Compile",
    }


def test_comfyui_registry_metadata_matches_node_requirements() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["tool"]["comfy"] == {
        "PublisherId": "chelokot",
        "DisplayName": "RDNA3 FastMM",
        "requires-comfyui": ">=0.19.0",
    }
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["entry-points"]["torch_dynamo_backends"] == {
        "rdna3-fastmm": "rdna3_fastmm.inductor:compile_backend",
    }


def test_comfyui_archive_keeps_runtime_and_integration_docs() -> None:
    ignored = set(Path(".comfyignore").read_text().splitlines())

    assert {
        ".github/",
        "benchmarks/",
        "certificates/",
        "research/",
        "tests/",
        "tools/",
        "Makefile",
    } <= ignored
    assert {
        "src/",
        "docs/",
        "README.md",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    }.isdisjoint(ignored)


def test_custom_node_entrypoint_loads_source_package(tmp_path: Path) -> None:
    stale_package = tmp_path / "rdna3_fastmm"
    stale_package.mkdir()
    (stale_package / "__init__.py").write_text("SOURCE = 'stale install'\n")
    repository = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.util
import sys

module_name = "rdna3_fastmm_custom_node_test"
spec = importlib.util.spec_from_file_location(
    module_name,
    {str(repository / "__init__.py")!r},
    submodule_search_locations=[{str(repository)!r}],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
node = module.NODE_CLASS_MAPPINGS["RDNA3FastMMCompile"]
assert node.__module__ == "rdna3_fastmm.comfyui"
assert {str(repository / "src")!r} in sys.modules["rdna3_fastmm"].__file__
"""
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
    )


def test_custom_node_entrypoint_rejects_preloaded_conflict(tmp_path: Path) -> None:
    stale_package = tmp_path / "rdna3_fastmm"
    stale_package.mkdir()
    (stale_package / "__init__.py").write_text("SOURCE = 'stale install'\n")
    repository = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.util
import sys

import rdna3_fastmm

module_name = "rdna3_fastmm_custom_node_conflict_test"
spec = importlib.util.spec_from_file_location(
    module_name,
    {str(repository / "__init__.py")!r},
    submodule_search_locations=[{str(repository)!r}],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
try:
    spec.loader.exec_module(module)
except RuntimeError as error:
    assert "different rdna3_fastmm package is already loaded" in str(error)
else:
    raise AssertionError("preloaded conflicting package was accepted")
"""
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
