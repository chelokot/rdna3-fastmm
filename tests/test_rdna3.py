from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from benchmarks.benchmark import packing_break_even_reuses
from rdna3_fastmm.runtime import (
    ElementTransformConfig,
    MatrixShape,
    Rank7Plan,
    Rank49Plan,
    Rank343Plan,
    WeightTransformConfig,
    unsupported_runtime_reason,
)
from rdna3_fastmm.linear import rdna3_linear, rdna3_rank7_linear


def has_tested_rdna3_runtime() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        getattr(properties, "gcnArchName", "") == "gfx1100"
        and properties.multi_processor_count == 48
    )


def configure_supported_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.4.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            gcnArchName="gfx1100",
            multi_processor_count=48,
        ),
    )
    monkeypatch.setattr(torch, "__version__", "2.9.1+rocm6.4")
    monkeypatch.setattr(triton, "__version__", "3.5.1")


def test_packing_break_even_reuses_rounds_up() -> None:
    assert packing_break_even_reuses(1.5, 2.0, 1.5) == 3


def test_packing_break_even_reuses_rejects_nonwinning_candidate() -> None:
    assert packing_break_even_reuses(1.5, 1.5, 1.5) is None
    assert packing_break_even_reuses(1.5, 1.4, 1.5) is None


def test_runtime_support_reason_accepts_tested_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_supported_runtime(monkeypatch)

    assert unsupported_runtime_reason() is None


def test_runtime_support_reason_requires_rocm_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "hip", None)

    assert unsupported_runtime_reason() == "requires a ROCm PyTorch build"


def test_runtime_support_reason_requires_available_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_supported_runtime(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert unsupported_runtime_reason() == "requires an available ROCm GPU"


def test_runtime_support_reason_rejects_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_supported_runtime(monkeypatch)

    assert unsupported_runtime_reason(torch.device("cpu")) == (
        "requires a ROCm GPU device"
    )


def test_runtime_support_reason_describes_wrong_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_supported_runtime(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            gcnArchName="gfx1101",
            multi_processor_count=40,
        ),
    )

    assert unsupported_runtime_reason() == (
        "tested only on gfx1100 with 48 compute units; found gfx1101 with 40"
    )


@pytest.mark.parametrize(
    ("component", "version", "expected"),
    (
        (
            "torch",
            "2.10.0+rocm6.4",
            "tested only with PyTorch 2.9.1; found 2.10.0+rocm6.4",
        ),
        ("hip", "6.5.0", "tested only with ROCm 6.4; found 6.5.0"),
        ("triton", "3.6.0", "tested only with Triton 3.5.1; found 3.6.0"),
    ),
)
def test_runtime_support_reason_describes_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    version: str,
    expected: str,
) -> None:
    configure_supported_runtime(monkeypatch)
    if component == "torch":
        monkeypatch.setattr(torch, "__version__", version)
    elif component == "hip":
        monkeypatch.setattr(torch.version, "hip", version)
    else:
        monkeypatch.setattr(triton, "__version__", version)

    assert unsupported_runtime_reason() == expected


def test_rank_49_shape_uses_padded_quarters() -> None:
    shape = MatrixShape.from_dimensions(257, 263, 269, 4)

    assert shape.dimensions == (257, 263, 269)
    assert (shape.block_rows, shape.block_inner, shape.block_columns) == (65, 66, 68)
    assert shape.workspace_elements(49) == 49 * (65 * 66 + 66 * 68 + 65 * 68)


def test_rank_49_shape_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        MatrixShape.from_dimensions(4, 0, 4, 4)


@pytest.mark.parametrize(
    ("shape", "has_bias"),
    (
        ((5120, 4608, 12288), False),
        ((8214, 4608, 12288), False),
        ((5632, 12288, 4608), False),
        ((19968, 4096, 16384), True),
        ((19968, 16384, 4096), True),
    ),
)
def test_rank_49_measured_linear_families_are_eligible(
    shape: tuple[int, int, int], has_bias: bool
) -> None:
    assert Rank49Plan.has_measured_linear_win(shape, has_bias=has_bias)


@pytest.mark.parametrize(
    ("shape", "has_bias"),
    (
        ((4096, 4096, 12288), False),
        ((4095, 4096, 12288), False),
        ((5119, 4608, 12288), False),
        ((9217, 4608, 12288), False),
        ((5631, 12288, 4608), False),
        ((16384, 3072, 12288), True),
        ((16384, 3072, 12288), False),
        ((19968, 4096, 16384), False),
        ((19968, 16384, 4096), False),
    ),
)
def test_rank_49_unmeasured_linear_families_are_ineligible(
    shape: tuple[int, int, int], has_bias: bool
) -> None:
    assert not Rank49Plan.has_measured_linear_win(shape, has_bias=has_bias)


@pytest.mark.parametrize(
    ("shape", "has_bias"),
    (
        ((3_328, 4_608, 12_288), False),
        ((9_216, 4_608, 12_288), False),
        ((2_048, 12_288, 4_608), False),
        ((9_216, 12_288, 4_608), False),
        ((3_600, 4_096, 12_288), False),
        ((4_096, 12_288, 4_096), False),
        ((4_992, 4_096, 16_384), True),
        ((4_992, 16_384, 4_096), True),
        ((8_192, 3_072, 12_288), True),
        ((16_384, 12_288, 3_072), True),
    ),
)
def test_rank_7_measured_linear_families_are_eligible(
    shape: tuple[int, int, int], has_bias: bool
) -> None:
    assert Rank7Plan.has_measured_linear_win(shape, has_bias=has_bias)


@pytest.mark.parametrize(
    ("shape", "has_bias"),
    (
        ((3_072, 4_608, 12_288), False),
        ((9_217, 4_608, 12_288), False),
        ((2_047, 12_288, 4_608), False),
        ((19_968, 4_096, 16_384), True),
        ((8_192, 3_072, 12_288), False),
    ),
)
def test_rank_7_unmeasured_linear_families_are_ineligible(
    shape: tuple[int, int, int], has_bias: bool
) -> None:
    assert not Rank7Plan.has_measured_linear_win(shape, has_bias=has_bias)


def test_rank_7_prepacked_weight_adds_only_measured_shape() -> None:
    shape = (720, 4_096, 16_384)

    assert not Rank7Plan.has_measured_linear_win(shape, has_bias=True)
    assert Rank7Plan.has_measured_linear_win(
        shape,
        has_bias=True,
        prepacked_weight=True,
    )
    assert not Rank7Plan.has_measured_linear_win(
        shape,
        has_bias=False,
        prepacked_weight=True,
    )


@pytest.mark.parametrize(
    ("inner", "columns", "expected"),
    (
        (3072, 12288, WeightTransformConfig(2, 1024, 4)),
        (4096, 12288, WeightTransformConfig(8, 512, 8)),
        (4608, 12288, WeightTransformConfig(2, 1024, 4)),
        (12288, 4608, WeightTransformConfig(8, 512, 8)),
        (4096, 16384, WeightTransformConfig(4, 512, 8)),
        (16384, 4096, WeightTransformConfig(4, 512, 8)),
    ),
)
def test_rank_49_selects_measured_weight_transform_config(
    inner: int, columns: int, expected: WeightTransformConfig
) -> None:
    assert Rank49Plan.weight_transform_config(inner, columns) == expected


@pytest.mark.parametrize(
    ("inner", "columns", "transform", "weight"),
    (
        (
            3_072,
            12_288,
            ElementTransformConfig(256, 2),
            WeightTransformConfig(8, 256, 4),
        ),
        (
            4_096,
            12_288,
            ElementTransformConfig(256, 2),
            WeightTransformConfig(8, 512, 4),
        ),
        (
            4_096,
            16_384,
            ElementTransformConfig(256, 2),
            WeightTransformConfig(8, 256, 4),
        ),
        (
            4_608,
            12_288,
            ElementTransformConfig(512, 4),
            WeightTransformConfig(4, 1_024, 4),
        ),
        (
            12_288,
            4_608,
            ElementTransformConfig(1_024, 4),
            WeightTransformConfig(8, 512, 4),
        ),
        (
            16_384,
            4_096,
            ElementTransformConfig(256, 2),
            WeightTransformConfig(4, 1_024, 4),
        ),
    ),
)
def test_rank_7_selects_measured_transform_config(
    inner: int,
    columns: int,
    transform: ElementTransformConfig,
    weight: WeightTransformConfig,
) -> None:
    assert Rank7Plan.transform_config(inner, columns) == transform
    assert Rank7Plan.weight_transform_config(inner, columns) == weight


def test_rank_49_plan_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="ROCm"):
        Rank49Plan(4, 4, 4, torch.device("cpu"))


def test_rank_7_plan_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="ROCm"):
        Rank7Plan(4, 4, 4, torch.device("cpu"))


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_49_dynamic_and_packed_match_exact_small_product() -> None:
    torch.manual_seed(11)
    shape = (5, 6, 7)
    device = torch.device("cuda")
    left = torch.randint(-2, 3, shape[:2], device=device, dtype=torch.float16)
    right = torch.randint(-2, 3, shape[1:], device=device, dtype=torch.float16)
    reference = (left.float() @ right.float()).half()
    plan = Rank49Plan(*shape, device=device)
    dynamic_workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    packed = plan.pack_right(right, max_free_memory_fraction=0.1)
    packed_workspace = plan.allocate_workspace(
        max_free_memory_fraction=0.1, prepacked_right=True
    )

    for _ in range(3):
        assert torch.equal(plan.run(left, right, dynamic_workspace), reference)
        assert torch.equal(plan.run_packed(left, packed, packed_workspace), reference)

    packed_result = plan.run_packed(left, packed, packed_workspace).clone()
    right.zero_()
    assert torch.equal(plan.run_packed(left, packed, packed_workspace), packed_result)
    assert plan.workspace_bytes == 3 * 49 * 2 * 2 * 2
    assert plan.prepacked_workspace_bytes == 2 * 49 * 2 * 2 * 2
    assert plan.packed_right_bytes == 49 * 2 * 2 * 2


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_49_random_nondivisible_shape_has_bounded_error() -> None:
    torch.manual_seed(7)
    shape = (257, 263, 269)
    device = torch.device("cuda")
    left = torch.randn(shape[:2], device=device, dtype=torch.float16)
    right = torch.randn(shape[1:], device=device, dtype=torch.float16)
    reference = left.float() @ right.float()
    baseline = left @ right
    plan = Rank49Plan(*shape, device=device)
    workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    candidate = plan.run(left, right, workspace)

    candidate_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    baseline_error = torch.linalg.vector_norm(baseline.float() - reference)

    assert torch.isfinite(candidate).all()
    assert candidate_error / reference_norm < 0.005
    assert baseline_error / reference_norm < 0.001


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_49_bfloat16_inputs_with_float16_leaves_have_bounded_error() -> None:
    torch.manual_seed(13)
    shape = (257, 263, 269)
    device = torch.device("cuda")
    left = torch.randn(shape[:2], device=device, dtype=torch.bfloat16)
    right = torch.randn(shape[1:], device=device, dtype=torch.bfloat16)
    reference = left.float() @ right.float()
    baseline = left @ right
    plan = Rank49Plan(
        *shape,
        device=device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    candidate = plan.run(left, right, workspace)

    candidate_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    baseline_error = torch.linalg.vector_norm(baseline.float() - reference)

    assert workspace.output_dtype == torch.bfloat16
    assert workspace.compute_dtype == torch.float16
    assert workspace.left_transformed.dtype == torch.float16
    assert workspace.right_transformed.dtype == torch.float16
    assert workspace.products.dtype == torch.float16
    assert torch.isfinite(candidate).all()
    assert candidate_error / reference_norm < 0.005
    assert baseline_error / reference_norm < 0.005


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_49_linear_accepts_native_weight_layout_and_fuses_bias() -> None:
    torch.manual_seed(19)
    shape = (257, 263, 269)
    device = torch.device("cuda")
    input_tensor = torch.randn(shape[:2], device=device, dtype=torch.bfloat16)
    weight = torch.randn((shape[2], shape[1]), device=device, dtype=torch.bfloat16)
    bias = torch.randn(shape[2], device=device, dtype=torch.bfloat16)
    reference = input_tensor.float() @ weight.float().T + bias.float()
    baseline = torch.nn.functional.linear(input_tensor, weight, bias)
    plan = Rank49Plan(
        *shape,
        device=device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    candidate = plan.run_linear(input_tensor, weight, workspace, bias)
    packed_weight = plan.pack_weight(weight, max_free_memory_fraction=0.1)
    packed_workspace = plan.allocate_workspace(
        max_free_memory_fraction=0.1,
        prepacked_right=True,
    )
    packed_candidate = plan.run_linear_packed(
        input_tensor,
        packed_weight,
        packed_workspace,
        bias,
    )

    candidate_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    baseline_error = torch.linalg.vector_norm(baseline.float() - reference)

    assert torch.isfinite(candidate).all()
    assert candidate_error / reference_norm < 0.005
    assert baseline_error / reference_norm < 0.005
    assert torch.equal(candidate, packed_candidate)
    packed_snapshot = packed_candidate.clone()
    weight.zero_()
    assert torch.equal(
        plan.run_linear_packed(
            input_tensor,
            packed_weight,
            packed_workspace,
            bias,
        ),
        packed_snapshot,
    )


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_7_linear_accepts_native_weight_layout_and_fuses_bias() -> None:
    torch.manual_seed(23)
    shape = (257, 263, 269)
    device = torch.device("cuda")
    input_tensor = torch.randn(shape[:2], device=device, dtype=torch.bfloat16)
    weight = torch.randn((shape[2], shape[1]), device=device, dtype=torch.bfloat16)
    bias = torch.randn(shape[2], device=device, dtype=torch.bfloat16)
    reference = input_tensor.float() @ weight.float().T + bias.float()
    plan = Rank7Plan(
        *shape,
        device=device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    candidate = plan.run_linear(input_tensor, weight, workspace, bias)
    packed_weight = plan.pack_weight(weight, max_free_memory_fraction=0.1)
    packed_workspace = plan.allocate_workspace(
        max_free_memory_fraction=0.1,
        prepacked_right=True,
    )
    packed_candidate = plan.run_linear_packed(
        input_tensor,
        packed_weight,
        packed_workspace,
        bias,
    )

    relative_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)

    assert torch.isfinite(candidate).all()
    assert relative_error / reference_norm < 0.005
    assert torch.equal(candidate, packed_candidate)
    assert packed_weight.source_shape == (shape[2], shape[1])
    assert packed_workspace.right_transformed is None
    assert plan.packed_weight_bytes == plan.packed_right_bytes


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_7_packed_weight_reuses_across_rows_and_rejects_other_shape() -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    weight = torch.randn((7, 6), device=device, dtype=torch.bfloat16)
    source_plan = Rank7Plan(
        5,
        6,
        7,
        device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    packed_weight = source_plan.pack_weight(
        weight,
        max_free_memory_fraction=0.1,
    )
    input_tensor = torch.randn((9, 6), device=device, dtype=torch.bfloat16)
    target_plan = Rank7Plan(
        9,
        6,
        7,
        device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    dynamic_workspace = target_plan.allocate_workspace(
        max_free_memory_fraction=0.1,
    )
    packed_workspace = target_plan.allocate_workspace(
        max_free_memory_fraction=0.1,
        prepacked_right=True,
    )

    assert torch.equal(
        target_plan.run_linear(input_tensor, weight, dynamic_workspace),
        target_plan.run_linear_packed(
            input_tensor,
            packed_weight,
            packed_workspace,
        ),
    )

    incompatible_plan = Rank7Plan(
        9,
        8,
        7,
        device,
        dtype=torch.bfloat16,
        compute_dtype=torch.float16,
    )
    incompatible_input = torch.randn((9, 8), device=device, dtype=torch.bfloat16)
    incompatible_workspace = incompatible_plan.allocate_workspace(
        max_free_memory_fraction=0.1,
        prepacked_right=True,
    )
    with pytest.raises(ValueError, match="packed Linear weight shape"):
        incompatible_plan.run_linear_packed(
            incompatible_input,
            packed_weight,
            incompatible_workspace,
        )


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_triton_linear_op_supports_leading_dimensions() -> None:
    torch.manual_seed(29)
    input_tensor = torch.randn((2, 17, 19), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((23, 19), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((23,), device="cuda", dtype=torch.bfloat16)
    reference = input_tensor.float() @ weight.float().T + bias.float()

    candidate = rdna3_linear(input_tensor, weight, bias)
    relative_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)

    assert candidate.shape == (2, 17, 23)
    assert relative_error / reference_norm < 0.005


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank7_triton_linear_op_supports_leading_dimensions() -> None:
    torch.manual_seed(31)
    input_tensor = torch.randn((2, 17, 19), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((23, 19), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((23,), device="cuda", dtype=torch.bfloat16)
    reference = input_tensor.float() @ weight.float().T + bias.float()

    candidate = rdna3_rank7_linear(input_tensor, weight, bias)
    relative_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)

    assert candidate.shape == (2, 17, 23)
    assert relative_error / reference_norm < 0.005


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_343_rejects_bfloat16_inputs() -> None:
    with pytest.raises(ValueError, match="does not support"):
        Rank343Plan(
            8,
            8,
            8,
            torch.device("cuda"),
            dtype=torch.bfloat16,
            compute_dtype=torch.float16,
        )


@pytest.mark.skipif(not has_tested_rdna3_runtime(), reason="requires tested gfx1100")
def test_rank_343_dynamic_and_packed_match_exact_small_product() -> None:
    torch.manual_seed(17)
    shape = (5, 6, 7)
    device = torch.device("cuda")
    left = torch.randint(-2, 3, shape[:2], device=device, dtype=torch.float16)
    right = torch.randint(-2, 3, shape[1:], device=device, dtype=torch.float16)
    reference = (left.float() @ right.float()).half()
    plan = Rank343Plan(*shape, device=device)
    dynamic_workspace = plan.allocate_workspace(max_free_memory_fraction=0.1)
    packed = plan.pack_right(right, max_free_memory_fraction=0.1)
    packed_workspace = plan.allocate_workspace(
        max_free_memory_fraction=0.1, prepacked_right=True
    )

    torch.testing.assert_close(
        plan.run(left, right, dynamic_workspace), reference, rtol=0, atol=0.001
    )
    torch.testing.assert_close(
        plan.run_packed(left, packed, packed_workspace),
        reference,
        rtol=0,
        atol=0.001,
    )
