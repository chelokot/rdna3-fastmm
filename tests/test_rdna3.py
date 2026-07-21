import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from rdna3_fastmm.runtime import MatrixShape, Rank49Plan, Rank343Plan


def has_tested_rdna3_runtime() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        getattr(properties, "gcnArchName", "") == "gfx1100"
        and properties.multi_processor_count == 48
    )


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
        ((4096, 4096, 12288), False),
        ((5120, 4608, 12288), False),
        ((8214, 4608, 12288), False),
        ((5632, 12288, 4608), False),
        ((16384, 3072, 12288), True),
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
        ((4095, 4096, 12288), False),
        ((5119, 4608, 12288), False),
        ((9217, 4608, 12288), False),
        ((5631, 12288, 4608), False),
        ((16384, 3072, 12288), False),
        ((19968, 4096, 16384), False),
        ((19968, 16384, 4096), False),
    ),
)
def test_rank_49_unmeasured_linear_families_are_ineligible(
    shape: tuple[int, int, int], has_bias: bool
) -> None:
    assert not Rank49Plan.has_measured_linear_win(shape, has_bias=has_bias)


def test_rank_49_plan_rejects_cpu_device() -> None:
    with pytest.raises(ValueError, match="ROCm"):
        Rank49Plan(4, 4, 4, torch.device("cpu"))


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

    candidate_error = torch.linalg.vector_norm(candidate.float() - reference)
    reference_norm = torch.linalg.vector_norm(reference)
    baseline_error = torch.linalg.vector_norm(baseline.float() - reference)

    assert torch.isfinite(candidate).all()
    assert candidate_error / reference_norm < 0.005
    assert baseline_error / reference_norm < 0.005


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
