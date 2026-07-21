import torch

from rdna3_fastmm.runtime import is_tested_runtime, Rank49Plan


def is_rank49_dynamic_eligible(
    left: torch.Tensor, right: torch.Tensor, out: torch.Tensor
) -> bool:
    if torch.is_grad_enabled():
        return False
    if left.ndim != 2 or right.ndim != 2 or out.ndim != 2:
        return False
    rows, inner = left.shape
    right_inner, columns = right.shape
    if inner != right_inner or out.shape != (rows, columns):
        return False
    if (rows, inner, columns) not in Rank49Plan.dynamic_shapes:
        return False
    if (
        left.dtype != torch.float16
        or right.dtype != left.dtype
        or out.dtype != left.dtype
    ):
        return False
    if left.device != right.device or out.device != left.device:
        return False
    if not left.is_contiguous() or not right.is_contiguous() or not out.is_contiguous():
        return False
    if left.requires_grad or right.requires_grad or out.requires_grad:
        return False
    return left.device.type == "cuda" and is_tested_runtime(left.device)


def rdna3_rank49_dynamic_v1_out(
    left: torch.Tensor, right: torch.Tensor, *, out: torch.Tensor
) -> None:
    if not is_rank49_dynamic_eligible(left, right, out):
        torch.mm(left, right, out=out)
        return
    rows, inner = left.shape
    columns = right.shape[1]
    plan = Rank49Plan(rows, inner, columns, left.device, left.dtype)
    free_bytes, _ = torch.cuda.mem_get_info(left.device)
    if plan.workspace_bytes > free_bytes:
        torch.mm(left, right, out=out)
        return
    workspace = plan.allocate_workspace(max_free_memory_fraction=1.0)
    plan.run(left, right, workspace, out)
