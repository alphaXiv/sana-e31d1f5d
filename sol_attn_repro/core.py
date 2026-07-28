"""Paper-faithful references and fused Triton kernels for Sol-Attn.

The reference path follows Eqs. (3)--(10) of arXiv:2607.24027.  The Triton
path fuses streaming proxy routing and exact-or-approximate accumulation into
one kernel.  It never writes the N_block x N_block proxy map or routing
indices to global memory; it emits only one selected-block count per query
block for measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


@dataclass
class ThresholdStats:
    q_mean: torch.Tensor
    k_mean: torch.Tensor
    v_sum: torch.Tensor
    tau: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor


def pooled_statistics(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int, beta: float
) -> ThresholdStats:
    """Compute Eq. (5) without forming the block-proxy map."""
    n_blocks = q.shape[0] // block_size
    qb = q.reshape(n_blocks, block_size, q.shape[-1])
    kb = k.reshape(n_blocks, block_size, k.shape[-1])
    vb = v.reshape(n_blocks, block_size, v.shape[-1])
    # The paper writes the usual attention scale in Eq. (1); applying it to Q
    # keeps proxy, exact, and approximate logits in the same units.
    q_mean = qb.float().mean(dim=1) * (q.shape[-1] ** -0.5)
    k_mean = kb.mean(dim=1)
    v_sum = vb.sum(dim=1)
    key_first = k_mean.mean(dim=0)
    key_second = k_mean.T.float() @ k_mean.float() / n_blocks
    mu = q_mean.float() @ key_first.float()
    second = torch.einsum("nd,de,ne->n", q_mean.float(), key_second, q_mean.float())
    variance = (second - mu.square()).clamp_min(0)
    sigma = variance.sqrt()
    tau = mu + beta * sigma
    return ThresholdStats(q_mean, k_mean, v_sum, tau, mu, sigma)


def full_proxy_mask(stats: ThresholdStats) -> tuple[torch.Tensor, torch.Tensor]:
    proxy = stats.q_mean.float() @ stats.k_mean.float().T
    return proxy > stats.tau[:, None], proxy


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q4 = q[None, None]
    k4 = k[None, None]
    v4 = v[None, None]
    return F.scaled_dot_product_attention(q4, k4, v4).squeeze(0).squeeze(0)


def _reference_from_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
    correction: bool,
) -> torch.Tensor:
    """Small, explicit reference using one softmax state per query block."""
    n_blocks, dim = q.shape[0] // block_size, q.shape[-1]
    scale = dim**-0.5
    kb = k.reshape(n_blocks, block_size, dim)
    vb = v.reshape(n_blocks, block_size, dim)
    k_mean = kb.mean(1)
    v_sum = vb.sum(1)
    outputs = []
    for i in range(n_blocks):
        qi = q[i * block_size : (i + 1) * block_size].float() * scale
        den = torch.zeros(block_size, device=q.device)
        num = torch.zeros(block_size, dim, device=q.device)
        for j in range(n_blocks):
            if bool(mask[i, j]):
                score = qi @ kb[j].float().T
                weight = score.exp()
                den += weight.sum(1)
                num += weight @ vb[j].float()
            elif correction:
                weight = (qi @ k_mean[j].float()).exp()
                den += block_size * weight
                num += weight[:, None] * v_sum[j].float()[None, :]
        outputs.append(num / den.clamp_min(1e-20)[:, None])
    return torch.cat(outputs).to(q.dtype)


def sol_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: float, block_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, ThresholdStats]:
    stats = pooled_statistics(q, k, v, block_size, beta)
    mask, _ = full_proxy_mask(stats)
    return _reference_from_mask(q, k, v, mask, block_size, True), mask, stats


def exact_sparse_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, block_size: int = 64
) -> torch.Tensor:
    return _reference_from_mask(q, k, v, mask, block_size, False)


if HAS_TRITON:

    @triton.jit
    def _sol_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        kc_ptr,
        vc_ptr,
        tau_ptr,
        out_ptr,
        count_ptr,
        stride_qm: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_km: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vm: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_kcb: tl.constexpr,
        stride_kcd: tl.constexpr,
        stride_vcb: tl.constexpr,
        stride_vcd: tl.constexpr,
        n_blocks: tl.constexpr,
        block_size: tl.constexpr,
        dim: tl.constexpr,
        chunk_size: tl.constexpr,
        correction: tl.constexpr,
    ):
        block_i = tl.program_id(0)
        offs_m = tl.arange(0, block_size)
        offs_d = tl.arange(0, dim)
        offs_c = tl.arange(0, chunk_size)
        q = tl.load(
            q_ptr
            + (block_i * block_size + offs_m[:, None]) * stride_qm
            + offs_d[None, :] * stride_qd
        )
        score_scale = 0.125
        q_mean = tl.sum(q.to(tl.float32), axis=0) * (score_scale / float(block_size))
        row_max = tl.full((block_size,), -float("inf"), tl.float32)
        row_sum = tl.zeros((block_size,), tl.float32)
        accumulator = tl.zeros((block_size, dim), tl.float32)
        selected_count = 0
        tau = tl.load(tau_ptr + block_i)

        for chunk_start in range(0, n_blocks, chunk_size):
            block_ids = chunk_start + offs_c
            valid = block_ids < n_blocks
            kc = tl.load(
                kc_ptr + block_ids[:, None] * stride_kcb + offs_d[None, :] * stride_kcd,
                mask=valid[:, None],
                other=0.0,
            )
            score_tile = tl.dot(q, tl.trans(kc)) * score_scale
            proxy = tl.sum(q_mean[None, :] * kc.to(tl.float32), axis=1)
            selected = (proxy > tau) & valid
            selected_count += tl.sum(selected.to(tl.int32), axis=0)

            if correction:
                approx_score = tl.where(selected[None, :], -float("inf"), score_tile)
                approx_score = tl.where(valid[None, :], approx_score, -float("inf"))
                has_approx = tl.sum((valid & (selected == 0)).to(tl.int32), axis=0) > 0
                if has_approx:
                    approx_max = tl.max(approx_score, axis=1)
                    new_max_ap = tl.maximum(row_max, approx_max)
                    alpha_ap = tl.exp(row_max - new_max_ap)
                    p_approx = tl.exp(approx_score - new_max_ap[:, None])
                    vc = tl.load(
                        vc_ptr + block_ids[:, None] * stride_vcb + offs_d[None, :] * stride_vcd,
                        mask=valid[:, None],
                        other=0.0,
                    )
                    accumulator = accumulator * alpha_ap[:, None] + tl.dot(
                        p_approx.to(vc.dtype), vc
                    )
                    row_sum = row_sum * alpha_ap + float(block_size) * tl.sum(
                        p_approx, axis=1
                    )
                    row_max = new_max_ap

            for t in range(0, chunk_size):
                if selected[t]:
                    block_j = chunk_start + t
                    k = tl.load(
                        k_ptr
                        + (block_j * block_size + offs_m[:, None]) * stride_km
                        + offs_d[None, :] * stride_kd
                    )
                    exact_score = tl.dot(q, tl.trans(k)) * score_scale
                    exact_max = tl.max(exact_score, axis=1)
                    new_max_ex = tl.maximum(row_max, exact_max)
                    alpha_ex = tl.exp(row_max - new_max_ex)
                    p_exact = tl.exp(exact_score - new_max_ex[:, None])
                    v = tl.load(
                        v_ptr
                        + (block_j * block_size + offs_m[:, None]) * stride_vm
                        + offs_d[None, :] * stride_vd
                    )
                    accumulator = accumulator * alpha_ex[:, None] + tl.dot(
                        p_exact.to(v.dtype), v
                    )
                    row_sum = row_sum * alpha_ex + tl.sum(p_exact, axis=1)
                    row_max = new_max_ex

        output = accumulator / row_sum[:, None]
        tl.store(
            out_ptr
            + (block_i * block_size + offs_m[:, None]) * stride_qm
            + offs_d[None, :] * stride_qd,
            output,
        )
        tl.store(count_ptr + block_i, selected_count)


def triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float,
    block_size: int = 64,
    correction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, ThresholdStats]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if block_size != 64 or q.shape[-1] != 64:
        raise ValueError("The reproduction kernel fixes the paper's physical block and head dimension to 64")
    stats = pooled_statistics(q, k, v, block_size, beta)
    out = torch.empty_like(q)
    n_blocks = q.shape[0] // block_size
    counts = torch.empty(n_blocks, device=q.device, dtype=torch.int32)
    _sol_kernel[(n_blocks,)](
        q,
        k,
        v,
        stats.k_mean,
        stats.v_sum,
        stats.tau,
        out,
        counts,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        stats.k_mean.stride(0),
        stats.k_mean.stride(1),
        stats.v_sum.stride(0),
        stats.v_sum.stride(1),
        n_blocks=n_blocks,
        block_size=block_size,
        dim=q.shape[-1],
        chunk_size=16,
        correction=correction,
        num_warps=8,
        num_stages=2,
    )
    return out, counts, stats


def cuda_time_ms(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def incremental_peak_bytes(fn: Callable[[], object]) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return int(max(0, peak - before))


def relative_l2(observed: torch.Tensor, reference: torch.Tensor) -> float:
    return float((observed.float() - reference.float()).norm() / reference.float().norm())


def mean_row_cosine(observed: torch.Tensor, reference: torch.Tensor) -> float:
    return float(F.cosine_similarity(observed.float(), reference.float(), dim=-1).mean())


def gaussian_tail(beta: float) -> float:
    return 0.5 * math.erfc(beta / math.sqrt(2.0))
