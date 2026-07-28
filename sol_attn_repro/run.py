"""Distributed, log-complete benchmark entry point."""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path

import torch

from .core import (
    HAS_TRITON,
    cuda_time_ms,
    dense_attention,
    full_proxy_mask,
    gaussian_tail,
    incremental_peak_bytes,
    mean_row_cosine,
    pooled_statistics,
    relative_l2,
    triton_attention,
)


def emit(kind: str, payload: dict) -> None:
    print(f"{kind} {json.dumps(payload, sort_keys=True)}", flush=True)


def make_qkv(
    length: int, dim: int, family: str, seed: int, noise: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.bfloat16
    if family == "random":
        q = torch.randn(length, dim, generator=generator, device=device, dtype=dtype)
        k = torch.randn(length, dim, generator=generator, device=device, dtype=dtype)
    elif family in {"smooth", "temporal"}:
        block = 64
        n_blocks = length // block
        latent_dim = 8
        latent = torch.randn(n_blocks, latent_dim, generator=generator, device=device)
        if family == "temporal":
            position = torch.linspace(0, 6 * math.pi, n_blocks, device=device)
            latent[:, 0] += 2.0 * position.sin()
            latent[:, 1] += 2.0 * position.cos()
        projection = torch.randn(latent_dim, dim, generator=generator, device=device) / math.sqrt(latent_dim)
        centers = latent @ projection
        q = centers.repeat_interleave(block, 0)
        k = torch.roll(centers, shifts=max(1, n_blocks // 13), dims=0).repeat_interleave(block, 0)
        q = q + noise * torch.randn(length, dim, generator=generator, device=device)
        k = k + noise * torch.randn(length, dim, generator=generator, device=device)
        q, k = q.to(dtype), k.to(dtype)
    elif family == "heavy_tail":
        q = torch.randn(length, dim, generator=generator, device=device)
        k = torch.randn(length, dim, generator=generator, device=device)
        q = q / torch.rand(length, 1, generator=generator, device=device).clamp_min(0.08).sqrt()
        k = k / torch.rand(length, 1, generator=generator, device=device).clamp_min(0.08).sqrt()
        q, k = q.clamp(-8, 8).to(dtype), k.clamp(-8, 8).to(dtype)
    else:
        raise ValueError(f"Unknown family: {family}")
    v = torch.randn(length, dim, generator=generator, device=device, dtype=dtype)
    return q.contiguous(), k.contiguous(), v.contiguous()


def benchmark_condition(profile: dict, seed: int, length: int, beta: float, device: torch.device) -> dict:
    dim = int(profile["dimension"])
    block_size = int(profile["block_size"])
    repeats = int(profile["benchmark_repeats"])
    q, k, v = make_qkv(length, dim, profile["family"], seed, float(profile["noise"]), device)

    stats = pooled_statistics(q, k, v, block_size, beta)
    full_mask, proxy = full_proxy_mask(stats)
    direct_mu = proxy.mean(1)
    direct_sigma = proxy.var(1, unbiased=False).clamp_min(0).sqrt()
    threshold_max_abs = float(
        torch.maximum((direct_mu - stats.mu).abs().max(), (direct_sigma - stats.sigma).abs().max())
    )
    density_by_query = full_mask.float().mean(1)

    dense = dense_attention(q, k, v)
    sol, counts, _ = triton_attention(q, k, v, beta, block_size, correction=True)
    exact, exact_counts, _ = triton_attention(q, k, v, beta, block_size, correction=False)
    torch.cuda.synchronize()
    if not torch.equal(counts, exact_counts):
        raise AssertionError("Exact-only and corrected paths did not use the same selected blocks")

    dense_ms = cuda_time_ms(lambda: dense_attention(q, k, v), 3, repeats)
    sol_ms = cuda_time_ms(lambda: triton_attention(q, k, v, beta, block_size, True), 3, repeats)
    exact_ms = cuda_time_ms(lambda: triton_attention(q, k, v, beta, block_size, False), 3, repeats)
    dense_peak = incremental_peak_bytes(lambda: dense_attention(q, k, v))
    sol_peak = incremental_peak_bytes(lambda: triton_attention(q, k, v, beta, block_size, True))
    exact_peak = incremental_peak_bytes(lambda: triton_attention(q, k, v, beta, block_size, False))

    n_blocks = length // block_size
    proxy_map_bytes = n_blocks * n_blocks * 4
    moment_aux_bytes = (dim * dim + 2 * dim + n_blocks) * 4
    result = {
        "family": profile["family"],
        "seed": seed,
        "length": length,
        "dimension": dim,
        "block_size": block_size,
        "beta": beta,
        "target_density_gaussian": gaussian_tail(beta),
        "density_mean": float(density_by_query.mean()),
        "density_std": float(density_by_query.std(unbiased=False)),
        "density_p10": float(torch.quantile(density_by_query, 0.1)),
        "density_p90": float(torch.quantile(density_by_query, 0.9)),
        "threshold_moment_max_abs_error": threshold_max_abs,
        "stream_count_matches_full_map": bool(torch.equal(counts, full_mask.sum(1).to(torch.int32))),
        "exact_relative_l2": relative_l2(exact, dense),
        "sol_relative_l2": relative_l2(sol, dense),
        "error_reduction_fraction": 1.0 - relative_l2(sol, dense) / max(relative_l2(exact, dense), 1e-12),
        "exact_cosine": mean_row_cosine(exact, dense),
        "sol_cosine": mean_row_cosine(sol, dense),
        "dense_ms": dense_ms,
        "exact_ms": exact_ms,
        "sol_ms": sol_ms,
        "sol_speedup_over_dense": dense_ms / sol_ms,
        "exact_speedup_over_dense": dense_ms / exact_ms,
        "sol_overhead_vs_exact": sol_ms / exact_ms - 1.0,
        "dense_peak_incremental_bytes": dense_peak,
        "exact_peak_incremental_bytes": exact_peak,
        "sol_peak_incremental_bytes": sol_peak,
        "proxy_map_bytes_avoided": proxy_map_bytes,
        "moment_aux_bytes": moment_aux_bytes,
        "routing_aux_reduction": proxy_map_bytes / moment_aux_bytes,
    }
    del q, k, v, dense, sol, exact, stats, full_mask, proxy
    torch.cuda.empty_cache()
    return result


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads((Path(__file__).parent / "config.json").read_text())
    profile = config["profiles"][rank % len(config["profiles"])]
    start = time.perf_counter()
    emit(
        "ORX_RUN_INFO",
        {
            "label": config["label"],
            "rank": rank,
            "world_size": world_size,
            "profile": profile,
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton_available": HAS_TRITON,
            "python": platform.python_version(),
        },
    )
    if not HAS_TRITON:
        raise RuntimeError("The selected container does not provide Triton")

    metrics = []
    for seed in profile["seeds"]:
        for length in profile["lengths"]:
            for beta in profile["betas"]:
                result = benchmark_condition(profile, int(seed), int(length), float(beta), device)
                result["rank"] = rank
                result["label"] = config["label"]
                metrics.append(result)
                emit("ORX_METRIC", result)

    summary = {
        "label": config["label"],
        "rank": rank,
        "conditions": len(metrics),
        "family": profile["family"],
        "all_stream_counts_match": all(m["stream_count_matches_full_map"] for m in metrics),
        "max_threshold_error": max(m["threshold_moment_max_abs_error"] for m in metrics),
        "mean_error_reduction_fraction": sum(m["error_reduction_fraction"] for m in metrics) / len(metrics),
        "mean_sol_speedup_over_dense": sum(m["sol_speedup_over_dense"] for m in metrics) / len(metrics),
        "mean_sol_memory_ratio": sum(
            m["sol_peak_incremental_bytes"] / max(m["dense_peak_incremental_bytes"], 1) for m in metrics
        )
        / len(metrics),
        "elapsed_seconds": time.perf_counter() - start,
    }
    emit("ORX_SUMMARY", summary)


if __name__ == "__main__":
    main()
