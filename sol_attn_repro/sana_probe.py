"""Extract per-head QKV from a released SANA-600M transformer forward prefix."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


MODEL_ID = "Efficient-Large-Model/Sana_600M_512px_diffusers"


class _CapturedForward(RuntimeError):
    pass


def _ensure_dependencies(rank: int) -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if rank == 0 and importlib.util.find_spec("diffusers") is None:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "diffusers==0.37.0",
                "transformers==4.57.3",
                "accelerate>=1.3",
                "huggingface-hub==0.36.0",
                "safetensors",
            ]
        )
    dist.barrier()


def extract_sana_qkv(profile: dict[str, Any], device: torch.device) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Run the official transformer to a chosen self-attention layer and capture its QKV."""
    rank = int(os.environ.get("RANK", "0"))
    _ensure_dependencies(rank)
    from diffusers import SanaTransformer2DModel

    layer = int(profile["layer"])
    head = int(profile["head"])
    seed = int(profile["seeds"][0])
    spatial = int(profile.get("spatial", 64))
    generator = torch.Generator(device=device).manual_seed(seed)
    model = SanaTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        variant="fp16",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    attn = model.transformer_blocks[layer].attn1
    captured: dict[str, torch.Tensor] = {}

    def capture(module, args):
        hidden_states = args[0]
        batch, sequence, _ = hidden_states.shape
        query = module.to_q(hidden_states)
        key = module.to_k(hidden_states)
        value = module.to_v(hidden_states)
        if module.norm_q is not None:
            query = module.norm_q(query)
        if module.norm_k is not None:
            key = module.norm_k(key)
        head_dim = query.shape[-1] // module.heads
        query = F.relu(query.view(batch, sequence, module.heads, head_dim))
        key = F.relu(key.view(batch, sequence, module.heads, head_dim))
        value = value.view(batch, sequence, module.heads, head_dim)
        captured["q"] = query[0, :, head].detach().contiguous()
        captured["k"] = key[0, :, head].detach().contiguous()
        captured["v"] = value[0, :, head].detach().contiguous()
        raise _CapturedForward

    hook = attn.register_forward_pre_hook(capture)
    latent = torch.randn(
        1,
        model.config.in_channels,
        spatial,
        spatial,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    encoder_hidden = torch.zeros(
        1,
        16,
        model.config.caption_channels,
        device=device,
        dtype=torch.bfloat16,
    )
    encoder_mask = torch.ones(1, 16, device=device, dtype=torch.long)
    timestep = torch.tensor([500], device=device)
    try:
        with torch.inference_mode():
            model(
                hidden_states=latent,
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                timestep=timestep,
                return_dict=False,
            )
    except _CapturedForward:
        pass
    finally:
        hook.remove()
    if set(captured) != {"q", "k", "v"}:
        raise RuntimeError(f"Failed to capture SANA QKV at layer {layer}")
    qkv = (captured["q"], captured["k"], captured["v"])
    metadata = {
        "tensor_source": "released-sana-600m-transformer",
        "model_id": MODEL_ID,
        "layer": layer,
        "head": head,
        "spatial": spatial,
        "sequence": qkv[0].shape[0],
        "head_dim": qkv[0].shape[1],
        "conditioning": "zero-embedding",
        "latent": "seeded-gaussian",
    }
    del model, latent, encoder_hidden
    torch.cuda.empty_cache()
    return qkv, metadata
