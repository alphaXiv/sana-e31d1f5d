"""Claim-level reproduction harness for Sol-Attn."""

from .core import dense_attention, exact_sparse_reference, sol_attention_reference

__all__ = ["dense_attention", "exact_sparse_reference", "sol_attention_reference"]
