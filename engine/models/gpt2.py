"""GPT-2 forward pass — THE part you write by hand.

This file is a skeleton. Every `raise NotImplementedError` is yours to fill in.
The surrounding plumbing (weight loading, parity checking, benchmarking) is
already done, so the moment your forward pass is correct,
`python -m scripts.check_parity` will tell you.

Architecture (GPT-2 small, 124M):
    tokens -> wte + wpe -> 12 x Block -> ln_f -> lm_head -> logits
    Block:  x = x + attn(ln_1(x));  x = x + mlp(ln_2(x))     (pre-norm)

Naming contract: module/parameter names below mirror the HuggingFace GPT-2
checkpoint (wte, wpe, h, ln_1, attn.c_attn, ...). Keep them — the loader in
engine/loader.py maps HF weights onto these names, and a mismatch will fail
loudly with the exact missing key.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024   # max sequence length
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        # One fused projection for Q, K, V — HF GPT-2 stores them concatenated.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n_embd) -> (B, T, n_embd)

        TODO (yours):
          1. Project x with c_attn, split into q, k, v           (B, T, n_embd) each
          2. Reshape each to (B, n_head, T, head_dim)            head_dim = n_embd // n_head
          3. Scaled dot-product scores: q @ k^T / sqrt(head_dim) (B, n_head, T, T)
          4. Causal mask: position i may attend to j <= i only   (-inf before softmax)
          5. softmax, then @ v                                   (B, n_head, T, head_dim)
          6. Merge heads back to (B, T, n_embd), project with c_proj

        Write it with explicit ops first — you'll replace step 3-5 with your own
        fused kernel in a later phase, so you need to own every intermediate.
        """
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: c_fc -> GELU (tanh approximation! GPT-2 predates exact GELU) -> c_proj"""
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: pre-norm residual wiring (see module docstring)."""
        raise NotImplementedError


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config | None = None):
        super().__init__()
        self.config = config or GPT2Config()
        c = self.config
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)   # token embeddings
        self.wpe = nn.Embedding(c.n_positions, c.n_embd)  # learned position embeddings
        self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        # Weight tying: the LM head IS the token embedding matrix (no separate param).
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) int64 token ids -> (B, T, vocab_size) logits

        TODO: embeddings (token + position), blocks, final norm, head.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Phase 2 — naive greedy decoding: recompute the FULL sequence every
        step, append argmax of the last position's logits. Deliberately wasteful;
        it is the baseline your KV cache (phase 3) gets measured against.
        """
        raise NotImplementedError
