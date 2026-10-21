from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    n_layer: int = 24
    n_head: int = 14
    n_kv_head: int = 2
    head_dim: int = 64
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.eps)

        return self.weight * x.to(input_dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    exponents = torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim
    inv_freq = 1 / (theta ** exponents)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos().to(dtype=x.dtype)
    sin = angles.sin().to(dtype=x.dtype)

    half = head_dim // 2

    rotate_half = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotate_half * sin


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        c = config
        self.q_proj = nn.Linear(c.hidden_size, c.n_head * c.head_dim, bias=True)
        self.k_proj = nn.Linear(c.hidden_size, c.n_kv_head * c.head_dim, bias=True)
        self.v_proj = nn.Linear(c.hidden_size, c.n_kv_head * c.head_dim, bias=True)
        self.o_proj = nn.Linear(c.n_head * c.head_dim, c.hidden_size, bias=False)
        self.config = config

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                k_cache=None, v_cache=None):
        c = self.config
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, c.n_head, c.head_dim).transpose(1, 2)
        k = k.view(B, T, c.n_kv_head, c.head_dim).transpose(1, 2)
        v = v.view(B, T, c.n_kv_head, c.head_dim).transpose(1, 2)
        q = apply_rope(q, positions, c.rope_theta)
        k = apply_rope(k, positions, c.rope_theta)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
        if v_cache is not None:
            v = torch.cat([v_cache, v], dim=2)
        repeating = c.n_head // c.n_kv_head
        k_expanded = k.repeat_interleave(repeating, dim=1)
        v_expanded = v.repeat_interleave(repeating, dim=1)
        scores = (q @ k_expanded.transpose(-2, -1)) / (c.head_dim ** 0.5)

        if T > 1:
            total_length = k.shape[2]
            past_length = total_length - T

            query_positions = torch.arange(past_length, total_length, device=x.device)
            key_positions = torch.arange(total_length, device=x.device)

            mask = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v_expanded
        out = out.transpose(1, 2).contiguous().view(B, T, c.n_head * c.head_dim)

        out = self.o_proj(out)
        return out, k, v


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        c = config
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                k_cache=None, v_cache=None):
        attn_out, k, v = self.self_attn(self.input_layernorm(x), positions, k_cache, v_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class Qwen2(nn.Module):
    def __init__(self, config: Qwen2Config | None = None):
        super().__init__()
        self.config = config or Qwen2Config()
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, past_kvs=None):
        B, T = input_ids.shape

        x = self.embed_tokens(input_ids)

        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].shape[2]

        positions = torch.arange(past_len, past_len + T, device=input_ids.device)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            if past_kvs is None:
                k_cache, v_cache = None, None
            else:
                k_cache, v_cache = past_kvs[i]

            x, new_k, new_v = layer(x, positions, k_cache, v_cache)
            new_kvs.append((new_k, new_v))

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_kvs

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                 use_cache: bool = False) -> torch.Tensor:
        past_kvs = None

        for _ in range(max_new_tokens):
            if use_cache:
                if past_kvs is None:
                    logits, past_kvs = self(input_ids)
                else:
                    logits, past_kvs = self(input_ids[:, -1:], past_kvs)
            else:
                logits, _ = self(input_ids)

            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
