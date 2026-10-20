from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor, k_cache=None, v_cache=None):
        B, T, n_embd = x.shape
        Q, K, V = self.c_attn(x).split(self.n_embd, dim=2)
        dimsize = self.n_embd // self.n_head

        Q = Q.reshape(B, T, self.n_head, self.n_embd // self.n_head).permute(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_head, self.n_embd // self.n_head).permute(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_head, self.n_embd // self.n_head).permute(0, 2, 1, 3)
        if k_cache is not None:
            K = torch.cat([k_cache, K], dim=2)
        if v_cache is not None:
            V = torch.cat([v_cache, V], dim=2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (dimsize ** 0.5)

        if T > 1:
            past_len = K.size(2) - T
            q_positions = torch.arange(T, device=x.device) + past_len
            k_positions = torch.arange(K.size(2), device=x.device)

            mask = k_positions[None, :] > q_positions[:, None]
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1) @ V
        attn = attn.permute(0, 2, 1, 3).reshape(B, T, self.n_embd)

        return self.c_proj(attn), K, V


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inter = self.c_fc(x)
        inter = F.gelu(inter, approximate="tanh")
        result = self.c_proj(inter)
        return result


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, k_cache=None, v_cache=None):
        attn_out, K, V = self.attn(self.ln_1(x), k_cache, v_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, K, V


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config | None = None):
        super().__init__()
        self.config = config or GPT2Config()
        c = self.config
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.n_positions, c.n_embd)
        self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids: torch.Tensor, past_kvs=None):
        B, T = input_ids.shape
        tok_emb = self.wte(input_ids)
        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].size(2)
        pos = torch.arange(past_len, past_len + T, device=input_ids.device)
        pos_emb = self.wpe(pos)
        new_kvs = []
        x = tok_emb + pos_emb
        for i, block in enumerate(self.h):
            if past_kvs is None:
                k_cache = None
                v_cache = None
            else:
                k_cache, v_cache = past_kvs[i]
            x, k, v = block(x, k_cache, v_cache)
            new_kvs.append((k, v))
        x = self.ln_f(x)
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
