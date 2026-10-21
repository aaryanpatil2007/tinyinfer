"""Load real pretrained checkpoints into the hand-written models. (Plumbing — done for you.)

GPT-2: HF stores projections as Conv1D with (in, out) weights; our nn.Linear
wants (out, in), so those get transposed. Qwen2 checkpoints already use Linear
layout and our module names mirror HF's exactly, so that mapping is a prefix strip.
"""

import torch

from engine.models.gpt2 import GPT2, GPT2Config
from engine.models.qwen2 import Qwen2, Qwen2Config

GPT2_NAME = "gpt2"
QWEN2_NAME = "Qwen/Qwen2.5-0.5B"

_GPT2_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def _check_loaded(model, sd, tied_key):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != tied_key]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch — missing: {missing}, unexpected: {unexpected}")


def load_gpt2(device: str = "cpu") -> GPT2:
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(GPT2_NAME)
    model = GPT2(GPT2Config(
        vocab_size=hf.config.vocab_size,
        n_positions=hf.config.n_positions,
        n_embd=hf.config.n_embd,
        n_head=hf.config.n_head,
        n_layer=hf.config.n_layer,
    ))

    sd = {}
    for key, tensor in hf.state_dict().items():
        if not key.startswith("transformer."):
            continue  # lm_head.weight is tied to wte in our model
        key = key.removeprefix("transformer.")
        if key.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # HF's precomputed causal-mask buffers; we build the mask ourselves
        if key.endswith(_GPT2_TRANSPOSED):
            tensor = tensor.t()
        sd[key] = tensor

    _check_loaded(model, sd, tied_key="lm_head.weight")
    return model.to(device).eval()


def load_qwen2(device: str = "cpu") -> Qwen2:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(QWEN2_NAME, torch_dtype=torch.float32)
    c = hf.config
    model = Qwen2(Qwen2Config(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        n_layer=c.num_hidden_layers,
        n_head=c.num_attention_heads,
        n_kv_head=c.num_key_value_heads,
        head_dim=c.hidden_size // c.num_attention_heads,
        rope_theta=c.rope_parameters["rope_theta"],
        rms_norm_eps=c.rms_norm_eps,
    ))

    sd = {key.removeprefix("model."): tensor
          for key, tensor in hf.state_dict().items()
          if key.startswith("model.")}

    _check_loaded(model, sd, tied_key="lm_head.weight")
    return model.to(device).eval()


# name -> (our loader, HF checkpoint id)
MODELS = {
    "gpt2": (load_gpt2, GPT2_NAME),
    "qwen2": (load_qwen2, QWEN2_NAME),
}


def load_model(name: str = "gpt2", device: str = "cpu"):
    """Returns (our model, tokenizer) for a registry name."""
    loader, hf_name = MODELS[name]
    return loader(device), get_tokenizer(name)


def load_reference(name: str = "gpt2"):
    """The HF reference implementation the parity referee compares against."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        MODELS[name][1], torch_dtype=torch.float32).eval()


def get_tokenizer(name: str = "gpt2"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODELS[name][1])
