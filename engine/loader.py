"""Load real pretrained weights into the hand-written model. (Plumbing — done for you.)

HF's GPT-2 uses Conv1D layers whose weights are stored (in_features, out_features);
our nn.Linear wants (out_features, in_features), so projection weights get transposed.
Everything else is a straight name-for-name copy.
"""

import torch

from engine.models.gpt2 import GPT2, GPT2Config

# Conv1D -> Linear: these weight matrices need a transpose.
_TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2(model_name: str = "gpt2", device: str = "cpu") -> GPT2:
    """Build our GPT2 and fill it with the real HF checkpoint weights."""
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf.state_dict()

    model = GPT2(GPT2Config(
        vocab_size=hf.config.vocab_size,
        n_positions=hf.config.n_positions,
        n_embd=hf.config.n_embd,
        n_head=hf.config.n_head,
        n_layer=hf.config.n_layer,
    ))

    sd = {}
    for key, tensor in hf_sd.items():
        if not key.startswith("transformer."):
            continue  # lm_head.weight is tied to wte in our model
        key = key.removeprefix("transformer.")
        if key.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # HF's precomputed causal-mask buffers; we build the mask ourselves
        if key.endswith(_TRANSPOSED):
            tensor = tensor.t()
        sd[key] = tensor

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied, filled via wte
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch — missing: {missing}, unexpected: {unexpected}")

    return model.to(device).eval()


def get_tokenizer(model_name: str = "gpt2"):
    from transformers import GPT2TokenizerFast

    return GPT2TokenizerFast.from_pretrained(model_name)
