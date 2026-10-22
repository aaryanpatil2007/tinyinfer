import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.zeros(out_features, 1))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        W = linear.weight.detach()
        r, c = W.shape

        scales = W.abs().amax(dim=1) / 127
        scales = torch.where(scales == 0, 1.0, scales)

        quantized = torch.round(W / scales[:, None])
        quantized = torch.clip(quantized, -127, 127)
        quantized = quantized.to(torch.int8)

        model = cls(c, r, linear.bias is not None)
        model.weight_q.copy_(quantized)
        model.scale.copy_(scales[:, None])

        if linear.bias is not None:
            model.bias.copy_(linear.bias.detach())

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_q.float() * self.scale
        return F.linear(x, W, self.bias)


class Int4Linear(nn.Module):
    GROUP_SIZE = 64

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        assert in_features % self.GROUP_SIZE == 0
        self.register_buffer("weight_packed",
                             torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale",
                             torch.zeros(out_features, in_features // self.GROUP_SIZE))
        self.register_buffer("bias", torch.zeros(out_features) if bias else None)
        self.in_features = in_features

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int4Linear":
        W = linear.weight.detach()
        out_features, in_features = W.shape
        groups = W.view(out_features, in_features // cls.GROUP_SIZE, cls.GROUP_SIZE)
        scales = groups.abs().amax(dim=2) / 7
        scales = torch.where(scales == 0, 1.0, scales)
        quantized = torch.round(groups / scales.unsqueeze(-1))
        quantized = torch.clip(quantized, -7, 7).to(torch.int8)
        q_unsigned = (quantized + 8).to(torch.uint8)
        low = q_unsigned[..., 0::2]
        high = q_unsigned[..., 1::2]
        packed = low | (high << 4)
        packed = packed.reshape(out_features, in_features // 2)

        model = cls(in_features, out_features, linear.bias is not None)
        model = model.to(device=W.device)

        model.weight_packed.copy_(packed)
        model.scale.copy_(scales)

        if linear.bias is not None:
            model.bias.copy_(linear.bias.detach())

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = (self.weight_packed & 0x0F).to(torch.int8) - 8
        high = (self.weight_packed >> 4).to(torch.int8) - 8
        q = torch.empty(
            self.weight_packed.shape[0],
            self.weight_packed.shape[1] * 2,
            dtype=torch.int8, device=self.weight_packed.device
        )
        q[:, 0::2] = low
        q[:, 1::2] = high

        q = q.reshape(self.weight_packed.shape[0], -1, self.GROUP_SIZE)

        W = q.float() * self.scale.unsqueeze(-1)
        W = W.reshape(self.weight_packed.shape[0], self.in_features)

        return F.linear(x, W, self.bias)


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    quant_class = Int4Linear if bits == 4 else Int8Linear

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head":
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        replacement = quant_class.from_linear(module)
        setattr(parent, child_name, replacement)

    return model
