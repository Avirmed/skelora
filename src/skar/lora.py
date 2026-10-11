"""Low-Rank Adaptation (LoRA) for skeleton GCN backbones.

LoRAConv2d wraps a frozen nn.Conv2d W and adds a low-rank residual path
    y = W * x + (alpha / r) * B(A * x)
where A is a conv with the same kernel/stride/dilation as W but only r output
channels, and B is a 1x1 conv (r -> C_out) initialised to zero. For 1x1
convolutions this is exactly LoRA (Hu et al., 2022); for temporal (k x 1)
convolutions it is the LoCon-style extension. After training the update can be
merged into W, so inference cost is unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        assert base.groups == 1
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        kh, kw = base.kernel_size
        self.rank = max(1, min(rank, base.out_channels, base.in_channels * kh * kw))
        self.scale = alpha / self.rank
        self.lora_down = nn.Conv2d(base.in_channels, self.rank, base.kernel_size, base.stride,
                                   base.padding, base.dilation, bias=False)
        self.lora_up = nn.Conv2d(self.rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        # conv(x, W) + conv(x, dW) == conv(x, W + dW): one convolution per step,
        # so LoRA training runs at the same speed as full fine-tuning.
        b = self.base
        return F.conv2d(x, b.weight + self._delta(), b.bias, b.stride, b.padding, b.dilation)

    def _delta(self):
        up = self.lora_up.weight.flatten(1)  # (C_out, r)
        down = self.lora_down.weight.flatten(1)  # (r, C_in*kh*kw)
        return (up @ down).view_as(self.base.weight) * self.scale

    @torch.no_grad()
    def delta_weight(self):
        return self._delta()

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True


def conv_group(name):
    """Map a CTR-GCN conv module name to its adaptation group.

    'f': spatial feature transforms of the graph conv (CTRGC conv3, residual down)
    'r': channel-wise topology relation convs (CTRGC conv1, conv2, conv4)
    't': temporal convs (multi-scale TCN branches, unit residual conv)
    """
    if '.gcn1.' in name:
        return 'r' if name.split('.')[-1] in ('conv1', 'conv2', 'conv4') else 'f'
    if '.tcn1.' in name or name.endswith('residual.conv'):
        return 't'
    return None


def inject_lora(model: nn.Module, targets: str, rank: int, alpha: float):
    """Replace target nn.Conv2d modules with LoRAConv2d.

    targets: subset of 'frt' (see conv_group); 'g' is shorthand for 'fr'.
    Returns the list of wrapped module names.
    """
    targets = targets.replace('g', 'fr')
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f'{name}.{child_name}' if name else child_name
            if isinstance(child, nn.Conv2d) and not isinstance(module, LoRAConv2d):
                if conv_group(full) in tuple(targets):
                    setattr(module, child_name, LoRAConv2d(child, rank, alpha))
                    wrapped.append(full)
    return wrapped


def merge_lora(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRAConv2d):
            m.merge()


def lora_parameters(model: nn.Module):
    for n, p in model.named_parameters():
        if 'lora_' in n:
            yield p
