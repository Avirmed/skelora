"""LoRAConv2d: merged-weight forward equals base(x) + up(down(x)) * scale, also for strided/dilated temporal convs."""
import sys, os.path as osp
import torch, torch.nn as nn
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from skar.lora import LoRAConv2d
torch.manual_seed(0)
for conv in [nn.Conv2d(64, 32, 1), nn.Conv2d(32, 32, (5, 1), padding=(4, 0), stride=(2, 1), dilation=(2, 1)),
             nn.Conv2d(3, 64, 1)]:
    m = LoRAConv2d(conv, rank=8, alpha=8)
    nn.init.normal_(m.lora_up.weight)
    x = torch.randn(2, conv.in_channels, 64, 25)
    ref = conv(x) + m.lora_up(m.lora_down(x)) * m.scale
    out = m(x)
    m.merge()
    print(tuple(conv.weight.shape), 'rank', m.rank, 'max err', (ref - out).abs().max().item(),
          'merged err', (ref - m(x)).abs().max().item())
