"""Check that the batched GPU augmentation matches CTR-GCN's per-sample feeder."""
import sys, os.path as osp
import numpy as np, torch
ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(ROOT, 'src')); sys.path.insert(0, osp.join(ROOT, 'third_party', 'CTR-GCN'))
from skar.data import load_meta, split_indices, NTUData, crop_resize, rotation_matrices, to_model_input
from feeders import tools

meta = load_meta()
_, te, _ = split_indices(meta, 'ntu60_xsub')
d = NTUData(meta, te[:16])
x, lens, y = d.batch(np.arange(16))
xg = torch.from_numpy(x).cuda()

# Test-mode crop/resize vs tools.valid_crop_resize (p_interval=[0.95]).
ours = to_model_input(xg, torch.from_numpy(lens), False, 'joint').cpu().numpy()
err = 0
for k in range(16):
    full = np.zeros((300, 150), np.float32); full[:lens[k]] = x[k, :lens[k]]
    cv = full.reshape(300, 2, 25, 3).transpose(3, 0, 2, 1)  # C,T,V,M
    valid = np.sum(cv.sum(0).sum(-1).sum(-1) != 0)
    ref = tools.valid_crop_resize(cv, valid, [0.95], 64)
    err = max(err, np.abs(ref - ours[k]).max())
print('crop/resize max abs err:', err)

# Rotation vs tools._rot.
ang = torch.rand(5, 3) * 0.6 - 0.3
R_ours = rotation_matrices(ang)
R_ref = torch.stack([tools._rot(a[None])[0] for a in ang])
print('rotation max abs err:', (R_ours - R_ref).abs().max().item())

# Train-mode with a fixed crop: compare against the reference using the same p and bias.
class FixedRng:
    def __init__(s, p, b): s.p, s.b = p, b
    def random(s, n): return (s.p - 0.5) / 0.5  # crop_resize maps U[0,1) to [0.5,1)
    def integers(s, lo, hi): return s.b.pop(0)
lens_np = lens.copy()
p = np.full(16, 0.7)
crop = np.minimum(np.maximum(np.floor(lens_np * p).astype(int), 64), lens_np)
bias = [int((v - c) // 2) for v, c in zip(lens_np, crop)]
out = crop_resize(xg, torch.from_numpy(lens), True, rng=FixedRng(p, list(bias))).cpu().numpy()
err = 0
for k in range(16):
    seg = x[k, bias[k]:bias[k] + crop[k]]
    t = torch.from_numpy(seg.T.copy())[None, None]
    ref = torch.nn.functional.interpolate(t, size=(150, 64), mode='bilinear', align_corners=False)[0, 0].T.numpy()
    err = max(err, np.abs(ref - out[k]).max())
print('train crop/resize max abs err:', err)
