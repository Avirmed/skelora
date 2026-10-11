"""NTU RGB+D 60/120 data pipeline with GPU-side augmentation.

Samples are stored as variable-length sequences (see src/prep/prepare_ntu.py).
A background thread assembles padded CPU batches; temporal crop + resize,
random rotation and modality conversion (bone / motion) run on the GPU in
batch form. The augmentations replicate CTR-GCN's feeder exactly:
valid_crop_resize -> random_rot -> bone -> vel.
"""
import os.path as osp
import queue
import threading

import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DATA_DIR = osp.join(ROOT, 'data', 'ntu_processed')

NTU60_XSUB_TRAIN = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU120_XSUB_TRAIN = NTU60_XSUB_TRAIN + [45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74,
                                        78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
                                        98, 100, 103]

# (child, parent) 1-based joint pairs used by CTR-GCN for the bone modality.
NTU_PAIRS = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
             (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
             (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))

BENCHMARKS = ('ntu60_xsub', 'ntu60_xview', 'ntu120_xsub', 'ntu120_xset', 'ntu60_xset_src')


def load_meta():
    return dict(np.load(osp.join(DATA_DIR, 'meta.npz')))


def split_indices(meta, benchmark):
    """Return (train_idx, test_idx, num_classes) for a benchmark.

    'ntu60_xset_src' is the NTU60 part (setup <= 17, classes A001-A060) of the
    NTU120 cross-setup training set (even setups). It is the source set used
    before adapting to NTU120 X-Set, so no NTU120 X-Set test sample is seen.
    """
    setup, cam, perf, label = meta['setup'], meta['camera'], meta['performer'], meta['label']
    ntu60 = setup <= 17
    if benchmark == 'ntu60_xsub':
        tr = ntu60 & np.isin(perf, NTU60_XSUB_TRAIN)
        te = ntu60 & ~np.isin(perf, NTU60_XSUB_TRAIN)
        nc = 60
    elif benchmark == 'ntu60_xview':
        tr = ntu60 & np.isin(cam, [2, 3])
        te = ntu60 & (cam == 1)
        nc = 60
    elif benchmark == 'ntu120_xsub':
        tr = np.isin(perf, NTU120_XSUB_TRAIN)
        te = ~tr
        nc = 120
    elif benchmark == 'ntu120_xset':
        tr = setup % 2 == 0
        te = ~tr
        nc = 120
    elif benchmark == 'ntu60_xset_src':
        tr = ntu60 & (setup % 2 == 0)
        te = ntu60 & (setup % 2 == 1)
        nc = 60
    else:
        raise ValueError(benchmark)
    return np.where(tr)[0], np.where(te)[0], nc


class NTUData:
    """Holds the memory-mapped joints and builds padded CPU batches."""

    def __init__(self, meta, indices):
        self.joints = np.load(osp.join(DATA_DIR, 'joints.npy'), mmap_mode='r')
        self.indices = np.asarray(indices)
        self.offsets = meta['offsets'][self.indices]
        self.lengths = meta['lengths'][self.indices]
        self.labels = meta['label'][self.indices]

    def __len__(self):
        return len(self.indices)

    def batch(self, pos):
        """pos: positions into this split. Returns (x[B,Lmax,150], lengths, labels)."""
        lens = self.lengths[pos]
        x = np.zeros((len(pos), int(lens.max()), 150), dtype=np.float32)
        for k, p in enumerate(pos):
            o = self.offsets[p]
            x[k, :lens[k]] = self.joints[o:o + lens[k]]
        return x, lens.astype(np.int64), self.labels[pos].astype(np.int64)


class BatchLoader:
    """Iterates over batches using a prefetching background thread."""

    def __init__(self, data, batch_size, shuffle, drop_last, seed=0, prefetch=6):
        self.data, self.bs, self.shuffle, self.drop_last = data, batch_size, shuffle, drop_last
        self.rng = np.random.default_rng(seed)
        self.prefetch = prefetch

    def __len__(self):
        n = len(self.data)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        order = self.rng.permutation(len(self.data)) if self.shuffle else np.arange(len(self.data))
        chunks = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.drop_last and len(chunks[-1]) < self.bs:
            chunks = chunks[:-1]
        q = queue.Queue(maxsize=self.prefetch)

        def worker():
            for c in chunks:
                x, lens, y = self.data.batch(c)
                q.put((torch.from_numpy(x).pin_memory(), torch.from_numpy(lens),
                       torch.from_numpy(y), torch.from_numpy(c)))
            q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        t.join()


def crop_resize(x, lens, train, window=64, p_interval=(0.5, 1.0), p_test=0.95, rng=None):
    """Batched equivalent of CTR-GCN tools.valid_crop_resize.

    x: (B, L, 150) on GPU, lens: (B,) CPU int64. Returns (B, window, 150).
    Linear interpolation along time with align_corners=False semantics.
    """
    B = x.shape[0]
    lens = lens.numpy()
    if train:
        p = rng.random(B) * (p_interval[1] - p_interval[0]) + p_interval[0]
        crop = np.minimum(np.maximum(np.floor(lens * p).astype(np.int64), window), lens)
        bias = np.array([rng.integers(0, v - c + 1) for v, c in zip(lens, crop)])
    else:
        bias = ((1 - p_test) * lens / 2).astype(np.int64)
        crop = lens - 2 * bias
    dev = x.device
    crop_t = torch.from_numpy(crop).to(dev, torch.float32)[:, None]
    bias_t = torch.from_numpy(bias).to(dev)[:, None]
    j = torch.arange(window, device=dev, dtype=torch.float32)[None]
    src = ((j + 0.5) * crop_t / window - 0.5).clamp_min(0)
    i0 = src.floor().long()
    i0 = torch.minimum(i0, crop_t.long() - 1)
    i1 = torch.minimum(i0 + 1, crop_t.long() - 1)
    lam = (src - i0.float()).unsqueeze(-1)
    g0 = torch.gather(x, 1, (i0 + bias_t).unsqueeze(-1).expand(-1, -1, x.shape[2]))
    g1 = torch.gather(x, 1, (i1 + bias_t).unsqueeze(-1).expand(-1, -1, x.shape[2]))
    return g0 * (1 - lam) + g1 * lam


def rotation_matrices(angles):
    """Batched CTR-GCN _rot: angles (B,3) -> Rz @ Ry @ Rx, (B,3,3)."""
    c, s = angles.cos(), angles.sin()
    o, z = torch.ones_like(c[:, 0]), torch.zeros_like(c[:, 0])
    rx = torch.stack([o, z, z, z, c[:, 0], s[:, 0], z, -s[:, 0], c[:, 0]], 1).view(-1, 3, 3)
    ry = torch.stack([c[:, 1], z, -s[:, 1], z, o, z, s[:, 1], z, c[:, 1]], 1).view(-1, 3, 3)
    rz = torch.stack([c[:, 2], s[:, 2], z, -s[:, 2], c[:, 2], z, z, z, o], 1).view(-1, 3, 3)
    return rz @ ry @ rx


def to_model_input(x, lens, train, modality='joint', rng=None, theta=0.3, window=64):
    """(B, L, 150) padded GPU batch -> augmented (B, 3, T, 25, 2) model input."""
    x = crop_resize(x, lens, train, window=window, rng=rng)
    B, T = x.shape[:2]
    x = x.view(B, T, 2, 25, 3)  # (B,T,M,V,C)
    if train:
        ang = torch.from_numpy(rng.uniform(-theta, theta, size=(B, 3))).to(x.device, torch.float32)
        R = rotation_matrices(ang)
        x = torch.einsum('bij,btmvj->btmvi', R, x)
    x = x.permute(0, 4, 1, 3, 2).contiguous()  # (B,C,T,V,M)
    if modality in ('bone', 'bone_motion'):
        child = torch.tensor([a - 1 for a, _ in NTU_PAIRS], device=x.device)
        parent = torch.tensor([b - 1 for _, b in NTU_PAIRS], device=x.device)
        bone = torch.zeros_like(x)
        bone[:, :, :, child] = x[:, :, :, child] - x[:, :, :, parent]
        x = bone
    if modality in ('joint_motion', 'bone_motion'):
        x = torch.cat([x[:, :, 1:] - x[:, :, :-1], torch.zeros_like(x[:, :, :1])], dim=2)
    return x
