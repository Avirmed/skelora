"""Parallel NTU RGB+D 60/120 skeleton preprocessing.

Reuses the official CTR-GCN preprocessing functions (parsing, denoising and
sequence translation) but runs them per sample in a process pool, and stores
the result in a compact variable-length format instead of a padded npz:

    data/ntu_processed/joints.npy   float32 (total_frames, 150)
                                    [body-1 (25x3) | body-2 (25x3)],
                                    body-2 is zero when only one actor exists
    data/ntu_processed/meta.npz     names, offsets, lengths, setup, camera,
                                    performer, replication, label (0-based)

NTU60 is the subset with setup <= 17, so a single pass over all NTU120
samples covers both datasets.
"""
import argparse
import logging
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
CTR_DIR = osp.join(ROOT, 'third_party', 'CTR-GCN', 'data', 'ntu120')
WORK_DIR = osp.join(ROOT, 'data', 'work')  # '../nturgbd_raw' resolves from here
SKES_PATH = '../nturgbd_raw/nturgb+d_skeletons/'

_grs = None
_grd = None
_null_logger = None


def _init_worker():
    global _grs, _grd, _null_logger
    # Old CTR-GCN code uses removed numpy aliases.
    for alias, typ in (('int', int), ('float', float), ('bool', bool)):
        if not hasattr(np, alias):
            setattr(np, alias, typ)
    os.chdir(WORK_DIR)
    sys.path.insert(0, CTR_DIR)
    import get_raw_skes_data as grs
    import get_raw_denoised_data as grd  # creates ./denoised_data log dirs
    _grs, _grd = grs, grd
    _null_logger = logging.getLogger('null')
    _null_logger.addHandler(logging.NullHandler())
    _null_logger.propagate = False


def _seq_translation(ske_joints):
    """Per-sample version of CTR-GCN seq_translation (origin = joint-2 of first valid frame)."""
    num_frames = ske_joints.shape[0]
    two = ske_joints.shape[1] == 150
    if two:
        miss1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
        miss2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
    i = 0
    while i < num_frames:
        if np.any(ske_joints[i, :75] != 0):
            break
        i += 1
    origin = np.copy(ske_joints[i, 3:6])
    ske_joints = ske_joints - np.tile(origin, 50 if two else 25)
    if two:
        ske_joints[miss1, :75] = 0
        ske_joints[miss2, 75:] = 0
    return ske_joints


def _process(name):
    bodies = _grs.get_raw_bodies_data(SKES_PATH, name, {}, _null_logger)
    if len(bodies['data']) == 1:
        body = list(bodies['data'].values())[0]
        joints, _ = _grd.get_one_actor_points(body, bodies['num_frames'])
    else:
        joints, colors = _grd.get_two_actors_points(bodies)
        joints, _ = _grd.remove_missing_frames(name, joints, colors)
    joints = _seq_translation(joints.astype(np.float32))
    if joints.shape[1] == 75:  # zero-pad the second body (NTU60 convention)
        joints = np.hstack((joints, np.zeros_like(joints)))
    return name, joints.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--out', default=osp.join(ROOT, 'data', 'ntu_processed'))
    parser.add_argument('--limit', type=int, default=0, help='debug: only first N samples')
    args = parser.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    names = np.loadtxt(osp.join(CTR_DIR, 'statistics', 'skes_available_name.txt'), dtype=str)
    if args.limit:
        names = names[:args.limit]
    print('samples:', len(names), flush=True)

    t0 = time.time()
    results = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, res in enumerate(pool.imap(_process, names, chunksize=32)):
            results.append(res)
            if (k + 1) % 5000 == 0:
                print('%d / %d  (%.0fs)' % (k + 1, len(names), time.time() - t0), flush=True)

    assert [r[0] for r in results] == list(names)
    lengths = np.array([r[1].shape[0] for r in results], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    joints = np.lib.format.open_memmap(osp.join(args.out, 'joints.npy'), mode='w+',
                                       dtype=np.float32, shape=(int(lengths.sum()), 150))
    for (_, arr), off in zip(results, offsets):
        joints[off:off + arr.shape[0]] = arr
    joints.flush()

    meta = dict(
        names=names,
        offsets=offsets,
        lengths=lengths,
        setup=np.array([int(n[1:4]) for n in names]),
        camera=np.array([int(n[5:8]) for n in names]),
        performer=np.array([int(n[9:12]) for n in names]),
        replication=np.array([int(n[13:16]) for n in names]),
        label=np.array([int(n[17:20]) - 1 for n in names]),
    )
    np.savez(osp.join(args.out, 'meta.npz'), **meta)
    print('done: %d samples, %d frames, max len %d, %.0fs' % (
        len(names), lengths.sum(), lengths.max(), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
