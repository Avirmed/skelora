"""Top-1 accuracy of NTU RGB+D 120 runs split into the old (A001-A060) and new (A061-A120) classes.

The old classes of the NTU120 X-Sub test set are exactly the NTU60 X-Sub test set.

Usage:  python src/analysis/class_split.py xsub_lora_ft_r8_topo xsub_full
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(ROOT, 'src'))
from skar.data import load_meta, split_indices  # noqa: E402

N_OLD = 60


def class_split_top1(run, meta=None):
    """Return (all, old, new) top-1 accuracy of a run's final test scores, or None if missing."""
    d = osp.join(ROOT, 'experiments', run)
    if not osp.exists(osp.join(d, 'final_scores.npy')):
        return None
    meta = load_meta() if meta is None else meta
    _, te, _ = split_indices(meta, json.load(open(osp.join(d, 'summary.json')))['benchmark'])
    y = meta['label'][te]
    ok = np.load(osp.join(d, 'final_scores.npy')).argmax(1) == y
    old = y < N_OLD
    return float(ok.mean()), float(ok[old].mean()), float(ok[~old].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    meta = load_meta()
    print(f'{"run":32s} {"all":>6s} {"old-60":>7s} {"new-60":>7s}')
    for run in a.runs:
        r = class_split_top1(run, meta)
        print(f'{run:32s} missing run' if r is None else
              f'{run:32s} {100 * r[0]:6.2f} {100 * r[1]:7.2f} {100 * r[2]:7.2f}')


if __name__ == '__main__':
    main()
