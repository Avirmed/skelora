"""Multi-stream ensemble: sum the final test scores of several runs (CTR-GCN convention).

Usage:  python src/analysis/ensemble.py xsub_full xsub_full_bone [--weights 1 1]
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(ROOT, 'src'))
from skar.data import load_meta, split_indices  # noqa: E402


def ensemble_top1(runs, weights=None):
    """Return top-1 accuracy of the weighted score sum, or None if a run is missing."""
    weights = weights or [1.0] * len(runs)
    total, bench = None, None
    for run, w in zip(runs, weights):
        d = osp.join(ROOT, 'experiments', run)
        if not osp.exists(osp.join(d, 'final_scores.npy')):
            return None
        b = json.load(open(osp.join(d, 'summary.json')))['benchmark']
        assert bench in (None, b), 'runs must share a benchmark'
        bench = b
        s = np.load(osp.join(d, 'final_scores.npy')) * w
        total = s if total is None else total + s
    _, te, _ = split_indices(load_meta(), bench)
    labels = load_meta()['label'][te]
    return float((total.argmax(1) == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--weights', type=float, nargs='+')
    a = ap.parse_args()
    acc = ensemble_top1(a.runs, a.weights)
    print('missing run' if acc is None else f'ensemble top1 {100 * acc:.2f}')


if __name__ == '__main__':
    main()
