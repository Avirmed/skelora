"""Low-data robustness tables (NTU RGB+D 120 X-Sub, joint stream, final-epoch top-1).

  subsets         three independently sampled class-balanced subsets (subset seeds 0-2, training seed 1)
  ablation        component ablation with 5% and 10% of the data (subset 0, training seeds 1-3)
  onepct          1% of the data: fewer trainable parameters (subset 0, training seeds 1-3)
  learning_rate   full fine-tuning and SkelLoRA with learning rates 0.1 / 0.02 / 0.01

Run names follow scripts/all_runs.txt (low{01,05,10}_xsub_<method>[_d<subset>][_s<seed>][_lr<rate>]).

Usage:  python src/analysis/lowdata.py
"""
import json
import os.path as osp
import sys

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
EXP = osp.join(ROOT, 'experiments')


def load(name):
    """summary.json of a run, or None if the run does not exist."""
    p = osp.join(EXP, name, 'summary.json')
    if not osp.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

FRACS = ('01', '05', '10')
SEEDS = ('', '_s2', '_s3')


def values(runs):
    return [100 * s['final_top1'] for s in map(load, runs) if s is not None]


def fmt(v):
    if not v:
        return '--'
    return f'{np.mean(v):.2f}' if len(v) == 1 else f'{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f} (n={len(v)})'


def trainable(run):
    s = load(run)
    return '--' if s is None else f'{s["params_trainable"] / 1e3:.0f}K'


def subsets():
    print('subsets 0-2 (training seed 1): mean ± std over subsets; per-subset values')
    for f in FRACS:
        per = {}
        for m in ('linear', 'full', 'lora'):
            per[m] = values([f'low{f}_xsub_{m}', f'low{f}_xsub_{m}_d1', f'low{f}_xsub_{m}_d2'])
            print(f'  {f}%  {m:7s} {fmt(per[m]):28s} ' + ' '.join(f'{v:.2f}' for v in per[m]))
        if len(per['lora']) == len(per['full']) and per['lora']:
            d = np.array(per['lora']) - np.array(per['full'])
            print(f'        lora-full per subset: ' + ' '.join(f'{x:+.2f}' for x in d) + f'  mean {d.mean():+.2f}')


def ablation():
    print('ablation, subset 0, training seeds 1-3')
    rows = [('SkelLoRA (f+t+topology)', 'lora'), ('f+t, no topology', 'notopo'), ('t+topology', 'ttopo'),
            ('f+topology', 'ftopo'), ('topology+BN (no LoRA)', 'bntopo'), ('BN only', 'bn'),
            ('linear probe', 'linear'), ('full fine-tuning', 'full')]
    for label, key in rows:
        cells = [fmt(values([f'low{f}_xsub_{key}{s}' for s in SEEDS])) for f in ('05', '10')]
        print(f'  {label:26s} {trainable(f"low05_xsub_{key}"):>6s}  5%: {cells[0]:28s} 10%: {cells[1]}')


def onepct():
    print('1% of the data (659 samples), subset 0, training seeds 1-3')
    for label, key in [('linear probe', 'linear'), ('BN only', 'bn'), ('f+t, no topology', 'notopo'),
                       ('topology+BN', 'bntopo'),
                       ('SkelLoRA r=2', 'r2'), ('SkelLoRA r=8', 'lora'), ('full fine-tuning', 'full')]:
        runs = [f'low01_xsub_{key}{s}' for s in SEEDS]
        print(f'  {label:18s} {trainable(runs[0]):>6s}  {fmt(values(runs))}')


def learning_rate():
    print('LR check: lr 0.1 (default) / 0.02 / 0.01, subset 0, training seed 1; full FT lr 0.02 over seeds 1-3')
    for f in FRACS:
        for m in ('full', 'lora'):
            cells = [fmt(values([f'low{f}_xsub_{m}' + (f'_lr{t}' if t else '')])) for t in ('', '02', '01')]
            print(f'  {f}%  {m:5s} ' + ' / '.join(cells))
        if f != '01':
            print(f'        full lr 0.02, seeds 1-3: {fmt(values([f"low{f}_xsub_full_lr02{s}" for s in SEEDS]))}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    subsets()
    ablation()
    onepct()
    learning_rate()
