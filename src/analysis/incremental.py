"""Learning the new NTU RGB+D 120 classes without old-class data (class-incremental evaluation).

The adapted runs (--train-classes new) are trained only on A061-A120 with a loss over the new classes
and tested on all 120 classes. An old head (60 scores) and a new head (60 scores) are combined by their
maximum softmax probability: the more confident head predicts (fixed before seeing test results; it is
invariant to the arbitrary logit offset of separately trained heads). From the saved test scores:
  single     one adapted model: old head = fixed source classifier on the adapted features.
  two-model  old head from the untouched source model, new head from the adapted model
             (two forward passes; the source model keeps its old-class accuracy).
  oracle     the old/new decision is given (diagnostic upper bound of the two-model prediction).
H is the harmonic mean of the old- and new-class accuracy. The '(logit)' modes repeat single and two-model
with the larger raw maximum logit instead of the softmax probability (sensitivity check of the rule; for the
single model it equals the plain 120-way argmax that train.py reports).

Usage:  python src/analysis/incremental.py ci_xsub_source ci_xsub_linear ci_xsub_full ci_xsub_lora
        (the first run is the --eval-only source model on the NTU120 test set)
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


def load_scores(run):
    d = osp.join(ROOT, 'experiments', run)
    if not osp.exists(osp.join(d, 'final_scores.npy')):
        return None, None
    return np.load(osp.join(d, 'final_scores.npy')), json.load(open(osp.join(d, 'summary.json')))['benchmark']


def accuracies(pred, y):
    old = y < N_OLD
    ok = pred == y
    a_old, a_new = ok[old].mean(), ok[~old].mean()
    return {'all': ok.mean(), 'old': a_old, 'new': a_new, 'H': 2 * a_old * a_new / (a_old + a_new)}


def max_softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).max(1)


def combine(old_scores, new_scores, rule='msp'):
    """Predict with the head (old: classes 0-59, new: 60-119) that has the higher max softmax probability
    (rule 'msp') or the higher max logit (rule 'logit')."""
    conf = max_softmax if rule == 'msp' else (lambda s: s.max(1))
    use_old = conf(old_scores) >= conf(new_scores)
    return np.where(use_old, old_scores.argmax(1), N_OLD + new_scores.argmax(1))


def predictions(source, adapted):
    """source: [N, 60] scores of the source model; adapted: [N, 120] scores of an adapted run."""
    out = {'new_pred': N_OLD + adapted[:, N_OLD:].argmax(1), 'old_pred': source.argmax(1)}
    for rule, tag in (('msp', ''), ('logit', ' (logit)')):
        out['single' + tag] = combine(adapted[:, :N_OLD], adapted[:, N_OLD:], rule)
        out['two-model' + tag] = combine(source, adapted[:, N_OLD:], rule)
    return out


def evaluate_runs(source_run, runs):
    """Return {run: {mode: accuracies}}, or None if the source scores are missing."""
    source, bench = load_scores(source_run)
    if source is None:
        return None
    meta = load_meta()
    _, te, _ = split_indices(meta, bench)
    y = meta['label'][te]
    out = {}
    for run in runs:
        adapted, b = load_scores(run)
        if adapted is None:
            continue
        assert b == bench, f'{run}: benchmark {b} != {bench}'
        p = predictions(source, adapted)
        oracle = np.where(y < N_OLD, p['old_pred'], p['new_pred'])
        out[run] = {m: accuracies(p[m], y) for m in ('single', 'two-model', 'single (logit)', 'two-model (logit)')}
        out[run]['oracle'] = accuracies(oracle, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source_run')
    ap.add_argument('runs', nargs='+')
    a = ap.parse_args()
    res = evaluate_runs(a.source_run, a.runs)
    if res is None:
        print('missing source run')
        return
    print(f'{"run":24s} {"mode":18s} {"all":>6s} {"old":>6s} {"new":>6s} {"H":>6s}')
    for run, modes in res.items():
        for mode, r in modes.items():
            print(f'{run:24s} {mode:18s} ' + ' '.join(f'{100 * r[k]:6.2f}' for k in ('all', 'old', 'new', 'H')))


if __name__ == '__main__':
    main()
