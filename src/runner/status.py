"""Print training progress: the running job (epoch, accuracy, ETA), the queue with estimated
finish times, and the results of finished runs.

Usage:  python src/runner/status.py
"""
import datetime as dt
import glob
import json
import os
import os.path as osp
import re

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
EXP = osp.join(ROOT, 'experiments')
N_TRAIN = {'ntu60_xsub': 40091, 'ntu60_xview': 37646, 'ntu120_xsub': 63026,
           'ntu120_xset': 54468, 'ntu60_xset_src': 26476}
N_TEST = {'ntu60_xsub': 16487, 'ntu60_xview': 18932, 'ntu120_xsub': 50919,
          'ntu120_xset': 59477, 'ntu60_xset_src': 30102}
NEW_CLASS_SHARE = {'ntu120_xsub': 22935 / 63026, 'ntu120_xset': 27992 / 54468}  # A061-A120 in the train set
SEC_PER_TRAIN_SAMPLE = 83 / 40091  # measured: 83 s per NTU60 X-Sub epoch
SEC_PER_TEST_SAMPLE = 0.0007


def estimate_hours(args):
    b = re.search(r'--benchmark\s+(\S+)', args).group(1)
    eval_time = N_TEST[b] * SEC_PER_TEST_SAMPLE
    if '--eval-only' in args:
        return eval_time / 3600
    ep = int(re.search(r'--epochs\s+(\d+)', args).group(1)) if '--epochs' in args else 65
    ev_last = int(re.search(r'--eval-last\s+(\d+)', args).group(1)) if '--eval-last' in args else 10
    ev_every = int(re.search(r'--eval-every\s+(\d+)', args).group(1)) if '--eval-every' in args else 5
    n_eval = ev_last + max(0, (ep - ev_last)) // ev_every + 1
    factor = 0.6 if '--tune linear' in args else 1.0
    if '--train-classes new' in args:
        factor *= NEW_CLASS_SHARE[b]
    m = re.search(r'--train-frac\s+([\d.]+)', args)
    if m:
        factor *= float(m.group(1))
    return (ep * N_TRAIN[b] * SEC_PER_TRAIN_SAMPLE * factor + n_eval * eval_time) / 3600


def running():
    for log in sorted(glob.glob(osp.join(EXP, '*', 'log.txt')), key=osp.getmtime, reverse=True):
        d = osp.dirname(log)
        if not osp.exists(osp.join(d, 'summary.json')) and dt.datetime.now().timestamp() - osp.getmtime(log) < 900:
            return d, open(log, encoding='utf-8').read().splitlines()
    return None, None


def main():
    now = dt.datetime.now()
    print(f'=== {now:%Y-%m-%d %H:%M} ===')
    t_free = now
    d, lines = running()
    if d:
        epochs = [l for l in lines if l.startswith('epoch')]
        args = json.loads(lines[0][6:]) if lines and lines[0].startswith('args:') else {}
        total = args.get('epochs', 65)
        done = len(epochs)
        times = [float(re.search(r'time (\d+)s', l).group(1)) for l in epochs]
        per = sum(times[-5:]) / max(1, len(times[-5:])) if times else 0
        tests = [l for l in epochs if 'test top1' in l]
        eta = now + dt.timedelta(seconds=(total - done) * per * 1.08)
        t_free = eta
        print(f'RUNNING  {osp.basename(d)}  epoch {done}/{total}  ({per:.0f}s/epoch)  ETA {eta:%a %H:%M}')
        if tests:
            print('         last test:', tests[-1].split('|')[1].strip(), f'(epoch {len(epochs)})' if tests[-1] == epochs[-1] else '')
    else:
        print('RUNNING  (none)')

    qfile = osp.join(EXP, 'queue.txt')
    jobs = [l.strip() for l in open(qfile, encoding='utf-8')] if osp.exists(qfile) else []
    jobs = [j for j in jobs if j and not j.startswith('#') and not j.startswith('WAIT')]
    print(f'\nQUEUE ({len(jobs)} jobs)')
    for j in jobs:
        t_free += dt.timedelta(hours=estimate_hours(j))
        name = re.search(r'--name\s+(\S+)', j).group(1)
        print(f'  {name:28s} ~{estimate_hours(j):.1f}h   done ~{t_free:%a %H:%M}')

    print('\nFINISHED')
    for s in sorted(glob.glob(osp.join(EXP, '*', 'summary.json')), key=osp.getmtime):
        r = json.load(open(s))
        extra = 'eval only' if r.get('tune') == 'eval_only' else f'trainable {r["trainable_pct"]:.1f}%'
        print(f'  {r["name"]:28s} {r["benchmark"]:12s} top1 {100 * r["final_top1"]:.2f}  {extra}')


if __name__ == '__main__':
    main()
