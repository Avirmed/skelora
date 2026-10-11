"""Sequential GPU job queue.

Reads experiments/queue.txt, pops the first non-empty, non-comment line and
runs it; repeats until the queue is empty. The file may be edited while the
runner is active. Special line: `WAIT <path>` blocks until <path> exists.
Each job's output goes to experiments/logs/<--name>.out; a record of every
finished job is appended to experiments/queue_done.txt.
"""
import os
import os.path as osp
import re
import shlex
import subprocess
import sys
import time

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
QUEUE = osp.join(ROOT, 'experiments', 'queue.txt')
DONE = osp.join(ROOT, 'experiments', 'queue_done.txt')
PY = sys.executable  # jobs run with the same Python as the runner


def pop():
    if not osp.exists(QUEUE):
        return None
    with open(QUEUE, encoding='utf-8') as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith('#'):
            with open(QUEUE, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines[:i] + lines[i + 1:]) + '\n')
            return line.strip()
    return None


def main():
    while True:
        job = pop()
        if job is None:
            print('queue empty, exiting', flush=True)
            return
        if job.startswith('WAIT '):
            path = osp.join(ROOT, job[5:].strip())
            print('waiting for', path, flush=True)
            while not osp.exists(path):
                time.sleep(30)
            continue
        m = re.search(r'--name\s+(\S+)', job)
        name = m.group(1) if m else 'job_%d' % int(time.time())
        cmd = [PY, osp.join(ROOT, 'src', 'skar', 'train.py')] + shlex.split(job)
        t0 = time.time()
        print('start', name, flush=True)
        with open(osp.join(ROOT, 'experiments', 'logs', name + '.out'), 'w') as out:
            rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
        with open(DONE, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M")}\trc={rc}\t{(time.time() - t0) / 3600:.2f}h\t{job}\n')
        print('done', name, 'rc', rc, flush=True)


if __name__ == '__main__':
    main()
