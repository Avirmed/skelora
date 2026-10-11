"""Train / fine-tune / LoRA-adapt CTR-GCN on NTU RGB+D 60/120.

Examples
--------
Base training (standard CTR-GCN recipe):
    python src/skar/train.py --benchmark ntu60_xsub --name base_xsub_joint

LoRA adaptation NTU60 -> NTU120 (SkelLoRA):
    python src/skar/train.py --benchmark ntu120_xsub --init experiments/base_xsub_joint/final.pt \
        --tune lora --lora-targets ft --lora-rank 8 --tune-topo --epochs 20 --steps 12 17 \
        --warmup 2 --eval-last 5 --name xsub_lora_ft_r8_topo

Learning only the new classes (A061-A120) without old-class data: add --train-classes new.
Low-data adaptation: add --train-frac 0.1 (class-balanced subset, identical for every method).
Evaluating a checkpoint without training: --eval-only --init <checkpoint>.
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
sys.path.insert(0, osp.join(ROOT, 'src'))
sys.path.insert(0, osp.join(ROOT, 'third_party', 'CTR-GCN'))

from skar.data import BatchLoader, NTUData, load_meta, split_indices, to_model_input  # noqa: E402
from skar.lora import LoRAConv2d, inject_lora, merge_lora  # noqa: E402
from model.ctrgcn import Model  # noqa: E402

N_OLD = 60  # classes A001-A060 of the NTU RGB+D 60 source models


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--modality', default='joint', choices=['joint', 'bone', 'joint_motion', 'bone_motion'])
    p.add_argument('--name', required=True)
    p.add_argument('--epochs', type=int, default=65)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--steps', type=int, nargs='+', default=[35, 55])
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--wd', type=float, default=4e-4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--init', default='', help='checkpoint of the source (base) model')
    p.add_argument('--tune', default='full', choices=['full', 'linear', 'bn', 'lora'])
    p.add_argument('--lora-targets', default='ft', help="subset of 'frt'; 'g' = 'fr'")
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--lora-alpha', type=float, default=0, help='0 -> alpha = rank')
    p.add_argument('--tune-topo', action='store_true', help='train shared topology PA and alpha')
    p.add_argument('--freeze-bn', action='store_true', help='keep BN frozen in lora mode')
    p.add_argument('--train-classes', default='all', choices=['all', 'new'],
                   help="'new': train only on A061-A120; the source classifier rows of A001-A060 stay fixed")
    p.add_argument('--train-frac', type=float, default=1.0, help='class-balanced fraction of the training set')
    p.add_argument('--eval-only', action='store_true', help='evaluate --init on the test set without training')
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-last', type=int, default=10, help='evaluate every epoch for the last N epochs')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--max-batches', type=int, default=0, help='debug: limit batches per epoch')
    return p.parse_args()


def build_model(num_class):
    return Model(num_class=num_class, num_point=25, num_person=2,
                 graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})


def load_source(model, path):
    """Load source weights; the classifier rows of shared classes are copied."""
    sd = torch.load(path, map_location='cpu')
    own = model.state_dict()
    for k, v in sd.items():
        k = k.replace('module.', '', 1)  # checkpoints saved from nn.DataParallel
        if k not in own:
            continue
        if k.startswith('fc.') and v.shape != own[k].shape:
            own[k][:v.shape[0]] = v  # classes A001-A060 keep their source classifier
        else:
            own[k] = v
    model.load_state_dict(own)


def class_balanced_subset(idx, labels, frac, seed=0):
    """Keep round(frac * n_c), at least 1, samples of every class. The subset depends only on frac,
    so every method and training seed sees the same samples."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[idx]):
        members = idx[labels[idx] == c]
        keep.append(rng.choice(members, max(1, round(frac * len(members))), replace=False))
    return np.sort(np.concatenate(keep))


def configure_tuning(model, args):
    """Set requires_grad according to the tuning mode. Returns the list of BN modules kept frozen."""
    if args.tune == 'full':
        return []
    for p in model.parameters():
        p.requires_grad = False
    if args.tune == 'lora':
        alpha = args.lora_alpha or args.lora_rank
        inject_lora(model, args.lora_targets, args.lora_rank, alpha)  # new LoRA params are trainable
    tune_bn = args.tune == 'bn' or (args.tune == 'lora' and not args.freeze_bn)
    frozen_bn = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if tune_bn:
                for p in m.parameters():
                    p.requires_grad = True
            else:
                frozen_bn.append(m)
    if args.tune_topo:
        for n, p in model.named_parameters():
            if n.endswith('.PA') or n.endswith('.alpha'):
                p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    return frozen_bn


def lr_at(epoch, args):
    if epoch < args.warmup:
        return args.lr * (epoch + 1) / args.warmup
    return args.lr * (0.1 ** int(np.sum(epoch >= np.array(args.steps))))


@torch.no_grad()
def evaluate(model, loader, modality, device):
    """Test in float32: bfloat16 inference costs up to 0.4 points (official CTR-GCN NTU60 X-Sub joint
    checkpoint: 89.49% in bfloat16, 89.92% in float32)."""
    model.eval()
    scores, labels = [], []
    for x, lens, y, _ in loader:
        x = to_model_input(x.to(device, non_blocking=True), lens, False, modality)
        out = model(x)
        scores.append(out.float().cpu())
        labels.append(y)
    scores, labels = torch.cat(scores), torch.cat(labels)
    top1 = (scores.argmax(1) == labels).float().mean().item()
    top5 = (scores.topk(5, 1).indices == labels[:, None]).any(1).float().mean().item()
    return top1, top5, scores.numpy()


def evaluate_checkpoint(args, test_loader, out_dir, device):
    """Evaluate a checkpoint as it is; its number of classes is taken from its classifier."""
    sd = torch.load(args.init, map_location='cpu')
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model = build_model(sd['fc.weight'].shape[0])
    model.load_state_dict(sd)
    model.to(device)
    top1, top5, scores = evaluate(model, test_loader, args.modality, device)
    np.save(osp.join(out_dir, 'final_scores.npy'), scores)
    summary = {'name': args.name, 'benchmark': args.benchmark, 'modality': args.modality, 'tune': 'eval_only',
               'init': args.init, 'num_class': int(sd['fc.weight'].shape[0]),
               'final_top1': top1, 'final_top5': top5, 'eval_precision': 'fp32'}
    with open(osp.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'eval-only {args.init}: top1 {100 * top1:.2f} top5 {100 * top5:.2f}', flush=True)


def count_flops(model, device):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, 3, 64, 25, 2, device=device)
    model.eval()
    for p in model.parameters():  # FlopCounterMode's module tracker fails under no_grad
        p.requires_grad_(False)
    with FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2  # multiply-accumulates -> GMACs convention (FLOPs/2)


def main():
    args = get_args()
    out_dir = osp.join(ROOT, 'experiments', args.name)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    amp = not args.no_amp

    meta = load_meta()
    tr_idx, te_idx, num_class = split_indices(meta, args.benchmark)
    if args.train_classes == 'new':
        assert args.init and num_class > N_OLD, '--train-classes new needs a source model and NTU120'
        tr_idx = tr_idx[meta['label'][tr_idx] >= N_OLD]
    if args.train_frac < 1:
        tr_idx = class_balanced_subset(tr_idx, meta['label'], args.train_frac)
    train_data, test_data = NTUData(meta, tr_idx), NTUData(meta, te_idx)
    train_loader = BatchLoader(train_data, args.batch_size, True, True, seed=args.seed)
    test_loader = BatchLoader(test_data, 64, False, False)

    if args.eval_only:
        evaluate_checkpoint(args, test_loader, out_dir, device)
        return

    model = build_model(num_class)
    if args.init:
        load_source(model, args.init)
    frozen_bn = configure_tuning(model, args)
    model.to(device)
    # Without old-class data, the loss covers only the new classes (separated softmax; a loss over all
    # 120 classes would teach the model to never predict an old class), and the source classifier rows
    # of A001-A060 are kept fixed (restored after every step against weight decay and momentum).
    keep_rows, first = None, 0
    if args.train_classes == 'new':
        keep_rows = (model.fc.weight[:N_OLD].detach().clone(), model.fc.bias[:N_OLD].detach().clone())
        first = N_OLD

    n_total = sum(p.numel() for p in model.parameters())
    n_lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    n_backbone = n_total - n_lora
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if keep_rows is not None:
        n_train -= keep_rows[0].numel() + keep_rows[1].numel()
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    log_path = osp.join(out_dir, 'log.txt')
    logf = open(log_path, 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log('args: ' + json.dumps(vars(args)))
    log(f'train {len(train_data)} test {len(test_data)} classes {num_class}')
    log(f'params: backbone {n_backbone} lora {n_lora} trainable {n_train} '
        f'({100.0 * n_train / n_backbone:.2f}% of backbone)')

    history, best, epoch_times = [], (0.0, -1), []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        lr = lr_at(epoch, args)
        for g in optim.param_groups:
            g['lr'] = lr
        model.train()
        for m in frozen_bn:
            m.eval()
        t0, n_seen, loss_sum, correct = time.time(), 0, 0.0, 0
        for b, (x, lens, y, _) in enumerate(train_loader):
            if args.max_batches and b >= args.max_batches:
                break
            x = to_model_input(x.to(device, non_blocking=True), lens, True, args.modality, rng=rng)
            y = y.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                out = model(x)[:, first:]
                y = y - first
                loss = loss_fn(out.float(), y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if keep_rows is not None:
                with torch.no_grad():
                    model.fc.weight[:N_OLD] = keep_rows[0]
                    model.fc.bias[:N_OLD] = keep_rows[1]
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n_seen += len(y)
        epoch_times.append(time.time() - t0)
        msg = (f'epoch {epoch + 1}/{args.epochs} lr {lr:.4f} loss {loss_sum / n_seen:.4f} '
               f'train_acc {100.0 * correct / n_seen:.2f} time {epoch_times[-1]:.0f}s')
        last = epoch + 1 == args.epochs
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 > args.epochs - args.eval_last or last:
            top1, top5, scores = evaluate(model, test_loader, args.modality, device)
            history.append({'epoch': epoch + 1, 'top1': top1, 'top5': top5})
            if top1 > best[0]:
                best = (top1, epoch + 1)
                np.save(osp.join(out_dir, 'best_scores.npy'), scores)
            msg += f' | test top1 {100 * top1:.2f} top5 {100 * top5:.2f}'
            if last:
                np.save(osp.join(out_dir, 'final_scores.npy'), scores)
        log(msg)

    peak_mem = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.save(model.state_dict(), osp.join(out_dir, 'final.pt'))
    # Persist only the trainable (task-specific) tensors: this is what a LoRA adapter costs on disk.
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    task_sd = {k: v for k, v in model.state_dict().items()
               if k in trainable_names or 'running_' in k or 'num_batches' in k}
    torch.save(task_sd, osp.join(out_dir, 'task_params.pt'))

    # Merge LoRA and check that inference cost and predictions are unchanged.
    merged_check = None
    if args.tune == 'lora':
        top1_before = history[-1]['top1']
        merge_lora(model)
        top1_merged, _, _ = evaluate(model, test_loader, args.modality, device)
        merged_check = {'top1_unmerged': top1_before, 'top1_merged': top1_merged}
        log(f'merged LoRA: top1 {100 * top1_merged:.2f} (unmerged {100 * top1_before:.2f})')
    plain = build_model(num_class).to(device)
    gmacs = count_flops(plain, device) / 1e9

    summary = {
        'name': args.name, 'benchmark': args.benchmark, 'modality': args.modality, 'tune': args.tune,
        'lora_targets': args.lora_targets if args.tune == 'lora' else None,
        'lora_rank': args.lora_rank if args.tune == 'lora' else None,
        'tune_topo': args.tune_topo, 'init': args.init, 'epochs': args.epochs, 'seed': args.seed,
        'train_classes': args.train_classes, 'train_frac': args.train_frac, 'n_train_samples': len(train_data),
        'final_top1': history[-1]['top1'], 'final_top5': history[-1]['top5'],
        'best_top1': best[0], 'best_epoch': best[1],
        'params_backbone': n_backbone, 'params_lora': n_lora, 'params_trainable': n_train,
        'trainable_pct': 100.0 * n_train / n_backbone,
        'task_file_mb': osp.getsize(osp.join(out_dir, 'task_params.pt')) / 2 ** 20,
        'peak_mem_gb': peak_mem, 'sec_per_epoch': float(np.mean(epoch_times)),
        'gmacs_per_sample': gmacs, 'merged_check': merged_check, 'eval_precision': 'fp32', 'history': history,
    }
    with open(osp.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'final top1 {100 * summary["final_top1"]:.2f} best {100 * best[0]:.2f} @ {best[1]} '
        f'| trainable {n_train} | peak mem {peak_mem:.2f} GB | {summary["sec_per_epoch"]:.0f}s/epoch')


if __name__ == '__main__':
    main()
