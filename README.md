# SkelLoRA: low-rank adaptation of skeleton GCNs

Code for adapting a CTR-GCN skeleton action recognition model, trained on NTU RGB+D 60, to
NTU RGB+D 120 with low-rank adaptation (LoRA) and with the usual baselines (training from scratch,
linear probe, BN tuning, full fine-tuning).

What the code provides:

- `LoRAConv2d` (`src/skar/lora.py`): LoRA for 1×1 and temporal k×1 convolutions; the update is computed
  as a merged kernel in every forward pass and can be merged into the frozen weights after training
  (`merge_lora`), so the adapted model has the same architecture and FLOPs as the source model.
- Placement options for CTR-GCN (`--lora-targets`, any subset of `frt`): `f` = spatial feature
  transforms (CTRGC conv3 and the residual down-sampling), `r` = channel-wise relation convolutions
  (conv1, conv2, conv4), `t` = temporal convolutions and block residuals. `--tune-topo` also trains the
  shared topology (PA) and alpha.
- Adaptation settings: complete target training set, class-balanced low-data subsets
  (`--train-frac`), and learning only the new classes A061-A120 without old-class data
  (`--train-classes new`; loss over the new-class logits, source classifier rows kept fixed).
- A fast data pipeline: parallel preprocessing into one variable-length array, and GPU augmentation
  that matches the CTR-GCN feeder.
- Evaluation in float32 (training uses bfloat16 autocast).

## Requirements

- Python 3.11, PyTorch 2.11 with CUDA (tested on one RTX 4080 SUPER, 16 GB), see `requirements.txt`.
- The official CTR-GCN code (model, graph, preprocessing functions), which is not included here:

```
git clone https://github.com/Uason-Chen/CTR-GCN third_party/CTR-GCN
cd third_party/CTR-GCN && git checkout e13d7582e281d06711eeecb380a472b278ae1663
```

## Data

The NTU RGB+D skeleton data is not included and must be requested from the dataset authors
(https://rose1.ntu.edu.sg/dataset/actionRecognition). Extract the skeleton files as in CTR-GCN:

```
data/nturgbd_raw/nturgb+d_skeletons/      # nturgbd_skeletons_s001_to_s017.zip
data/nturgbd_raw/nturgb+d_skeletons120/   # nturgbd_skeletons_s018_to_s032.zip
```

Then preprocess all 113,945 valid samples once (CTR-GCN denoising and sequence translation, run in a
process pool):

```
python src/prep/prepare_ntu.py
```

This writes `data/ntu_processed/joints.npy` and `meta.npz`. Benchmarks: `ntu60_xsub`, `ntu60_xview`,
`ntu120_xsub`, `ntu120_xset`, and `ntu60_xset_src` (the NTU RGB+D 60 part of the NTU RGB+D 120 X-Set
training set, used as the source for X-Set adaptation so that no target test sample is seen).

## Training

Each run writes `experiments/<name>/` (log, `summary.json`, final checkpoint, test scores).

```
# source model on NTU RGB+D 60 (standard CTR-GCN recipe: 65 epochs)
python src/skar/train.py --benchmark ntu60_xsub --name base_ntu60xsub_joint

# LoRA adaptation to NTU RGB+D 120 (rank 8 on f and t, topology tuned)
python src/skar/train.py --benchmark ntu120_xsub --init experiments/base_ntu60xsub_joint/final.pt \
    --tune lora --lora-targets ft --lora-rank 8 --tune-topo --epochs 20 --warmup 2 --steps 12 17 \
    --eval-last 5 --name xsub_lora_ft_r8_topo

# baselines: --tune full | linear | bn ; training from scratch: omit --init
# low-data: --train-frac 0.1 ; new classes only: --train-classes new
# test a checkpoint: --eval-only --init <checkpoint>
```

`scripts/all_runs.txt` lists the arguments of every run of the study. Runs can be queued and
executed one after another on one GPU:

```
cp scripts/all_runs.txt experiments/queue.txt   # or only the lines you need
python src/runner/run_queue.py
python src/runner/status.py                     # progress, finish times, finished results
```

## Analysis

```
python src/analysis/ensemble.py xsub_full xsub_full_bone            # multi-stream score ensemble
python src/analysis/class_split.py xsub_lora_ft_r8_topo xsub_full   # accuracy on old / new classes
python src/analysis/incremental.py ci_xsub_source ci_xsub_linear ci_xsub_full ci_xsub_lora
```

`incremental.py` evaluates runs trained with `--train-classes new`: the old-class head and the
new-class head are combined by their maximum softmax probability (single adapted model, or source
model + adapted model), with the raw-logit rule and the oracle decision as references.

## Tests

```
python src/tests/test_lora.py      # merged LoRA kernel equals the two-branch computation
python src/tests/test_augment.py   # GPU augmentation matches the CTR-GCN feeder (needs the data)
```

## Layout

```
src/skar/       data.py (splits, loader, GPU augmentation), lora.py, train.py
src/prep/       prepare_ntu.py
src/runner/     run_queue.py, status.py
src/analysis/   ensemble.py, class_split.py, incremental.py
src/tests/      test_lora.py, test_augment.py
scripts/        all_runs.txt
```

## Third-party code and data

- CTR-GCN (https://github.com/Uason-Chen/CTR-GCN) is licensed under CC BY-NC 4.0. This repository
  imports its model and preprocessing code from `third_party/CTR-GCN`; the per-sample sequence
  translation in `src/prep/prepare_ntu.py` and the GPU augmentation in `src/skar/data.py` are adapted
  from it.
- NTU RGB+D and NTU RGB+D 120 are subject to the dataset authors' terms of use. No data, processed
  data, or model weights are distributed here.
