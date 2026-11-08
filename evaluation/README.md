# CrossTask Transformer downstream evaluation

[详细中文评测指南](docs/CrossTask评测指南.md)

This directory provides the complete runtime for TR (`task_cls`), SR (`step_cls`), and
SF (`step_forecasting`). The pretrained Adapter is frozen; only the Transformer head
is trained. No pretraining dataset is required.

## Download checkpoints and CrossTask data

Download the evaluation materials from [AAAI-26-material](https://bhpan.buaa.edu.cn/link/AAB62640268C3946B69D15936EE50E1EB9).

- Folder name: `AAAI-26-material`
- Available until: **2026-12-09 03:27**
- Contents: pretrained checkpoints and CrossTask data for downstream evaluation.

Place the downloaded `.pth` files under `evaluation/checkpoints/`. Place
`CrossTask_required_full.tar.gz` under `evaluation/` and extract it from the repository root:

```bash
mkdir -p evaluation/data
tar -xzf evaluation/CrossTask_required_full.tar.gz -C evaluation/data
```

The extracted data root should be `evaluation/data/CrossTask/`. The default checkpoint
is `evaluation/checkpoints/task-step-state-step-task.pth`. For another location, pass
`--data-dir` and `--checkpoint` explicitly. See the detailed guide for the required files.
The share link expires at the time listed above; the data and weights are not stored in Git.

## Setup and quick start

Run from the repository root:

```bash
python3.9 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r evaluation/requirements.lock.txt
bash evaluation/quick_start.sh \
  --data-dir /absolute/path/to/CrossTask \
  --checkpoint /absolute/path/to/task-step-state-step-task.pth
```

Defaults: `evaluation/data/CrossTask`, `evaluation/checkpoints/task-step-state-step-task.pth`,
CPU, batch 16, learning rate 1e-4, up to 1000 epochs, patience 50, seed 42.
Use `--device cuda:0` for an available CUDA device, `--tasks step_cls` for one task,
`--epochs 1 --patience 1` for a one-epoch check, or `--prepare-only` to generate configs.
Run `bash evaluation/quick_start.sh --help` for all wrapper options.
The interpreter can be overridden with `PYTHON=/path/to/python`.

Data, split metadata, annotations, and pretrained weights are external inputs; they
are not distributed in this Git repository. See the guide for exact directory layout.
Each run creates a unique `evaluation/experiments/<timestamp>/` containing generated
configs, console logs, engine logs, best head checkpoints, raw results and summary.csv.

## Runtime and reference protocol

`runtime/` contains the downstream engine, dataset loaders, model modules and helper
functions copied from the locally validated implementation. It runs in its own working
directory so that imports do not use or modify the repository's root pretraining code.
The small pretrain_config YAML defines only the Adapter dimensions needed for inference.

The guide documents all critical parameters and the SF history filtering behavior.
Reference results in `reference_results.csv` came from the original local full runs;
they are not newly reproduced training results from this checkout. The reference
protocol retains 33 overlapping TR train/test videos and chooses epochs by test accuracy.
These numbers should not be described as held-out validation-selected results.

`summary.csv` reports the best test accuracy across all completed epochs, not the last
epoch. This entry point trains a new downstream head; it does not implement eval-only
loading of an existing TaskHead checkpoint or resuming interrupted training.


## Local paths and publishing

Pass your local paths through `--data-dir`, `--checkpoint`, `--output-dir`, and optionally
`PYTHON`; see section 11 of the detailed guide. Generated configs and logs contain local
absolute paths and should not be committed. Default runtime directories are ignored.
