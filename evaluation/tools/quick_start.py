#!/usr/bin/env python3
"""Prepare portable configs and run CrossTask Transformer head training/evaluation."""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys

TASKS = ('task_cls', 'step_cls', 'step_forecasting')


def main():
    root = Path(__file__).resolve().parents[2]
    base = root / 'evaluation'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=base / 'checkpoints/task-step-state-step-task.pth')
    parser.add_argument('--data-dir', type=Path, default=base / 'data/CrossTask')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cpu', help='cpu or cuda:0; MPS is not used')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import yaml
    import torch
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.num_workers < 0:
        parser.error('epochs, patience, batch-size must be positive; num-workers >= 0')
    if args.device != 'cpu' and not args.device.startswith('cuda:'):
        parser.error('device must be cpu or cuda:N')
    if args.device.startswith('cuda:') and not torch.cuda.is_available():
        parser.error('CUDA requested but torch.cuda.is_available() is false')
    args.data_dir = args.data_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file() or not args.checkpoint.stat().st_size:
        parser.error(f'missing or empty checkpoint: {args.checkpoint}')
    feats = args.data_dir / 'feats_3segs_no_mask'
    required = []
    for task in args.tasks:
        required.extend(feats / f'{task}_{split}_split.pickle' for split in ('train', 'test'))
        if task != 'task_cls':
            required.extend(feats / f'{task}_{suffix}.pickle'
                            for suffix in ('task2step_meta', 'set_of_steps'))
    missing = [str(p) for p in required if not p.is_file()]
    if any(t != 'task_cls' for t in args.tasks):
        if not any((args.data_dir / 'crosstask_release/annotations').glob('*.csv')):
            missing.append(str(args.data_dir / 'crosstask_release/annotations/*.csv'))
    if not any(feats.glob('*/video.npy')):
        missing.append(str(feats / '*/video.npy'))
    if missing:
        parser.error('missing data:\n' + '\n'.join(missing))
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_dir or base / 'experiments' / stamp).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env['MPLCONFIGDIR'] = str(out / 'matplotlib')
    commands = []
    for task in args.tasks:
        cfg = yaml.safe_load((base / f'configs/{task}_Full.yml').read_text())
        cfg.update(pretrain_config=str(base / 'runtime/configs/CoP_pretrain/CoP_pretrain_by_stage.yml'),
                   cross_dir=str(args.data_dir), cross_task_video_dir=str(args.data_dir / 'raw_video'),
                   cross_task_s3d_feat_dir=str(feats), cross_task_annot_dir=str(args.data_dir),
                   cross_task_annoataion_dir=str(args.data_dir / 'crosstask_release/annotations'),
                   final_acc_output=str(out / 'raw_results.csv'), checkpoint_dir=str(out / 'checkpoints'),
                   log_dir=str(out / 'logs'), device=args.device, seed=args.seed,
                   num_workers=args.num_workers, task_head_num_epochs=args.epochs,
                   task_head_early_stop_patience=args.patience, task_head_batch_size=args.batch_size)
        cfg_path = out / f'{task}.yml'
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        commands.append((task, [sys.executable, str(base / 'runtime/engines/main_train_taskhead.py'),
                               '--cfg', str(cfg_path), '--checkpoint', str(args.checkpoint),
                               '--downstream_dataset_name', 'CrossTask', '--downstream_task_name', task,
                               '--notes', 'QuickStart', '--use_wandb', '0']))
    print(f'Output: {out}\nCheckpoint: {args.checkpoint}', flush=True)
    if args.prepare_only:
        print('Configs prepared. No training started; basic file checks only.')
        return
    for task, command in commands:
        print(f'Starting {task}; console: {out / (task + ".console.log")}', flush=True)
        with (out / f'{task}.console.log').open('w') as console:
            subprocess.run(command, cwd=base / 'runtime', env=env, stdout=console,
                           stderr=subprocess.STDOUT, check=True)
    rows = []
    for task in args.tasks:
        logfile = next((out / 'logs').glob(f'*withAdapter({task})_QuickStart.log'))
        text = logfile.read_text()
        best = re.findall(r'Best acc so far: ([\d.]+) \(@task_head_epoch-(\d+)\)', text)[-1]
        rows.append(dict(task=task, best_acc_percent=best[0], best_epoch=best[1], log=str(logfile)))
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Finished. Summary: {out / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
