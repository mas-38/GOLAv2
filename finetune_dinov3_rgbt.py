"""Prepare RGB-T fine-tuning; training requires the explicit --execute flag."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import torch
import yaml

from trackit.data.methods.rgbt_temporal import LasHeRTemporalClips, collate_clips, split_names, manifest_digest, audit_lasher
from trackit.models.methods.dinov3_missing.checkpoint import build_model, read_checkpoint, file_sha256
from trackit.models.methods.dinov3_missing.pipeline import forward_clip, MissingTrackingLoss
from trackit.runner.training.rgbt_temporal import train_stage, move_batch, autocast_context, seed_everything


ROOT = Path(__file__).resolve().parent


def assert_committed_sources():
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    pending = [path for path in changed + untracked if path != 'consts.yaml']
    if pending:
        raise RuntimeError('Commit the fine-tuning version before training: ' + ', '.join(pending))


def preflight(config, output, device, adaptation_checkpoint=None):
    train, test = split_names(config['data']['root'], 'train'), split_names(config['data']['root'], 'test')
    if config['data']['train_split'] != 'train' or config['data']['test_split'] != 'test':
        raise ValueError('Use the official 979 train / 245 test splits')
    layout = audit_lasher(config['data']['root'])
    seed_everything(config['seed'])
    model = build_model(config, 'adapt', device=device).eval()
    criterion = MissingTrackingLoss(**config['loss'])
    reports = []
    for stage in ('adapt', 'temporal'):
        data = LasHeRTemporalClips(config['data']['root'], stage=stage, samples=1, seed=config['seed'],
                                  template_hw=tuple(v * 16 for v in reversed(config['model']['template_feat_size'])),
                                  search_hw=tuple(v * 16 for v in reversed(config['model']['search_feat_size'])),
                                  horizons=config['model']['temporal']['horizons'],
                                  history_frames=config['data']['history_frames'])
        sample = data[0]
        batch = move_batch(collate_clips([sample]), device)
        if stage == 'temporal':
            if adaptation_checkpoint:
                del model
                model = build_model(config, 'temporal', adaptation_checkpoint, device).eval()
            else:
                # Structural smoke check only. This snapshot is never saved as a trainable teacher.
                model.activate_temporal('preflight-only:unadapted-tir:no-training-checkpoint')
                model.to(device).eval()
            batch['inputs']['availability'][:, -1] = torch.tensor([True, False], device=device)
            batch['inputs']['context'][:, -1] = batch['targets']['current_image'].clone()
            batch['inputs']['context'][:, -1, 3:] = float('nan')
        with torch.no_grad(), autocast_context(device, config['runtime']['precision']):
            result = forward_clip(model, batch['inputs'], config['runtime']['train_reconstruction_steps'])
            loss, metrics = criterion(model, result, batch['targets'])
        if not torch.isfinite(loss) or not torch.isfinite(result['boxes']).all():
            raise FloatingPointError('Nonfinite full-size preflight output')
        reports.append(dict(stage=stage, sequence=sample['sequence'], loss=float(loss),
                            finite_outputs=True, current_frame=int(batch['inputs']['frame_indices'][0, -1]),
                            future_frames=batch['targets']['future_indices'][0].tolist(),
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            reference_is_adapted=stage == 'temporal' and adaptation_checkpoint is not None,
                            metrics=metrics))
    report = dict(status='preflight_passed', training_started=False, optimizer_updates=0,
                  training_sequences=len(train), test_sequences=len(test), train_test_overlap=len(set(train) & set(test)),
                  train_manifest_sha256=manifest_digest(train), test_manifest_sha256=manifest_digest(test),
                  initialization_sha256=file_sha256(config['model']['initialization']), source_revision=config['source_revision'],
                  data_layout=layout,
                  device=str(device), checks=reports,
                  temporal_training_dependency='A completed RGB-T adaptation checkpoint; never copied RGB-only TIR weights',
                  note='Losses here only check numerical execution, not tracking accuracy or learned reconstruction quality.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/DINOv3Missing/temporal_v1.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/dinov3_rgbt_temporal/v1')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=['all', 'adapt', 'temporal'], default='all')
    parser.add_argument('--adaptation-checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--execute', action='store_true', help='Actually start training; default is preflight only')
    parser.add_argument('--disable-wandb', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.disable_wandb:
        config['wandb']['enabled'] = False
    torch.set_num_threads(config['runtime']['torch_threads'])
    cv2.setNumThreads(0)
    if not args.execute:
        preflight(config, args.output, args.device, args.adaptation_checkpoint)
        return
    assert_committed_sources()
    stages = ['adapt', 'temporal'] if args.stage == 'all' else [args.stage]
    resume_stage = read_checkpoint(args.resume)['stage'] if args.resume else None
    if resume_stage and resume_stage not in stages:
        raise ValueError('Resume stage differs from requested stages')
    if resume_stage == 'temporal':
        stages = ['temporal']
    if stages == ['temporal'] and not (args.adaptation_checkpoint or args.resume):
        raise ValueError('Temporal stage requires --adaptation-checkpoint or a complete --resume snapshot')
    for stage in stages:
        if (args.output / stage / 'last.pt').exists() and resume_stage != stage:
            raise FileExistsError('Run directory already contains a stage checkpoint; choose a new --output or --resume')
    args.output.mkdir(parents=True, exist_ok=True)
    launch = dict(created_utc=datetime.now(timezone.utc).isoformat(), training_started=True,
                  command_args=vars(args), configuration=config, stages=stages)
    (args.output / 'launch.json').write_text(json.dumps(launch, default=str, indent=2) + '\n')
    adaptation = args.adaptation_checkpoint
    for stage in stages:
        result = train_stage(config, stage, args.output / stage, initialization_checkpoint=adaptation,
                             resume=args.resume if resume_stage == stage else None, device=args.device)
        if stage == 'adapt':
            adaptation = result
    print(json.dumps(dict(status='completed', stages=stages, selected_checkpoint=str(result),
                          test_evaluation='Run evaluate_dinov3_rgbt.py explicitly on the 245 test sequences.')))


if __name__ == '__main__':
    main()
