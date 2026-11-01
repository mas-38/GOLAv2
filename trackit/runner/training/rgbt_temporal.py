"""Two-stage, single-GPU fine-tuning with complete, atomic resume snapshots."""
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from trackit.data.methods.rgbt_temporal import LasHeRTemporalClips, collate_clips, manifest_digest
from trackit.models.methods.dinov3_missing.checkpoint import (
    build_model, read_checkpoint, save_checkpoint, restore_training_state, training_signature)
from trackit.models.methods.dinov3_missing.pipeline import forward_clip, MissingTrackingLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {section: {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                      for key, value in values.items()}
            for section, values in batch.items()}


def make_optimization(model, stage_config, *, device, steps_per_epoch):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        multiplier = stage_config['head_lr_multiplier'] if name.startswith('tracker.head.') else 1.
        decay = 0. if parameter.ndim <= 1 or 'modality_embed' in name or 'role_embed' in name else stage_config['weight_decay']
        groups.setdefault((multiplier, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=parameters, lr=stage_config['lr'] * multiplier, weight_decay=decay)
                                    for (multiplier, decay), parameters in groups.items()], fused=str(device).startswith('cuda'))
    total = stage_config['epochs'] * steps_per_epoch
    warmup = stage_config['warmup_epochs'] * steps_per_epoch
    minimum_ratio = stage_config['min_lr'] / stage_config['lr']
    def factor(step):
        if warmup and step < warmup:
            return .001 + .999 * step / warmup
        progress = min(1., max(0., (step - warmup) / max(1, total - warmup)))
        return minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    return optimizer, scheduler


def update_parameters(loss, model, optimizer, scheduler, scaler, max_grad_norm, require_reconstruction=False):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss; optimizer was not updated')
    scaler.scale(loss).backward()
    if model.stage == 'temporal':
        required = [model.future_predictor.condition.project.weight, model.future_predictor.appearance.weight]
        if require_reconstruction:
            required.extend([model.reconstructor.initial.weight, model.reconstructor.readout[1].weight,
                             model.fusion.output.weight])
        if any(parameter.grad is None for parameter in required):
            raise RuntimeError('A trainable temporal projection has no gradient; optimizer was not updated')
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradients; optimizer was not updated')
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    if updated:
        scheduler.step()
    return float(norm), updated


def autocast_context(device, precision):
    if not str(device).startswith('cuda') or precision == 'float32':
        return nullcontext()
    return torch.autocast('cuda', dtype={'bfloat16': torch.bfloat16, 'float16': torch.float16}[precision])


def train_stage(config, stage, output, *, initialization_checkpoint=None, resume=None, device='cuda'):
    settings = config['stages'][stage]
    if settings['samples_per_epoch'] % settings['batch_size']:
        raise ValueError('samples_per_epoch must be divisible by batch_size')
    seed_everything(config['seed'])
    data = LasHeRTemporalClips(config['data']['root'], stage=stage, samples=settings['samples_per_epoch'],
                              seed=config['seed'], history_frames=config['data']['history_frames'],
                              template_hw=tuple(v * 16 for v in reversed(config['model']['template_feat_size'])),
                              search_hw=tuple(v * 16 for v in reversed(config['model']['search_feat_size'])),
                              horizons=config['model']['temporal']['horizons'])
    signature = training_signature(config, stage, manifest_digest(data.names))
    checkpoint = read_checkpoint(resume) if resume else None
    if checkpoint is not None and checkpoint['stage'] != stage:
        raise ValueError('Resume stage differs from the selected stage')
    model = build_model(config, stage, resume or initialization_checkpoint, device).train()
    criterion = MissingTrackingLoss(**config['loss'])
    steps_per_epoch = settings['samples_per_epoch'] // settings['batch_size']
    optimizer, scheduler = make_optimization(model, settings, device=device, steps_per_epoch=steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda') and config['runtime']['precision'] == 'float16')
    epoch_start, batch_start, updates = 0, 0, 0
    if checkpoint is not None:
        epoch_start, batch_start, updates = restore_training_state(checkpoint, optimizer, scheduler, scaler, signature)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous_info = json.loads((output / 'run_info.json').read_text()) if (output / 'run_info.json').is_file() else {}
    info = dict(stage=stage, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                train_sequences=len(data.names), training_signature=signature, configuration=config,
                reference_space_id=model.reference.space_id if model.reference else None,
                adaptation_source=model.adaptation_source, resume=str(resume) if resume else None,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
    wandb_run = None
    try:
        if config['wandb']['enabled']:
            import wandb
            info['wandb_run_id'] = (previous_info.get('wandb_run_id') if resume else None) or wandb.util.generate_id()
            (output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
            wandb_run = wandb.init(entity=config['wandb']['entity'], project=config['wandb']['project'],
                                   name=f'{output.parent.name}-{stage}', job_type='finetune',
                                   id=info['wandb_run_id'], resume='allow' if resume else 'never', group=output.parent.name,
                                   config=info, dir=str(output), tags=['DINOv3', 'RGBT', stage, config['version']])
            info['wandb_url'] = wandb_run.url
        with (output / 'metrics.jsonl').open('a', buffering=1) as log:
            for epoch in range(epoch_start, settings['epochs']):
                data.epoch = epoch
                first = batch_start if epoch == epoch_start else 0
                epoch_statistics = ((checkpoint.get('epoch_statistics') or dict(count=0, sums={}))
                                    if checkpoint is not None and first else dict(count=0, sums={}))
                # An independent generator prevents worker creation from consuming model/dropout RNG.
                generator = torch.Generator().manual_seed(config['seed'] + epoch)
                loader = DataLoader(Subset(data, range(first * settings['batch_size'], len(data))),
                                    batch_size=settings['batch_size'], shuffle=False, drop_last=True,
                                    num_workers=config['data']['workers'], pin_memory=str(device).startswith('cuda'),
                                    collate_fn=collate_clips, generator=generator)
                ready = time.perf_counter()
                for batch_index, batch in enumerate(loader, start=first):
                    started = time.perf_counter()
                    data_wait = started - ready
                    info['last_batch'] = dict(epoch=epoch + 1, batch=batch_index + 1,
                                              sequences=batch['metadata']['sequences'],
                                              frame_indices=batch['inputs']['frame_indices'].tolist())
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(device, config['runtime']['precision']):
                        result = forward_clip(model, batch['inputs'], config['runtime']['train_reconstruction_steps'])
                        loss, metrics = criterion(model, result, batch['targets'])
                    norm, updated = update_parameters(loss, model, optimizer, scheduler, scaler,
                                                      config['runtime']['max_grad_norm'],
                                                      require_reconstruction=any(sample['reconstruction'] for sample in result['samples']))
                    updates += int(updated)
                    flags = batch['inputs']['availability'][:, -1]
                    metrics.update(epoch=epoch + 1, batch=batch_index + 1, optimizer_updates=updates,
                                   grad_norm=norm, lr=optimizer.param_groups[0]['lr'], optimizer_step_applied=updated,
                                   lr_min=min(group['lr'] for group in optimizer.param_groups),
                                   lr_max=max(group['lr'] for group in optimizer.param_groups),
                                   rgb_only=int((flags[:, 0] & ~flags[:, 1]).sum()),
                                   tir_only=int((~flags[:, 0] & flags[:, 1]).sum()),
                                   complete=int(flags.all(1).sum()),
                                   data_wait_seconds=data_wait,
                                   batch_seconds=time.perf_counter() - started)
                    if str(device).startswith('cuda'):
                        metrics.update(gpu_memory_mib=torch.cuda.memory_allocated() / 2**20,
                                       gpu_peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
                    if stage == 'temporal':
                        for module in ('future_predictor', 'reconstructor', 'fusion'):
                            metrics[f'gradient_tensors/{module}'] = sum(
                                parameter.grad is not None for parameter in getattr(model, module).parameters())
                    epoch_statistics['count'] += 1
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)) and key not in ('epoch', 'batch', 'optimizer_updates'):
                            epoch_statistics['sums'][key] = epoch_statistics['sums'].get(key, 0.) + value
                    if (batch_index + 1) % config['runtime']['log_interval'] == 0 or batch_index == first:
                        metrics['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                        metrics['sequences'] = info['last_batch']['sequences']
                        metrics['frame_indices'] = info['last_batch']['frame_indices']
                        log.write(json.dumps(metrics, allow_nan=False) + '\n')
                        print(json.dumps(dict(stage=stage, **metrics), allow_nan=False), flush=True)
                        if wandb_run is not None:
                            wandb_run.log({f'{stage}/{key}': value for key, value in metrics.items() if isinstance(value, (int, float))},
                                          step=epoch * steps_per_epoch + batch_index)
                    if (batch_index + 1) % config['runtime']['checkpoint_interval'] == 0:
                        save_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler, config=config,
                                        epoch=epoch, next_batch=batch_index + 1, optimizer_updates=updates, signature=signature,
                                        statistics=epoch_statistics)
                    ready = time.perf_counter()
                count = epoch_statistics['count']
                epoch_summary = dict(epoch=epoch + 1, batches=count, optimizer_updates=updates,
                                     means={key: value / max(1, count) for key, value in epoch_statistics['sums'].items()})
                with (output / 'epoch_summary.jsonl').open('a') as stream:
                    stream.write(json.dumps(epoch_summary, allow_nan=False) + '\n')
                save_checkpoint(output / 'last.pt', model, optimizer, scheduler, scaler, config=config,
                                epoch=epoch + 1, next_batch=0, optimizer_updates=updates, signature=signature)
                # Retain the exact stage endpoint and a short trailing history.
                snapshot = output / f'epoch_{epoch + 1:02d}.pt'
                import shutil
                shutil.copy2(output / 'last.pt', snapshot)
                snapshots = sorted(output.glob('epoch_*.pt'))
                for old in snapshots[:-3]:
                    old.unlink()
                (output / 'progress.json').write_text(json.dumps(dict(completed_epochs=epoch + 1,
                    target_epochs=settings['epochs'], optimizer_updates=updates)) + '\n')
        info.update(status='completed', optimizer_updates=updates, selected_checkpoint='last.pt',
                    selection='Fixed final epoch; no test-set model selection')
    except BaseException as error:
        info.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                    error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'run_info.json').write_text(json.dumps(info, indent=2) + '\n')
        if wandb_run is not None:
            wandb_run.summary.update(dict(status=info['status'], optimizer_updates=updates))
            wandb_run.finish(exit_code=0 if info['status'] == 'completed' else 1)
    return output / 'last.pt'
