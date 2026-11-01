"""Full-size forward/backward checks; never constructs an optimizer or updates weights."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
import yaml

from trackit.data.methods.rgbt_temporal import LasHeRTemporalClips, collate_clips
from trackit.models.methods.dinov3_missing.checkpoint import build_model
from trackit.models.methods.dinov3_missing.pipeline import forward_clip, MissingTrackingLoss
from trackit.runner.training.rgbt_temporal import move_batch, autocast_context, seed_everything


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/DINOv3Missing/temporal_v1.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/dinov3_rgbt_temporal/preflight_v1/batch_gradient_check.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--adapt-batch', type=int)
    parser.add_argument('--temporal-batch', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config['runtime']['torch_threads'])
    cv2.setNumThreads(0)
    seed_everything(config['seed'])
    model = build_model(config, device=args.device).train()
    criterion = MissingTrackingLoss(**config['loss'])
    reports = []
    for stage in ('adapt', 'temporal'):
        if stage == 'temporal':
            model.zero_grad(set_to_none=True)
            model.activate_temporal('structural-gradient-check-only:unadapted:no-training-checkpoint')
            model.to(args.device).train()
        batch_size = ((args.adapt_batch if stage == 'adapt' else args.temporal_batch)
                      or config['stages'][stage]['batch_size'])
        data = LasHeRTemporalClips(config['data']['root'], stage=stage, samples=1,
                                  history_frames=config['data']['history_frames'], seed=config['seed'])
        # Repeating one real clip tests the configured kernel/memory shape, not sampling quality.
        sample = data[0]
        batch = move_batch(collate_clips([sample] * batch_size), args.device)
        batch['inputs']['context'][:, -1] = batch['targets']['current_image'].clone()
        if stage == 'adapt':
            # Maximum number of joint tokens for the first stage.
            batch['inputs']['availability'][:, -1] = True
        else:
            # Every sample exercises all five reconstruction iterations.
            batch['inputs']['availability'][:, -1] = torch.tensor([True, False], device=args.device)
            batch['inputs']['context'][:, -1, 3:] = float('nan')
        model.zero_grad(set_to_none=True)
        if str(args.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context(args.device, config['runtime']['precision']):
            output = forward_clip(model, batch['inputs'], config['runtime']['train_reconstruction_steps'])
            loss, _ = criterion(model, output, batch['targets'])
        loss.backward()
        if str(args.device).startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        module_gradients = {}
        if stage == 'temporal':
            for name in ('future_predictor', 'reconstructor', 'fusion'):
                parameters = list(getattr(model, name).parameters())
                module_gradients[name] = dict(total=len(parameters), with_gradient=sum(p.grad is not None for p in parameters))
            for parameter in (model.future_predictor.condition.project.weight, model.future_predictor.appearance.weight,
                              model.reconstructor.initial.weight, model.reconstructor.readout[1].weight,
                              model.fusion.output.weight):
                assert parameter.grad is not None, 'Missing trainable temporal projection gradient'
        report = dict(stage=stage, batch_size=batch_size, loss=float(loss.detach()),
                      forward_backward_seconds=elapsed, gradient_tensors=len(gradients),
                      all_gradients_finite=True, frozen_parameters_without_gradients=True,
                      module_gradients=module_gradients, optimizer_updates=0, reference_is_adapted=False)
        if str(args.device).startswith('cuda'):
            report['peak_gpu_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
        reports.append(report)
        print(json.dumps(report), flush=True)
        del output, loss, batch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(training_started=False, checks=reports,
        note='Real-clip repeated batches, forward/backward only. No optimizer constructed, parameters updated, or model checkpoint exported. Timings are cold checks, not training ETA.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
