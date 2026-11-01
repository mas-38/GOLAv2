"""Evaluate a trained RGB-T checkpoint on LasHeR test with causal modality routing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import yaml

from trackit.data.methods.rgbt_temporal import LasHeRSequence, availability_schedule, split_names
from trackit.data.methods.siamese_tracker_eval.transform.missing import MissingModalitySchedule
from trackit.data.components.result_collector.handler.one_pass_evaluation.ope_metrics import (
    compute_one_pass_evaluation_metrics, compute_OPE_metrics_mean)
from trackit.models.methods.dinov3_missing.checkpoint import build_model, read_checkpoint, file_sha256
from trackit.models.methods.dinov3_missing.runtime import RGBTSequenceTracker
from trackit.runner.training.rgbt_temporal import autocast_context


def report_metrics(metrics):
    return dict(AO=metrics.average_overlap, SR50=metrics.success_rate_at_iou_0_5,
                SR75=metrics.success_rate_at_iou_0_75, success_auc=metrics.success_score,
                precision20=metrics.precision_score, normalized_precision20=metrics.normalized_precision_score,
                fps=metrics.get_fps(), success_curve=metrics.success_curve.tolist(),
                precision_curve=metrics.precision_curve.tolist(),
                normalized_precision_curve=metrics.normalized_precision_curve.tolist())


def evaluate(config, checkpoint, output, modes, *, device='cuda', schedule=None, limit=None):
    saved = read_checkpoint(checkpoint)
    model = build_model(config, saved['stage'], checkpoint, device).eval()
    names = split_names(config['data']['root'], 'test')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        names = names[:limit]
    schedule_reader = MissingModalitySchedule(schedule) if schedule else None
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                  stage=saved['stage'], split='LasHeR-test', sequences=len(names), training_sequences=979,
                  protocol='Repository OPE: sequence mean, 21-point IoU success curve, PR@20px, NPR@0.2; initialization frame included.',
                  initialization='Both real modalities at frame zero, matching the existing SCDT protocol.',
                  synthetic_note='synthetic_missing is a reproducible three-state Markov schedule, not an official SCDT marker file.',
                  test_used_for_checkpoint_selection=False, modes={})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        if mode == 'schedule' and schedule_reader is None:
            raise ValueError('schedule mode requires --missing-mask-file')
        metrics_list, details = [], []
        for name in names:
            sequence = LasHeRSequence(config['data']['root'], name, 'test')
            frame_count = len(sequence.rgb)
            seed = config['seed'] + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            synthetic = availability_schedule(frame_count, np.random.default_rng(seed),
                                               config['evaluation']['synthetic_missing_persistence']).numpy()
            tracker = RGBTSequenceTracker(model, reconstruction_steps=config['runtime']['inference_reconstruction_steps'],
                                           template_threshold=config['runtime']['template_update_threshold'])
            predictions, times, state_log = [], [], []
            with torch.no_grad(), autocast_context(device, config['runtime']['precision']):
                for frame in range(frame_count):
                    if frame == 0 or mode == 'complete':
                        available = (True, True)
                    elif mode == 'rgb_only':
                        available = (True, False)
                    elif mode == 'tir_only':
                        available = (False, True)
                    elif mode == 'synthetic_missing':
                        available = tuple(map(bool, synthetic[frame]))
                    elif mode == 'schedule':
                        available = schedule_reader.get(name, frame)
                    else:
                        raise ValueError(mode)
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    # Missing image files are never opened, including both-missing frames.
                    image = sequence.image(frame, available)
                    if frame == 0:
                        tracker.initialize(image, sequence.boxes[0], frame_index=0, track_id=name)
                        box, confidence, diagnostics = tracker.pixel_box(), 1., dict(initial=True)
                    else:
                        box, confidence, diagnostics = tracker.track(image, frame, available)
                    if str(device).startswith('cuda'):
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - started)
                    predictions.append(box)
                    state_log.append(dict(frame=frame, confidence=confidence, **{'available': list(available), **diagnostics}))
            predictions = np.asarray(predictions, dtype=np.float64)
            if predictions.shape != sequence.boxes.shape or not np.isfinite(predictions).all():
                raise ValueError(f'{name}: invalid/incomplete predictions')
            metric, _ = compute_one_pass_evaluation_metrics(predictions, sequence.boxes.astype(np.float64),
                                                            None, np.asarray(times))
            metrics_list.append(metric)
            destination = output / mode / name
            destination.mkdir(parents=True, exist_ok=True)
            xywh = predictions.copy()
            xywh[:, 2:] -= xywh[:, :2]
            np.savetxt(destination / f'{name}.txt', xywh, fmt='%.4f', delimiter=',')
            np.savetxt(destination / 'time.txt', times, fmt='%.8f')
            with (destination / 'diagnostics.jsonl').open('w') as stream:
                for row in state_log:
                    # numpy.bool_ is not JSON serializable.
                    row['available'] = list(map(bool, row['available']))
                    stream.write(json.dumps(row) + '\n')
            detail = dict(sequence=name, frames=frame_count, **report_metrics(metric))
            details.append(detail)
            print(json.dumps(dict(mode=mode, completed=len(details), total=len(names), sequence=name,
                                  AO=metric.average_overlap, success_auc=metric.success_score)), flush=True)
        report['modes'][mode] = dict(summary=report_metrics(compute_OPE_metrics_mean(metrics_list)), sequences=details)
        report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--config', type=Path, default=root / 'config/DINOv3Missing/temporal_v1.yaml')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', action='append', choices=['complete', 'rgb_only', 'tir_only', 'synthetic_missing', 'schedule'])
    parser.add_argument('--missing-mask-file', type=Path)
    parser.add_argument('--limit-sequences', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(config['runtime']['torch_threads'])
    cv2.setNumThreads(0)
    if not args.execute:
        print(json.dumps(dict(evaluation_started=False, test_sequences=len(split_names(config['data']['root'], 'test')),
                              checkpoint_exists=args.checkpoint.is_file(), modes=args.mode or config['evaluation']['modes'])))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose a fresh output directory to retain prior evaluation results')
    evaluate(config, args.checkpoint, args.output, args.mode or config['evaluation']['modes'], device=args.device,
             schedule=args.missing_mask_file, limit=args.limit_sequences)


if __name__ == '__main__':
    main()
