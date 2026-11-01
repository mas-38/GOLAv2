import argparse
import contextlib
import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import time

import cv2
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_arg_parser
from trackit.core.boot.funcs.main.load_config import load_config
from trackit.core.boot.funcs.main.init_application import prepare_application
from trackit.core.boot.workarounds.numpy import numpy_no_multithreading
from trackit.datasets.common.cache_path import set_dataset_attributes_cache_path
from trackit.core.runtime.services.event import emit_epoch_begin_event
from trackit.core.runtime.metric_logger import enable_metric_logger

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--workers', type=int, required=True)
parser.add_argument('--pin', action='store_true')
parser.add_argument('--report', type=Path)
opts = parser.parse_args()
numpy_no_multithreading()
torch.set_num_threads(4)
cv2.setNumThreads(0)
with TemporaryDirectory(prefix='dinov3_5090_tuning_') as temp:
    root = Path(temp)
    from trackit.core.runtime.global_constant import get_global_constant
    source = Path(get_global_constant('GOT10k_PATH'))
    for split, count in [('train', 128), ('val', 1)]:
        target = root / 'dataset' / split
        target.mkdir(parents=True)
        names = (source / split / 'list.txt').read_text().splitlines()[:count]
        for name in names:
            (target / name).symlink_to(source / split / name, target_is_directory=True)
        (target / 'list.txt').write_text('\n'.join(names) + '\n')
    set_dataset_attributes_cache_path(str(root / 'cache'))
    cli = ['DINOv3RGB', 'dinov3', '--mixin_config', 'fast_transfer', '--disable_wandb', '--output_dir', str(root / 'output')]
    if opts.pin:
        cli += ['--pin_memory']
    args = setup_arg_parser().parse_args(cli)
    args.root_path = str(PROJECT_ROOT)
    args.config_path = args.root_path + '/config'
    args.instance_id = 42
    args.run_id = '5090-performance-check-no-optimizer-updates'
    cfg = load_config(args)
    for name in ('train', 'eval'):
        cfg['run']['data'][name]['source']['parameters']['datasets'][0]['path'] = str(root / 'dataset')
    train_cfg = cfg['run']['data']['train']
    train_cfg['global_batch_size'] = opts.batch
    train_cfg['num_workers'] = opts.workers
    train_cfg['sampler']['samples_per_epoch'] = 16384
    with open(f'/tmp/dinov3_5090_b{opts.batch}_setup.log', 'w') as log, contextlib.redirect_stdout(log):
        app = prepare_application(args, cfg, None)
    net = app._model_manager.create(torch.device('cuda')).model.train()
    criterion = app._all_context.runners['train'].runner._criterion
    task = app._all_context.tasks['train']
    data_context = app._all_context.data_inputs['train']
    app._context_manager.activate('train', 0)
    parameters = tuple(p for p in net.parameters() if p.requires_grad)
    def forward_backward(data):
        net.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(net(**data.input), data.target).loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.).item()
        assert torch.isfinite(loss).item() and norm >= 0
    with enable_metric_logger(task.metric_logger, task.local_metric_logger):
        emit_epoch_begin_event((data_context.services_registry.event,), 0, True)
        pipeline = data_context.data_input_pipeline
        iterator = iter(pipeline)
        first = next(iterator)
        for _ in range(5):
            forward_backward(first)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(12):
            forward_backward(first)
        torch.cuda.synchronize()
        cached_seconds = (time.perf_counter() - begin) / 12
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            forward_backward(first)
        attention_ops = [event.key for event in profile.key_averages() if 'attention' in event.key]
        for _ in range(16):
            forward_backward(next(iterator))
        torch.cuda.synchronize()
        waits, durations = [], []
        begin = time.perf_counter()
        for _ in range(64):
            start = time.perf_counter()
            data = next(iterator)
            ready = time.perf_counter()
            forward_backward(data)
            waits.append(ready - start)
            durations.append(time.perf_counter() - start)
        torch.cuda.synchronize()
        measured_seconds = time.perf_counter() - begin
        if hasattr(pipeline, 'iter') and hasattr(pipeline.iter, '_shutdown_workers'):
            pipeline.iter._shutdown_workers()
    report = dict(batch=opts.batch, workers=opts.workers, pin_memory=opts.pin, opencv_threads=cv2.getNumThreads(),
                  rank=32, cached_seconds_per_batch=cached_seconds, cached_samples_per_second=opts.batch / cached_seconds,
                  measured_batches=64, end_to_end_seconds_per_batch=measured_seconds / 64,
                  end_to_end_samples_per_second=64 * opts.batch / measured_seconds,
                  data_wait_mean_seconds=statistics.mean(waits), batch_p95_seconds=sorted(durations)[60],
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  attention_ops=attention_ops, optimizer_steps=0)
    if opts.report:
        opts.report.parent.mkdir(parents=True, exist_ok=True)
        opts.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
