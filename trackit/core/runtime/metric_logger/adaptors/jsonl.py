"""Durable interval statistics and epoch summaries, optionally mirrored to W&B."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid

from ..interface import MetricLoggerInterface


class JsonlMetricLogger(MetricLoggerInterface):
    def __init__(self, output_path, task_name, interval=50, prefix='', wandb_instance=None):
        if interval < 1:
            raise ValueError('JSONL interval must be positive')
        self.output_path = Path(output_path)
        self.task_name = task_name
        self.interval = interval
        self.prefix = prefix or ''
        self.wandb = wandb_instance
        self.session_id = uuid.uuid4().hex
        self.epoch = None
        if self.wandb is not None:
            self.wandb.define_metric('sample_step')
            self.wandb.define_metric('epoch')
            self.wandb.define_metric('train/*', step_metric='sample_step')
            self.wandb.define_metric('val/*', step_metric='epoch')

    def on_epoch_begin(self, epoch, is_train):
        self.epoch = epoch
        self.is_train = is_train
        self._pending = {}
        self._summary = {}
        self._window = {}
        self._epoch_stats = {}
        self._batches = self._window_batches = 0
        self._last_step = self._first_step = None
        self._force = False

    def log(self, meters, force=False, step=None):
        self._pending.update({self.prefix + k: float(v) for k, v in meters.items()})
        self._last_step = step
        self._force |= force

    def log_summary(self, meters):
        self._summary.update({self.prefix + k: float(v) for k, v in meters.items()})

    @staticmethod
    def _accumulate(stats, meters):
        for name, value in meters.items():
            stat = stats.setdefault(name, dict(sum=0., count=0, min=None, max=None, last=None, nonfinite=0))
            stat['last'] = value if math.isfinite(value) else None
            if not math.isfinite(value):
                stat['nonfinite'] += 1
                continue
            stat['sum'] += value
            stat['count'] += 1
            stat['min'] = value if stat['min'] is None else min(stat['min'], value)
            stat['max'] = value if stat['max'] is None else max(stat['max'], value)

    @staticmethod
    def _statistics(stats):
        return {name: dict(mean=s['sum'] / s['count'] if s['count'] else None,
                           count=s['count'], min=s['min'], max=s['max'], last=s['last'],
                           nonfinite=s['nonfinite']) for name, s in stats.items()}

    def _means(self, stats, statistics):
        metrics = {k: s['mean'] for k, s in statistics.items()}
        samples = stats.get(self.prefix + 'timing/samples_in_batch')
        seconds = stats.get(self.prefix + 'timing/batch_seconds')
        if samples and seconds and samples['count'] == seconds['count'] and seconds['sum'] > 0:
            metrics[self.prefix + 'timing/samples_per_second'] = samples['sum'] / seconds['sum']
        return metrics

    def _write(self, filename, record):
        self.output_path.mkdir(parents=True, exist_ok=True)
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), session_id=self.session_id,
                      task=self.task_name, epoch=self.epoch + 1, epoch_index=self.epoch, **record)
        # Close each append so completed records survive an interrupted process.
        with (self.output_path / filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    def _publish(self, record, summary=False):
        if self.wandb is None:
            return
        values = {name: value for name, value in record['metrics'].items() if value is not None}
        values['epoch'] = self.epoch + 1
        if record['sample_step'] is not None:
            values['sample_step'] = record['sample_step']
        # Use W&B's monotonic internal step. Validation shares the training sample
        # counter, so passing it as W&B's explicit step would discard val events.
        self.wandb.log(values)
        if summary:
            self.wandb.summary.update({k: v for k, v in values.items() if k not in ('epoch', 'sample_step')})

    def commit(self):
        if not self._pending:
            return
        if self._window_batches == 0:
            self._first_step = self._last_step
        self._batches += 1
        self._window_batches += 1
        self._accumulate(self._window, self._pending)
        self._accumulate(self._epoch_stats, self._pending)
        nonfinite = {k: None for k, v in self._pending.items() if not math.isfinite(v)}
        if nonfinite:
            self._write('metrics.jsonl', dict(kind='nonfinite', sample_step=self._last_step,
                                            metrics=nonfinite, batch_in_epoch=self._batches))
        self._pending = {}
        if self._window_batches >= self.interval or self._force:
            self._flush_window()
        self._force = False

    def _flush_window(self):
        if not self._window_batches:
            return
        statistics = self._statistics(self._window)
        record = self._write('metrics.jsonl', dict(kind='window', sample_step=self._last_step,
                             sample_step_start=self._first_step, batch_in_epoch=self._batches,
                             batches=self._window_batches, statistics=statistics,
                             metrics=self._means(self._window, statistics)))
        self._publish(record)
        self._window.clear()
        self._window_batches = 0

    def on_epoch_end(self, epoch, is_train):
        self.commit()
        self._flush_window()
        statistics = self._statistics(self._epoch_stats)
        metrics = self._means(self._epoch_stats, statistics)
        metrics.update({k: v if math.isfinite(v) else None for k, v in self._summary.items()})
        record = self._write('epoch_summary.jsonl', dict(kind='epoch', sample_step=self._last_step,
                             batches=self._batches, statistics=statistics, metrics=metrics))
        # Training means get distinct keys; validation retains exact AO/SR names.
        cloud_record = dict(record)
        if self.is_train:
            cloud_record['metrics'] = {k.replace(self.prefix, self.prefix + 'epoch/', 1): v
                                       for k, v in metrics.items()}
        self._publish(cloud_record, summary=True)


def construct_jsonl_metric_logger(dispatcher, config, build_context, output_path, task_name):
    from trackit.miscellanies.torch.distributed import is_main_process
    options = (config or {}).get('jsonl', {})
    if not options.get('enabled', False) or output_path is None or not is_main_process():
        return False
    logger = JsonlMetricLogger(output_path, task_name, options.get('interval', 50),
                              (config or {}).get('metric_prefix', ''),
                              build_context.wandb_instance if options.get('sync_wandb') else None)
    build_context.services.event.register_on_epoch_begin_event_listener(logger.on_epoch_begin)
    build_context.services.event.register_on_epoch_end_event_listener(logger.on_epoch_end)
    dispatcher.register_logger('jsonl', logger, 'external')
    return bool(options.get('sync_wandb'))
