"""Summarize a running/completed RGB training run and plot its local metrics."""
import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path):
    if not path.is_file():
        return []
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A live writer may not have finished its final record yet.
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
    return records


def write_csv(path, records):
    names = sorted({key for record in records for key in record.get('metrics', {})})
    fixed = ['epoch', 'epoch_index', 'sample_step', 'batch_in_epoch', 'batches']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fixed + names)
        writer.writeheader()
        for record in records:
            writer.writerow({**{key: record.get(key) for key in fixed}, **record.get('metrics', {})})


def analyze(run_dir, plot=True):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    train_records = read_jsonl(run_dir / 'train/metrics.jsonl')
    windows = [r for r in train_records if r.get('kind') == 'window']
    train_epochs = read_jsonl(run_dir / 'train/epoch_summary.jsonl')
    validation = read_jsonl(run_dir / 'eval/epoch_summary.jsonl')
    ao_name = 'val/average_overlap_GOT-10k-val'
    scored = [r for r in validation if r['metrics'].get(ao_name) is not None]
    best = max(scored, key=lambda r: r['metrics'][ao_name]) if scored else None
    info_path = run_dir / 'diagnostics/run_info.json'
    config_path = run_dir / 'diagnostics/resolved_config.json'
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    summary = dict(status=info.get('status', 'unknown'),
                   completed_train_epochs=sorted({r['epoch'] for r in train_epochs}),
                   latest_train_window=windows[-1] if windows else None,
                   best_validation=best, validation_count=len(scored),
                   nonfinite_training_values=sum(s['nonfinite'] for r in windows
                                                 for s in r.get('statistics', {}).values()),
                   note='Training positive IoU uses labeled positive patches; closed-loop tracking quality uses GOT-10k validation AO.')
    output = run_dir / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    write_csv(output / 'train_windows.csv', windows)
    write_csv(output / 'train_epochs.csv', train_epochs)
    write_csv(output / 'validation.csv', validation)
    if plot and (windows or validation):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = config.get('run', {}).get('data', {}).get('train', {})
        steps_per_epoch = data.get('sampler', {}).get('samples_per_epoch', 131072) / data.get('global_batch_size', 16)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        panels = [
            ('Training loss', ['train/loss', 'train/Loss/bce', 'train/Loss/iou']),
            ('Positive patch IoU / confidence', ['train/diagnostics/positive_iou', 'train/diagnostics/foreground_confidence', 'train/diagnostics/background_confidence']),
            ('Learning rate', ['train/lr']),
            ('Gradient norm before clipping', ['train/grad_norm']),
            ('Throughput / data wait', ['train/timing/samples_per_second']),
            ('GOT-10k validation', [ao_name, 'val/success_rate_at_iou_0_5_GOT-10k-val']),
        ]
        for ax, (title, names) in zip(axes.flat, panels):
            records = validation if names[0].startswith('val/') else windows
            for name in names:
                points = [(r['epoch'] if records is validation else r['epoch_index'] + r['batch_in_epoch'] / steps_per_epoch,
                           r['metrics'][name]) for r in records if r['metrics'].get(name) is not None]
                if points:
                    x, y = zip(*points)
                    label = 'GIoU loss' if name == 'train/Loss/iou' else name.split('/')[-1]
                    ax.plot(x, y, label=label, linewidth=1.3, marker='o' if records is validation else None)
            ax.set(title=title, xlabel='Epoch')
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(fontsize=8)
        timing_ax = axes[1, 1]
        wait_points = [(r['epoch_index'] + r['batch_in_epoch'] / steps_per_epoch,
                        r['metrics']['train/timing/data_wait_seconds']) for r in windows
                       if r['metrics'].get('train/timing/data_wait_seconds') is not None]
        if wait_points:
            wait_ax = timing_ax.twinx()
            wait_ax.plot(*zip(*wait_points), color='tab:orange', alpha=.6, label='data wait (s)')
            wait_ax.set_ylabel('Data wait (s)')
        fig.savefig(output / 'curves.png', dpi=150)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    summary = analyze(args.run_dir, not args.no_plot)
    best = summary['best_validation']
    print(json.dumps(dict(status=summary['status'], completed_train_epochs=summary['completed_train_epochs'],
                          validation_count=summary['validation_count'],
                          best_epoch=best['epoch'] if best else None,
                          best_ao=best['metrics']['val/average_overlap_GOT-10k-val'] if best else None,
                          output=str(args.run_dir / 'analysis')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
