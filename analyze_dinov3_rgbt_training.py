"""Export sampled batch metrics, epoch means, and stage-specific training curves."""
import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    records = []
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
    return records


def analyze(root):
    root = Path(root)
    destination = root / 'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    summary = {}
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, stage in enumerate(('adapt', 'temporal')):
        directory = root / stage
        rows = read_rows(directory / 'metrics.jsonl')
        epochs = read_rows(directory / 'epoch_summary.jsonl')
        info = json.loads((directory / 'run_info.json').read_text()) if (directory / 'run_info.json').exists() else {}
        summary[stage] = dict(status=info.get('status', 'not_started'), logged_batches=len(rows),
                              completed_epochs=max((row['epoch'] for row in epochs), default=0),
                              latest=rows[-1] if rows else None, reference_space_id=info.get('reference_space_id'),
                              epoch_summaries=epochs)
        if rows:
            keys = sorted({key for row in rows for key in row})
            with (destination / f'{stage}_batches.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            # Batch rows are periodic samples; epoch_summary means include every batch.
            for metric in ('loss', 'Loss/bce', 'Loss/giou', 'Loss/reconstruction',
                           'Loss/future_appearance', 'Loss/future_motion'):
                selected = [row for row in rows if metric in row]
                if selected:
                    axes[0, column].plot([r['optimizer_updates'] for r in selected],
                                         [r[metric] for r in selected], label=metric)
            axes[0, column].legend(fontsize=8)
            axes[1, column].plot([r['optimizer_updates'] for r in rows], [r['grad_norm'] for r in rows], label='grad_norm')
            if any('fusion_gate' in row for row in rows):
                axes[1, column].plot([r['optimizer_updates'] for r in rows], [r.get('fusion_gate', 0.) for r in rows], label='fusion_gate')
            axes[1, column].legend(fontsize=8)
        for row in range(2):
            axes[row, column].set_title(stage + (' losses' if row == 0 else ' diagnostics'))
            axes[row, column].set_xlabel('optimizer updates')
            axes[row, column].grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(destination / 'curves.png', dpi=160)
    plt.close(figure)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    print(json.dumps({stage: {key: result[stage][key] for key in ('status', 'completed_epochs', 'logged_batches')}
                      for stage in result}, indent=2))
