"""Submit the versioned RGB experiment in tmux after committing its code."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    # These existing edits concern host paths and the separate MMLoRAT work.
    # Runtime provenance preserves their exact diff and content fingerprints.
    allowed = {'consts.yaml', 'trackit/core/transforms/dataset_norm_stats.py',
               'trackit/models/methods/dinov2_missing/builder.py',
               'trackit/models/methods/dinov2_missing/model.py'}
    changed = set(git('diff', '--name-only', 'HEAD').splitlines())
    if changed - allowed:
        raise SystemExit('Commit the experiment changes before submission: ' + ', '.join(sorted(changed - allowed)))
    if subprocess.run(['git', '-C', str(root), 'cat-file', '-e',
                       'HEAD:experiments/dinov3_rgb_got10k/v1/manifest.yaml'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL).returncode:
        raise SystemExit('The experiment manifest must be committed before submission')
    run_id = args.run_id or f'dinov3rgb_v1_{commit[:8]}_b64_e20'
    session = f'dinov3_rgb_v1_{commit[:8]}'
    output_root = root / 'output/dinov3_rgb_got10k'
    output = output_root / run_id
    if output.exists():
        raise SystemExit('Run directory already exists: ' + str(output))
    if subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode == 0:
        raise SystemExit('tmux session already exists: ' + session)
    argv = [sys.executable, '-u', str(root / 'main.py'), 'DINOv3RGB', 'dinov3',
            '--mixin_config', 'fast_transfer', '--mixin_config', 'rtx5090', '--device', 'cuda',
            '--pin_memory', '--seed', '42', '--instance_id', '42',
            '--run_id', run_id, '--output_dir', str(output_root)]
    environment = dict(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', WANDB_MODE='online',
                       WANDB_RESUME='never', PYTHONUNBUFFERED='1')
    output.mkdir(parents=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_id=run_id, session=session,
                    git_commit=commit, git_branch=git('branch', '--show-current'), command=argv,
                    environment=environment, experiment_manifest='experiments/dinov3_rgb_got10k/v1/manifest.yaml')
    (output / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
    runner = output / 'run_process.py'
    runner.write_text('import json, os, subprocess\nfrom pathlib import Path\n'
                      f'root = Path({str(root)!r})\noutput = Path({str(output)!r})\n'
                      f'env = dict(os.environ, **{environment!r})\n'
                      'code = 1\n'
                      'try:\n'
                      "    with (output / 'bootstrap.log').open('a', buffering=1) as log:\n"
                      f'        code = subprocess.run({argv!r}, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode\n'
                      'except KeyboardInterrupt:\n    code = 130\n'
                      'finally:\n'
                      "    (output / 'exit_status.json').write_text(json.dumps(dict(exit_code=code)) + '\\n')\n"
                      'raise SystemExit(code)\n')
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(root),
                     shlex.join([sys.executable, str(runner)])], check=True)
    print(json.dumps(dict(submitted=True, **manifest, output=str(output)), indent=2))


if __name__ == '__main__':
    main()
