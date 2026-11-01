"""Opt-in console capture, resolved configuration and launch outcome."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import os
import platform
import subprocess
import sys
import traceback
import uuid
from importlib import metadata as package_metadata

import torch
from trackit.miscellanies.torch.distributed import is_main_process, get_world_size


class _Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, value):
        self.logfile.write(value)
        return self.console.write(value)

    def flush(self):
        self.logfile.flush()
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def _save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


@contextmanager
def record_local_run(args, config, wandb_instance=None):
    if (not config['run'].get('diagnostics', {}).get('enabled', False)
            or args.output_dir is None or not is_main_process()):
        yield
        return
    root = Path(args.output_dir)
    directory = root / 'diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(session_id=uuid.uuid4().hex, run_id=args.run_id,
                    started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    python=platform.python_version(), torch=str(torch.__version__),
                    cuda=torch.version.cuda, world_size=get_world_size(),
                    runtime_args=vars(args).copy(),
                    environment={k: os.environ.get(k) for k in
                                 ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')})
    if str(args.device).startswith('cuda') and torch.cuda.is_available():
        metadata['gpu'] = torch.cuda.get_device_name(torch.device(args.device))
    metadata['packages'] = {}
    for package in ('torch', 'timm', 'wandb', 'numpy', 'opencv-python', 'opencv-python-headless', 'safetensors', 'PyYAML'):
        try:
            metadata['packages'][package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            pass
    if wandb_instance is not None:
        metadata['wandb_url'] = wandb_instance.url
    try:
        metadata['git_revision'] = subprocess.check_output(
            ['git', '-C', args.root_path, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata['git_revision'] = None
    _save_json(directory / 'resolved_config.json', config)
    from .training_provenance import capture_training_provenance
    metadata['provenance'] = capture_training_provenance(args.root_path, directory, config)
    metadata['resolved_config_sha256'] = hashlib.sha256((directory / 'resolved_config.json').read_bytes()).hexdigest()
    if wandb_instance is not None:
        wandb_instance.config.update({'provenance/git_commit': metadata['provenance']['git_commit'],
                                     'provenance/git_branch': metadata['provenance']['git_branch'],
                                     'provenance/source_manifest_sha256': metadata['provenance']['source_manifest_sha256'],
                                     'provenance/resolved_config_sha256': metadata['resolved_config_sha256'],
                                     'provenance/experiment': metadata['provenance'].get('experiment')})
    _save_json(directory / 'run_info.json', metadata)
    with (directory / 'console.log').open('a', encoding='utf-8', buffering=1) as logfile:
        with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
            try:
                yield
            except BaseException as error:
                metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                error_type=type(error).__name__, error_message=str(error))
                traceback.print_exc(file=logfile)
                raise
            else:
                metadata['status'] = 'completed'
            finally:
                metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
                _save_json(directory / 'run_info.json', metadata)
