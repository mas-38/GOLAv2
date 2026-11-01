"""Fingerprint the code, configuration, pretrained weights and dataset manifests."""
import hashlib
from pathlib import Path
import subprocess
import json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture_training_provenance(root, output, config):
    root, output = Path(root).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)

    result = dict(git_commit=git('rev-parse', 'HEAD').decode().strip(),
                  git_branch=git('branch', '--show-current').decode().strip(),
                  git_status=git('status', '--porcelain=v1').decode().splitlines())
    # Preserve local environment/pre-existing edits instead of pretending that
    # every file in the shared checkout is represented by the commit alone.
    patch = git('diff', 'HEAD', '--', '*.py', '*.yaml', '*.yml', '*.json', '*.md')
    (output / 'working_tree.patch').write_bytes(patch)
    result['working_tree_patch_sha256'] = hashlib.sha256(patch).hexdigest()
    files = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    suffixes = {'.py', '.yaml', '.yml', '.json', '.md', '.sh'}
    manifest = {name: sha256_file(root / name) for name in sorted(set(files))
                if name and (root / name).is_file() and (root / name).suffix in suffixes}
    manifest_path = output / 'source_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    result['source_manifest_sha256'] = sha256_file(manifest_path)
    experiment = root / 'experiments/dinov3_rgb_got10k/v1/manifest.yaml'
    if config.get('type') == 'DINOv3RGB' and experiment.is_file():
        import yaml
        specification = yaml.safe_load(experiment.read_text())
        result['experiment'] = dict(id=specification['experiment_id'], version=specification['version'],
                                    manifest_sha256=sha256_file(experiment))
    if (root / 'consts.yaml').is_file():
        (output / 'host_consts.yaml').write_bytes((root / 'consts.yaml').read_bytes())
        result['host_consts_sha256'] = sha256_file(root / 'consts.yaml')
    if config.get('type') == 'DINOv3RGB':
        from trackit.core.runtime.global_constant import get_global_constant
        backbone = config['model']['backbone']['parameters']
        weights = (root / backbone['weights']).resolve()
        source = (root / backbone['source_dir']).resolve()
        result['backbone'] = dict(weights_path=str(weights), weights_sha256=sha256_file(weights),
                                  source_path=str(source), expected_revision=backbone['expected_revision'],
                                  actual_revision=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                                  source_status=subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain=v1'], text=True).splitlines())
        data_root = Path(get_global_constant('GOT10k_PATH')).expanduser().resolve()
        result['data'] = dict(root=str(data_root), manifests={})
        for split in ('train', 'val'):
            path = data_root / split / 'list.txt'
            result['data']['manifests'][split] = dict(sha256=sha256_file(path),
                       sequences=len([line for line in path.read_text().splitlines() if line.strip()]))
        result['data']['fingerprint_scope'] = 'Official split list.txt files; image contents are not exhaustively hashed.'
    (output / 'provenance.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result
