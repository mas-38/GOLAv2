"""Read-only checks before launching GOT-10k RGB tracking pretraining."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from trackit.core.boot.funcs.main.load_config import load_config
from trackit.core.runtime.global_constant import get_global_constant
from trackit.models.backbone.dinov3.builder import PROJECT_ROOT


def check_got10k(root, split='train'):
    root = Path(root).expanduser()
    manifest = root / split / 'list.txt'
    if not manifest.is_file():
        raise FileNotFoundError(f'Missing official GOT-10k sequence list: {manifest}. '
                                'Finish extracting the official split before full training.')
    sequences = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError('GOT-10k list.txt must contain a nonempty list without duplicates')
    required = ['groundtruth.txt', 'meta_info.ini', 'absence.label', 'cover.label', 'cut_by_image.label']
    failures = []
    for sequence in sequences:
        if Path(sequence).name != sequence or sequence in ('.', '..'):
            raise ValueError(f'Invalid sequence name: {sequence}')
        folder = root / split / sequence
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            failures.append(f'{sequence}: missing {missing}')
            continue
        count = len((folder / 'groundtruth.txt').read_text().splitlines())
        images = len(list(folder.glob('*.jpg')))
        labels_match = all(len((folder / name).read_text().splitlines()) == count
                           for name in ('absence.label', 'cover.label', 'cut_by_image.label'))
        if count == 0 or images != count or not labels_match:
            failures.append(f'{sequence}: {images} images / {count} boxes; label lengths match={labels_match}')
    if failures:
        raise ValueError(f'{len(failures)} incomplete sequences; first failures:\n' + '\n'.join(failures[:10]))
    return len(sequences)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--got10k-root', type=Path)
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    config = load_config(SimpleNamespace(config_path=str(PROJECT_ROOT / 'config'),
                        method_name='DINOv3RGB', config_name='dinov3', mixin_config=None,
                        weight_path=None, eval=False))
    parameters = config['model']['backbone']['parameters']
    source = (PROJECT_ROOT / parameters['source_dir']).resolve()
    weights = (PROJECT_ROOT / parameters['weights']).resolve()
    if sys.version_info < (3, 11):
        raise RuntimeError('Official DINOv3 requires Python >= 3.11')
    import torch
    from packaging.version import Version
    if Version(torch.__version__.split('+')[0]) < Version('2.7.1'):
        raise RuntimeError('Use PyTorch >= 2.7.1 for the pinned official DINOv3 implementation')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != parameters['expected_revision']:
        raise ValueError(f'DINOv3 revision mismatch: {revision}')
    if not weights.is_file():
        raise FileNotFoundError(weights)
    print(f'DINOv3 source: {source} ({revision})')
    print(f'Pretrained weights: {weights}')
    root = args.got10k_root or Path(get_global_constant('GOT10k_PATH'))
    count = check_got10k(root, args.split)
    print(f'GOT-10k {args.split}: {count} sequences passed file/count checks')
    print('Setup checks passed. No training was started.')


if __name__ == '__main__':
    main()
