import importlib
from pathlib import Path
import subprocess
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_REVISION = '6876159a11b4df116f30f667f8c9888617df0751'


def import_official_dinov3(source_dir=None, expected_revision=None):
    if source_dir is not None:
        source = Path(source_dir).expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        source = source.resolve()
        if not (source / 'dinov3' / 'models' / 'vision_transformer.py').is_file():
            raise FileNotFoundError(f'Official DINOv3 source not found at {source}. '
                                    'See docs/dinov3_rgb_pretraining.md for setup.')
        if expected_revision:
            revision = subprocess.check_output(
                ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if revision != expected_revision:
                raise ValueError(f'DINOv3 revision {revision} != expected {expected_revision}')
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        module = importlib.import_module('dinov3.models.vision_transformer')
    except ImportError as error:
        raise ImportError('Install the official DINOv3 package or configure backbone.parameters.source_dir. '
                          'DINOv3 requires Python >= 3.11 and a compatible PyTorch; '
                          'see docs/dinov3_rgb_pretraining.md.') from error
    if source_dir is not None and not Path(module.__file__).resolve().is_relative_to(source):
        raise RuntimeError('A different dinov3 package was already imported; restart with the configured source.')
    return module


def build_dino_v3_backbone(source_dir=None, weights=None, load_pretrained=True,
                          expected_revision=None, name='ViT-B/16'):
    if name != 'ViT-B/16':
        raise ValueError('The RGB pretraining entry point currently supports DINOv3 ViT-B/16.')
    import_official_dinov3(source_dir, expected_revision)
    from dinov3.hub.backbones import dinov3_vitb16
    model = dinov3_vitb16(pretrained=False)
    if load_pretrained:
        if weights is None:
            raise ValueError('A local official DINOv3 weights file is required; no implicit downloads are performed.')
        path = Path(weights).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'DINOv3 checkpoint not found: {path}')
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model
