"""Export full RGB weights, or an RGB-T initialization with copied TIR adapters."""
import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from trackit.models.backbone.dinov3.builder import OFFICIAL_REVISION
from trackit.models.methods.dinov3_rgb.transfer import rgbt_initialization_state_dict


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rgbt', action='store_true', help='Initialize independent TIR adapters from RGB')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace an existing export: {args.output}')
    state = load_file(str(args.checkpoint), device='cpu')
    if 'architecture' not in state:
        raise ValueError('Expected a full DINOv3RGB checkpoint, not DINOv3 or DINOv2 weights')
    if args.rgbt:
        state = rgbt_initialization_state_dict(state)
    with args.checkpoint.open('rb') as source:
        source_sha = hashlib.file_digest(source, 'sha256').hexdigest()
    metadata = dict(format='DINOv3RGB/full/v1', source_sha256=source_sha,
                    official_dinov3_revision=OFFICIAL_REVISION,
                    modalities='rgb,tir' if args.rgbt else 'rgb',
                    architecture=json.dumps(state['architecture'].tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(args.output), metadata=metadata)
    print(json.dumps(dict(output=str(args.output.resolve()), **metadata), indent=2))


if __name__ == '__main__':
    main()
