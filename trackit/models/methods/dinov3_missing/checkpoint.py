import hashlib
import json
from pathlib import Path
import random

import numpy as np
from safetensors.torch import load_file
import torch

from trackit.models.backbone.dinov3.builder import build_dino_v3_backbone
from ..dinov3_rgb.model import DINOv3RGBTracker
from .model import DINOv3MissingTracker


SCHEMA = 'dinov3-rgbt-temporal/v1'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_spec(config):
    model = config['model']
    return dict(tracking=model['tracking'], template_feat_size=model['template_feat_size'],
                search_feat_size=model['search_feat_size'], temporal=model['temporal'],
                official_revision=model['official_revision'], normalization='rgb-imagenet_tir-mm0226-v1',
                reference_precision=config['runtime']['precision'])


def read_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('stage') not in ('adapt', 'temporal'):
        raise ValueError('Expected a complete versioned DINOv3 RGB-T checkpoint')
    return checkpoint


def build_model(config, stage='adapt', checkpoint_path=None, device='cpu'):
    settings = config['model']
    checkpoint = read_checkpoint(checkpoint_path) if checkpoint_path is not None else None
    if stage == 'temporal' and checkpoint is None:
        raise ValueError('First adapt TIR on real RGB-T data; temporal stage requires its checkpoint')
    spec = model_spec(config)
    if checkpoint is not None and checkpoint['model_spec'] != spec:
        raise ValueError('Checkpoint model/normalization/temporal configuration differs')
    vit = build_dino_v3_backbone(source_dir=settings['official_source'], load_pretrained=False,
                               expected_revision=settings['official_revision'])
    tracker = DINOv3RGBTracker(vit, settings['template_feat_size'], settings['search_feat_size'],
                              **settings['tracking'])
    tracker.backbone.add_modality('tir')
    model = DINOv3MissingTracker(tracker, **settings['temporal'], reference_precision=config['runtime']['precision'])
    if checkpoint is None:
        if stage != 'adapt':
            raise ValueError('First adapt TIR on real RGB-T data; temporal stage requires its checkpoint')
        path = settings['initialization']
        if file_sha256(path) != settings['initialization_sha256']:
            raise ValueError('RGB-T initialization SHA-256 differs from the selected RGB best export')
        tracker.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    elif checkpoint['stage'] == 'adapt':
        model.load_state_dict(checkpoint['model'], strict=True)
        if stage == 'temporal':
            if checkpoint.get('optimizer_updates', 0) < 1:
                raise ValueError('An untrained copied TIR branch cannot define the reference space')
            model.activate_temporal(file_sha256(checkpoint_path))
    else:
        if stage != 'temporal':
            raise ValueError('Cannot resume temporal weights as an adaptation run')
        model.activate_temporal(checkpoint['adaptation_source'])
        model.load_state_dict(checkpoint['model'], strict=True)
        if model.reference.space_id != checkpoint['reference_space_id']:
            raise ValueError('Frozen reference fingerprint does not match saved metadata')
    return model.to(device)


def capture_rng():
    numpy = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                numpy=dict(kind=numpy[0], keys=numpy[1].tolist(), position=numpy[2],
                           has_gauss=numpy[3], cached_gaussian=numpy[4]))


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device layout differs from the saved run')
        torch.cuda.set_rng_state_all(state['cuda'])
    numpy = state['numpy']
    np.random.set_state((numpy['kind'], np.array(numpy['keys'], dtype=np.uint32),
                         numpy['position'], numpy['has_gauss'], numpy['cached_gaussian']))


def save_checkpoint(path, model, optimizer, scheduler, scaler, *, config, epoch, next_batch,
                    optimizer_updates, signature, statistics=None):
    payload = dict(schema=SCHEMA, stage=model.stage, model_spec=model_spec(config),
                   model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                   optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                   epoch=epoch, next_batch=next_batch, optimizer_updates=optimizer_updates,
                   epoch_statistics=statistics,
                   rng=capture_rng(), training_signature=signature, adaptation_source=model.adaptation_source,
                   reference_space_id=model.reference.space_id if model.reference is not None else None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training_state(checkpoint, optimizer, scheduler, scaler, signature):
    required = {'optimizer', 'scheduler', 'scaler', 'rng', 'epoch', 'next_batch', 'optimizer_updates'}
    if not required.issubset(checkpoint):
        raise ValueError('Incomplete resume state; model weights alone cannot restore training')
    if checkpoint['training_signature'] != signature:
        raise ValueError('Training data/schedule/model changed; start a new fine-tuning run instead')
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    restore_rng(checkpoint['rng'])
    return checkpoint['epoch'], checkpoint['next_batch'], checkpoint['optimizer_updates']


def training_signature(config, stage, train_names_digest):
    runtime = {key: config['runtime'][key] for key in ('precision', 'max_grad_norm', 'train_reconstruction_steps')}
    selected = dict(model=model_spec(config), stage=stage, schedule=config['stages'][stage],
                    source_revision=config.get('source_revision'),
                    loss=config.get('loss', {}), runtime=runtime,
                    data=config['data'], seed=config['seed'], train_names_digest=train_names_digest)
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
