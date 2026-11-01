"""Frozen reference encoder; no cross-crop attention in supervised features."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import torch
from torch import nn

from .backbone import TrackingCrop, TrackingFeatureBundle
from .feature_contract import REFERENCE_PROTOCOL, FeatureObservation, FeatureSupervision


@dataclass(frozen=True)
class ReferenceContext:
    features: TrackingFeatureBundle
    roles: Mapping[str, tuple[str, str]]
    space_id: str


class FrozenReferenceEncoder(nn.Module):
    """Take a snapshot after RGB-T backbone adaptation, then keep it fixed.

    observe() is for available real images. supervise() can read a hidden modality
    or future image during training and returns a different, non-cacheable type.
    Both use the same search-role isolated final-LayerNorm feature space.
    """
    def __init__(self, backbone, *, search_grid_hw=(16, 16), normalization_profile='rgb-imagenet_tir-mm-v1'):
        super().__init__()
        if not normalization_profile:
            raise ValueError('The input normalization profile must be identified')
        if len(search_grid_hw) != 2 or min(search_grid_hw) < 1:
            raise ValueError('search_grid_hw must contain positive height/width')
        self.search_grid_hw = tuple(search_grid_hw)
        self.normalization_profile = normalization_profile
        self.encoder = deepcopy(backbone).requires_grad_(False).eval()
        self._space_id = self._fingerprint()
        self.train(False)

    @classmethod
    def from_tracker(cls, tracker, **kwargs):
        width, height = tracker.search_region_feat_size
        return cls(tracker.backbone, search_grid_hw=(height, width), **kwargs)

    def _fingerprint(self):
        digest = hashlib.sha256(REFERENCE_PROTOCOL.encode())
        layout = {'patch_size': self.encoder.patch_size,
                  'search_grid_hw': self.search_grid_hw,
                  'normalization_profile': self.normalization_profile,
                  'independent_depth': self.encoder.independent_depth,
                  'num_prefix_tokens': self.encoder.num_prefix_tokens,
                  'rope_normalize_coords': self.encoder.rope_embed.normalize_coords,
                  'rope_dtype': str(self.encoder.rope_embed.dtype),
                  'norm_eps': [(name, module.eps) for name, module in self.encoder.named_modules()
                               if hasattr(module, 'eps')],
                  'scalings': [(name, module.scaling) for name, module in self.encoder.named_modules()
                               if hasattr(module, 'scaling')]}
        digest.update(json.dumps(layout, sort_keys=True).encode())
        for name, tensor in sorted(self.encoder.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return REFERENCE_PROTOCOL + ':' + digest.hexdigest()

    def _apply(self, fn, recurse=True):
        before = tuple(t.dtype for t in self.parameters())
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, '_space_id') and before != tuple(t.dtype for t in self.parameters()):
            self._space_id = self._fingerprint()
        return result

    @property
    def space_id(self):
        return self._space_id

    def train(self, mode=True):
        # Calling parent_model.train() must not enable LoRA dropout or RoPE jitter.
        super().train(False)
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.encoder.requires_grad_(False)
        self.train(False)
        self._space_id = self._fingerprint()
        return result

    @staticmethod
    def _check_real_availability(real_available, batch, device):
        flags = torch.as_tensor(real_available, device=device)
        if flags.ndim == 0:
            flags = flags.expand(batch)
        if flags.shape != (batch,) or not ((flags == 0) | (flags == 1)).all() or not flags.bool().all():
            raise ValueError('Omit missing images; only physically available real images may be encoded')

    def _wrap(self, cls, tokens, grid_hw, modality, frame_indices, crop_boxes, real_available):
        self._check_real_availability(real_available, tokens.shape[0], tokens.device)
        frames = torch.as_tensor(frame_indices, device=tokens.device)
        if frames.ndim == 0:
            frames = frames.expand(tokens.shape[0])
        return cls(tokens.detach(), grid_hw,
                   torch.as_tensor(crop_boxes, device=tokens.device, dtype=torch.float32).detach().clone(),
                   frames.detach().clone(), modality, self.space_id)

    @torch.no_grad()
    def _encode(self, cls, image, modality, frame_indices, crop_boxes, real_available):
        self._check_real_availability(real_available, image.shape[0], image.device)
        expected_hw = tuple(size * self.encoder.patch_size for size in self.search_grid_hw)
        if image.shape[-2:] != expected_hw:
            raise ValueError(f'Reference observations require search-view crops of size {expected_hw}')
        tokens = self.encoder.encode_isolated_crops({'x': TrackingCrop(image, 'x', modality)})['x']
        grid_hw = (image.shape[-2] // self.encoder.patch_size, image.shape[-1] // self.encoder.patch_size)
        return self._wrap(cls, tokens, grid_hw, modality, frame_indices, crop_boxes, real_available)

    def observe(self, image, modality, frame_indices, crop_boxes, *, real_available=True):
        return self._encode(FeatureObservation, image, modality, frame_indices, crop_boxes, real_available)

    def supervise(self, image, modality, frame_indices, crop_boxes, *, real_available=True):
        return self._encode(FeatureSupervision, image, modality, frame_indices, crop_boxes, real_available)

    @torch.no_grad()
    def encode_context(self, crops, *, include_independent=False):
        """One frozen runtime backbone can share shallow work between solo and joint paths.

        Only real, causally available crops belong in this context. Encode hidden
        supervision images separately with supervise(), never concatenate them here.
        """
        expected_hw = tuple(size * self.encoder.patch_size for size in self.search_grid_hw)
        if any(crop.role == 'x' and crop.image.shape[-2:] != expected_hw for crop in crops.values()):
            raise ValueError('Context search crops must match the reference search grid')
        features = self.encoder.encode_feature_bundle(crops, include_independent=include_independent,
                                                       include_isolated=True, include_joint=True)
        return ReferenceContext(features, {name: (crop.role, crop.modality) for name, crop in crops.items()},
                                self.space_id)

    def observation_from_context(self, context, name, frame_indices, crop_boxes, *, real_available=True):
        if context.space_id != self.space_id:
            raise ValueError('Context belongs to a different frozen feature space')
        role, modality = context.roles[name]
        if role != 'x':
            raise ValueError('History observations must originate from current real search images')
        return self._wrap(FeatureObservation, context.features.isolated[name], context.features.grids[name],
                          modality, frame_indices, crop_boxes, real_available)
