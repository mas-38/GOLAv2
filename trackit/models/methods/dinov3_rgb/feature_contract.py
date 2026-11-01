"""Versioned contracts for causal observations and privileged training targets.

All boxes use normalized XYXY coordinates in the ORIGINAL image. Crop boxes may
extend outside [0, 1] because search images can contain padding. Grids use (H, W).
"""
from dataclasses import dataclass, replace

import torch
from torch import Tensor


REFERENCE_PROTOCOL = 'dinov3-isolated-final-ln-search-v1'


def validate_boxes(boxes, batch, name='boxes'):
    if boxes.shape != (batch, 4) or not boxes.is_floating_point():
        raise ValueError(f'{name} must be floating-point normalized XYXY with shape (B, 4)')
    if not torch.isfinite(boxes).all() or not (boxes[:, 2:] > boxes[:, :2]).all():
        raise ValueError(f'{name} must be finite and have positive width/height')


def validate_frame_indices(indices, batch):
    if indices.shape != (batch,) or indices.dtype != torch.long or (indices < 0).any():
        raise ValueError('frame_indices must be nonnegative int64 values with shape (B,)')


@dataclass(frozen=True)
class ReferenceFeatures:
    tokens: Tensor  # (B, H*W, C), final LayerNorm output, no CLS/register tokens
    grid_hw: tuple[int, int]
    crop_boxes: Tensor  # (B, 4), normalized original-image coordinates
    frame_indices: Tensor  # (B,)
    modality: str
    space_id: str  # exact frozen snapshot + feature protocol fingerprint

    def __post_init__(self):
        if self.tokens.ndim != 3 or self.tokens.shape[1] != self.grid_hw[0] * self.grid_hw[1]:
            raise ValueError('Reference tokens must match their dense patch grid')
        if min(self.grid_hw) < 1 or self.tokens.shape[-1] < 1:
            raise ValueError('Reference feature dimensions must be positive')
        if not torch.isfinite(self.tokens).all() or self.tokens.requires_grad:
            raise ValueError('Reference features must be finite, detached frozen-encoder outputs')
        validate_boxes(self.crop_boxes, self.tokens.shape[0], 'crop_boxes')
        validate_frame_indices(self.frame_indices, self.tokens.shape[0])
        if not self.space_id or not self.modality:
            raise ValueError('Reference feature space and modality must be identified')
        if self.crop_boxes.device != self.tokens.device or self.frame_indices.device != self.tokens.device:
            raise ValueError('Feature tensors and geometry/time metadata must share a device')

    def select(self, index):
        if not 0 <= index < self.tokens.shape[0]:
            raise IndexError(index)
        return replace(self, tokens=self.tokens[index:index + 1],
                       crop_boxes=self.crop_boxes[index:index + 1],
                       frame_indices=self.frame_indices[index:index + 1])


@dataclass(frozen=True)
class FeatureObservation(ReferenceFeatures):
    """A real, currently available observation; eligible for causal history."""


@dataclass(frozen=True)
class FeatureSupervision(ReferenceFeatures):
    """Training-only privileged features; never eligible for causal history."""


@dataclass(frozen=True)
class AlignedTarget:
    tokens: Tensor  # (B, roi_H*roi_W, C), interpolated reference features
    valid_mask: Tensor  # (B, roi_H*roi_W), excludes crop padding/out-of-view points
    boxes: Tensor
    roi_hw: tuple[int, int]
    modality: str
    space_id: str
    frame_indices: Tensor


@dataclass(frozen=True)
class ReconstructionTarget:
    tokens: Tensor  # dense missing-modality reference search grid
    valid_mask: Tensor
    grid_hw: tuple[int, int]
    modality: str
    space_id: str
    frame_indices: Tensor


@dataclass(frozen=True)
class FutureTarget:
    appearance: AlignedTarget
    appearance_valid: Tensor  # (B,)
    motion: Tensor  # (B, 4): dx/w_t, dy/h_t, log(w_future/w_t), log(h_future/h_t)
    motion_valid: Tensor  # valid even when the object has left the search crop
    visible: Tensor
    horizon: int
    anchor_frame_indices: Tensor
    anchor_boxes: Tensor  # baseline/direct prediction, fixed BEFORE future/reconstruction decisions
    anchor_box_source: str
