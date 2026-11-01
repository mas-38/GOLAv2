"""Object-relative appearance targets plus motion in a fixed current-frame anchor."""
from dataclasses import replace

import torch
from torch.nn import functional as F

from .feature_contract import (AlignedTarget, FeatureSupervision, FutureTarget, ReconstructionTarget,
                               validate_boxes, validate_frame_indices)


def crop_boxes_from_siamfc(cropping_params, image_sizes_wh, output_sizes_wh):
    """Convert ADJUSTED source-pixel -> crop-pixel parameters to image-normalized bounds.

    Use the parameters returned by apply_siamfc_cropping, not its pre-rounding input.
    Rotation/flip are outside this v1 axis-aligned reference-view contract.
    """
    parameters = torch.as_tensor(cropping_params, dtype=torch.float32)
    if parameters.ndim == 2:
        parameters = parameters.unsqueeze(0)
    if parameters.ndim != 3 or parameters.shape[1:] != (2, 2) or not torch.isfinite(parameters).all():
        raise ValueError('cropping_params must be finite (B,2,2) scale/translation parameters')
    batch = parameters.shape[0]
    image_sizes = torch.as_tensor(image_sizes_wh, dtype=torch.float32, device=parameters.device).expand(batch, 2)
    output_sizes = torch.as_tensor(output_sizes_wh, dtype=torch.float32, device=parameters.device).expand(batch, 2)
    if not torch.isfinite(image_sizes).all() or not torch.isfinite(output_sizes).all() or \
            not (image_sizes > 0).all() or not (output_sizes > 0).all() or not (parameters[:, 0] > 0).all():
        raise ValueError('Image/output sizes and axis-aligned crop scales must be positive')
    scale, translation = parameters[:, 0], parameters[:, 1]
    top_left = -translation / scale / image_sizes
    bottom_right = (output_sizes - translation) / scale / image_sizes
    return torch.cat((top_left, bottom_right), dim=-1)


def siamfc_from_crop_boxes(crop_boxes, image_sizes_wh, output_sizes_wh):
    """Build the same normalized anchor crop on another frame (use subpixel cropping)."""
    boxes = torch.as_tensor(crop_boxes, dtype=torch.float32)
    if boxes.ndim != 2:
        raise ValueError('crop_boxes must have shape (B,4)')
    validate_boxes(boxes, boxes.shape[0], 'crop_boxes')
    sizes = torch.as_tensor(image_sizes_wh, dtype=torch.float32, device=boxes.device).expand(boxes.shape[0], 2)
    outputs = torch.as_tensor(output_sizes_wh, dtype=torch.float32, device=boxes.device).expand(boxes.shape[0], 2)
    if not torch.isfinite(sizes).all() or not torch.isfinite(outputs).all() or \
            not (sizes > 0).all() or not (outputs > 0).all():
        raise ValueError('Image/output sizes must be finite and positive')
    scale = outputs / ((boxes[:, 2:] - boxes[:, :2]) * sizes)
    translation = -boxes[:, :2] * sizes * scale
    return torch.stack((scale, translation), dim=1)


def _sampling_points(boxes, output_hw):
    height, width = output_hw
    if min(height, width) < 1:
        raise ValueError('ROI grid dimensions must be positive')
    y = (torch.arange(height, device=boxes.device, dtype=torch.float32) + 0.5) / height
    x = (torch.arange(width, device=boxes.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    relative = torch.stack((xx, yy), dim=-1).unsqueeze(0)
    return boxes[:, None, None, :2] + relative * (boxes[:, 2:] - boxes[:, :2])[:, None, None]


def align_target_roi(features, target_boxes, roi_hw=(4, 4)):
    """Sample one target ROI per image using pixel-center grid_sample coordinates.

    Appearance is aligned inside the object box, not by a future search-grid index.
    Boxes/crop geometry are normalized in the original image, before crop resizing.
    Padded/out-of-crop sample centers are masked; valid edge values use border mode.
    """
    batch, _, channels = features.tokens.shape
    boxes = torch.as_tensor(target_boxes, device=features.tokens.device, dtype=torch.float32)
    validate_boxes(boxes, batch, 'target_boxes')
    points = _sampling_points(boxes, roi_hw)
    crops = features.crop_boxes.float()
    local = (points - crops[:, None, None, :2]) / (crops[:, 2:] - crops[:, :2])[:, None, None]
    valid = ((local >= 0) & (local <= 1) & (points >= 0) & (points <= 1)).all(dim=-1)
    height, width = features.grid_hw
    dense = features.tokens.float().transpose(1, 2).reshape(batch, channels, height, width)
    # align_corners=False maps [0,1] to the crop's outer edges; patch centers are
    # (i+0.5)/size. This preserves cell centers across different crop resolutions.
    pooled = F.grid_sample(dense, local * 2 - 1, mode='bilinear',
                           padding_mode='border', align_corners=False)
    tokens = pooled.flatten(2).transpose(1, 2)
    valid = valid.flatten(1)
    tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
    return AlignedTarget(tokens, valid, boxes, tuple(roi_hw), features.modality,
                         features.space_id, features.frame_indices)


def _require_anchor(features, anchor_frame_indices, anchor_crop_boxes, horizon):
    batch = features.tokens.shape[0]
    indices = torch.as_tensor(anchor_frame_indices, device=features.tokens.device)
    if indices.ndim == 0:
        indices = indices.expand(batch)
    validate_frame_indices(indices, batch)
    crops = torch.as_tensor(anchor_crop_boxes, dtype=torch.float32, device=features.tokens.device)
    validate_boxes(crops, batch, 'anchor_crop_boxes')
    if not torch.equal(features.frame_indices, indices + horizon):
        raise ValueError('Target frames must equal anchor frames plus the declared horizon')
    if not torch.allclose(features.crop_boxes, crops, rtol=0, atol=1e-6):
        raise ValueError('Targets must use the current anchor search crop; future-GT recentering is not allowed')
    return indices


def reconstruction_target(features, anchor_frame_indices, anchor_crop_boxes):
    if not isinstance(features, FeatureSupervision):
        raise TypeError('Reconstruction labels require training-only FeatureSupervision')
    _require_anchor(features, anchor_frame_indices, anchor_crop_boxes, 0)
    points = _sampling_points(features.crop_boxes.float(), features.grid_hw)
    valid = ((points >= 0) & (points <= 1)).all(dim=-1).flatten(1)
    return ReconstructionTarget(features.tokens.detach(), valid, features.grid_hw,
                                features.modality, features.space_id, features.frame_indices)


@torch.no_grad()
def future_target(features, *, anchor_frame_indices, anchor_crop_boxes, anchor_boxes,
                  future_boxes, horizon, visible=None, roi_hw=(4, 4), min_roi_fraction=0.5,
                  anchor_box_source='baseline_prediction'):
    """Training label only. The future encoder crop stays fixed at the current crop.

    future_boxes are privileged annotations used only to align the label ROI and
    derive motion. They must never be passed to the causal predictor or cache.
    Out-of-crop appearance is invalid; visible out-of-crop motion remains useful.
    """
    if not isinstance(features, FeatureSupervision):
        raise TypeError('Future labels require training-only FeatureSupervision')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('horizon must be a positive integer frame offset')
    if not 0 <= min_roi_fraction <= 1:
        raise ValueError('min_roi_fraction must lie in [0,1]')
    if anchor_box_source not in ('baseline_prediction', 'initial_ground_truth'):
        raise ValueError('Anchor must be known before future prediction, not the final reconstruction-dependent box')
    indices = _require_anchor(features, anchor_frame_indices, anchor_crop_boxes, horizon)
    batch = features.tokens.shape[0]
    device = features.tokens.device
    anchor = torch.as_tensor(anchor_boxes, dtype=torch.float32, device=device)
    future = torch.as_tensor(future_boxes, dtype=torch.float32, device=device)
    validate_boxes(anchor, batch, 'anchor_boxes')
    if future.shape != (batch, 4):
        raise ValueError('future_boxes must have shape (B,4)')
    box_valid = torch.isfinite(future).all(dim=-1) & (future[:, 2:] > future[:, :2]).all(dim=-1)
    if visible is None:
        visible = torch.ones(batch, dtype=torch.bool, device=device)
    else:
        visible = torch.as_tensor(visible, device=device)
        if visible.shape != (batch,) or not ((visible == 0) | (visible == 1)).all():
            raise ValueError('visible must have binary shape (B,)')
        visible = visible.bool()
    motion_valid = visible & box_valid
    safe_future = torch.where(motion_valid.unsqueeze(-1), future, anchor)
    appearance = align_target_roi(features, safe_future, roi_hw)
    fraction = appearance.valid_mask.float().mean(dim=-1)
    appearance_valid = motion_valid & (fraction >= min_roi_fraction) & appearance.valid_mask.any(dim=-1)
    mask = appearance.valid_mask & appearance_valid.unsqueeze(-1)
    appearance = replace(appearance, valid_mask=mask,
                         tokens=torch.where(mask.unsqueeze(-1), appearance.tokens, torch.zeros_like(appearance.tokens)))
    anchor_wh = anchor[:, 2:] - anchor[:, :2]
    future_wh = safe_future[:, 2:] - safe_future[:, :2]
    anchor_center = (anchor[:, :2] + anchor[:, 2:]) / 2
    future_center = (safe_future[:, :2] + safe_future[:, 2:]) / 2
    motion = torch.cat(((future_center - anchor_center) / anchor_wh,
                        (future_wh / anchor_wh).log()), dim=-1)
    motion = torch.where(motion_valid.unsqueeze(-1), motion, torch.zeros_like(motion))
    return FutureTarget(appearance, appearance_valid, motion.detach(), motion_valid, visible, horizon, indices,
                        anchor.detach().clone(), anchor_box_source)


def masked_cosine_loss(prediction, target_tokens, valid_mask):
    """Scale-insensitive reference-space supervision, including all-invalid batches."""
    if prediction.shape != target_tokens.shape or valid_mask.shape != prediction.shape[:-1]:
        raise ValueError('Prediction, target and per-token validity mask must agree')
    if valid_mask.dtype != torch.bool:
        raise ValueError('valid_mask must be boolean')
    mask = valid_mask.unsqueeze(-1)
    prediction = torch.where(mask, prediction.float(), torch.zeros_like(prediction, dtype=torch.float32))
    target = torch.where(mask, target_tokens.detach().float(), torch.zeros_like(prediction))
    losses = 1 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-6)
    return torch.where(valid_mask, losses, torch.zeros_like(losses)).sum() / valid_mask.sum().clamp_min(1)
