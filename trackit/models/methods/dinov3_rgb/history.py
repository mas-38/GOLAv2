"""Bounded per-track/per-modality history of accepted real observations only."""
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .alignment import align_target_roi
from .feature_contract import FeatureObservation


@dataclass(frozen=True)
class HistoryEntry:
    tokens: Tensor
    valid_mask: Tensor
    box: Tensor
    frame_index: int
    modality: str
    confidence: float
    initial: bool


@dataclass(frozen=True)
class HistoryRead:
    tokens: Tensor  # (entries, roi_tokens, C)
    valid_mask: Tensor
    boxes: Tensor
    frame_indices: Tensor
    ages: Tensor
    confidences: Tensor
    initial: Tensor
    modalities: tuple[str, ...]
    space_id: str


class CausalFeatureHistory:
    def __init__(self, space_id, feature_dim, *, roi_hw=(4, 4), capacity_per_modality=8,
                 min_confidence=0.84, max_recent_age=64, min_roi_fraction=0.5):
        if not space_id or feature_dim < 1 or capacity_per_modality < 2 or min(roi_hw) < 1:
            raise ValueError('History requires a space, feature dimension and capacity >= 2')
        if not 0 <= min_confidence <= 1 or not 0 <= min_roi_fraction <= 1 or max_recent_age < 1:
            raise ValueError('Invalid history quality or age settings')
        self.space_id = space_id
        self.feature_dim = feature_dim
        self.roi_hw = tuple(roi_hw)
        self.capacity = capacity_per_modality
        self.min_confidence = min_confidence
        self.max_recent_age = max_recent_age
        self.min_roi_fraction = min_roi_fraction
        self._entries = {}
        self._initial_frames = {}

    def start_track(self, track_id, *, initialization_frame_index=0):
        if track_id is None or track_id in self._initial_frames:
            raise ValueError('Use a distinct track ID or reset it before initialization')
        if not isinstance(initialization_frame_index, int) or initialization_frame_index < 0:
            raise ValueError('initialization_frame_index must be a nonnegative integer')
        self._initial_frames[track_id] = initialization_frame_index
        self._entries[track_id] = {}

    def reset(self, track_id=None):
        if track_id is None:
            self._entries.clear()
            self._initial_frames.clear()
        else:
            self._entries.pop(track_id, None)
            self._initial_frames.pop(track_id, None)

    def record(self, track_id, observation, target_box, *, now_frame_index, confidence,
               initial=False, box_source='prediction'):
        """Call AFTER the current frame's prediction and template decision.

        The initial user-provided GT box is permitted only for an initial entry.
        Later writes use predicted boxes. Future or reconstructed targets cannot
        be written through this API. Stored tensors are detached CPU copies.
        """
        if not isinstance(observation, FeatureObservation):
            raise TypeError('History accepts real FeatureObservation, never privileged labels or generated tensors')
        if track_id not in self._initial_frames:
            raise ValueError('Call start_track before recording history for a sequence')
        if observation.space_id != self.space_id or observation.tokens.shape[-1] != self.feature_dim:
            raise ValueError('Feature-space mismatch: reset/rebuild history after changing the encoder')
        if observation.tokens.shape[0] != 1:
            raise ValueError('Record one track at a time; use observation.select(batch_index)')
        frame = int(observation.frame_indices.item())
        if not isinstance(now_frame_index, int) or frame != now_frame_index:
            raise ValueError('Only an observation from the current decision frame may be recorded')
        if frame < self._initial_frames[track_id] or (initial and frame != self._initial_frames[track_id]):
            raise ValueError('Initial ground-truth anchors are allowed only at the declared initialization frame')
        if box_source not in ('prediction', 'initial_ground_truth') or (box_source == 'initial_ground_truth' and not initial):
            raise ValueError('Only predictions or the initial user-provided ground-truth box may enter history')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be finite and in [0,1]')
        per_track = self._entries.get(track_id, {})
        entries = per_track.get(observation.modality, [])
        if entries and frame <= entries[-1].frame_index:
            raise ValueError('History writes must advance time independently for each modality')
        if initial and entries:
            raise ValueError('An initial anchor cannot replace existing history')
        if not initial and confidence <= self.min_confidence:
            return False
        box = torch.as_tensor(target_box, device=observation.tokens.device, dtype=torch.float32).reshape(1, 4)
        aligned = align_target_roi(observation, box, self.roi_hw)
        fraction = aligned.valid_mask.float().mean().item()
        if fraction < self.min_roi_fraction or not aligned.valid_mask.any():
            return False
        entry = HistoryEntry(aligned.tokens[0].detach().float().cpu().clone(),
                             aligned.valid_mask[0].detach().cpu().clone(),
                             box[0].detach().cpu().clone(), frame, observation.modality, confidence, initial)
        entries = [*entries, entry]
        if len(entries) > self.capacity:
            # Preserve the original reliable anchor; evict the oldest recent entry.
            index = 1 if entries[0].initial else 0
            entries.pop(index)
        self._entries.setdefault(track_id, {})[observation.modality] = entries
        return True

    def read(self, track_id, *, now_frame_index, modalities=None, device='cpu'):
        if not isinstance(now_frame_index, int) or now_frame_index < 0:
            raise ValueError('now_frame_index must be a nonnegative integer')
        entries = []
        for modality, items in self._entries.get(track_id, {}).items():
            if modalities is not None and modality not in modalities:
                continue
            entries.extend(entry for entry in items
                           if entry.frame_index < now_frame_index and
                           (entry.initial or now_frame_index - entry.frame_index <= self.max_recent_age))
        entries.sort(key=lambda entry: (entry.frame_index, entry.modality))
        roi_length = self.roi_hw[0] * self.roi_hw[1]
        tokens = (torch.stack([entry.tokens for entry in entries]) if entries else
                  torch.empty(0, roi_length, self.feature_dim))
        mask = (torch.stack([entry.valid_mask for entry in entries]) if entries else
                torch.empty(0, roi_length, dtype=torch.bool))
        boxes = torch.stack([entry.box for entry in entries]) if entries else torch.empty(0, 4)
        frames = torch.tensor([entry.frame_index for entry in entries], dtype=torch.long)
        return HistoryRead(tokens.to(device), mask.to(device), boxes.to(device), frames.to(device),
                           (now_frame_index - frames).to(device),
                           torch.tensor([entry.confidence for entry in entries], dtype=torch.float32, device=device),
                           torch.tensor([entry.initial for entry in entries], dtype=torch.bool, device=device),
                           tuple(entry.modality for entry in entries), self.space_id)
