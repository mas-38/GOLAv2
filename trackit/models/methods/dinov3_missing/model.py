import torch
from torch import nn

from ..dinov3_rgb.backbone import TrackingCrop
from ..dinov3_rgb.reference import FrozenReferenceEncoder
from .modules import FuturePrediction, FuturePredictor, SharedReconstructor, ReconstructionFusion, MODALITIES


def decode_prediction(output, crop_boxes, window_penalty=.45):
    """Return detached original-image normalized XYXY and unpenalized confidence."""
    scores = output['score_map'].detach().float().sigmoid()
    batch, height, width = scores.shape
    window = torch.outer(torch.hann_window(height, periodic=False, device=scores.device),
                         torch.hann_window(width, periodic=False, device=scores.device)).flatten()
    flat = scores.flatten(1)
    index = (flat * (1 - window_penalty) + window * window_penalty).argmax(1)
    row = torch.arange(batch, device=scores.device)
    local = output['boxes'].detach().float().reshape(batch, -1, 4)[row, index]
    crops = crop_boxes.float()
    boxes = crops[:, :2].repeat(1, 2) + local * (crops[:, 2:] - crops[:, :2]).repeat(1, 2)
    # Keep a positive extent even if the entire proposed box leaves the image.
    lower = boxes[:, :2].clamp(0, 1 - 1e-6)
    upper = torch.maximum(boxes[:, 2:].clamp(0, 1), lower + 1e-6)
    return torch.cat((lower, upper), 1), flat[row, index]


class DINOv3MissingTracker(nn.Module):
    def __init__(self, rgbt_tracker, *, hidden=256, heads=8, horizons=(1, 2), roi_hw=(4, 4),
                 reference_precision='float32'):
        super().__init__()
        self.tracker = rgbt_tracker
        if set(self.tracker.backbone.modality_embed) != set(MODALITIES):
            raise ValueError('Load the RGB-T initialization with independent RGB/TIR adapters first')
        self.temporal_options = dict(hidden=hidden, heads=heads, horizons=tuple(horizons), roi_hw=tuple(roi_hw))
        if reference_precision not in ('float32', 'bfloat16', 'float16'):
            raise ValueError('Unsupported reference precision')
        self.reference_precision = reference_precision
        self.reference = None
        self.stage = 'adapt'
        self.adaptation_source = None

    def activate_temporal(self, adaptation_source):
        if self.stage != 'adapt' or not adaptation_source:
            raise ValueError('Temporal training requires an identified RGB-T adaptation snapshot')
        self.adaptation_source = str(adaptation_source)
        self.tracker.backbone.requires_grad_(False).eval()
        self.reference = FrozenReferenceEncoder.from_tracker(
            self.tracker, normalization_profile=f'rgb-imagenet_tir-mm0226-v1;compute={self.reference_precision}')
        dim = self.tracker.backbone.embed_dim
        common = dict(dim=dim, hidden=self.temporal_options['hidden'], heads=self.temporal_options['heads'])
        self.future_predictor = FuturePredictor(**common, horizons=self.temporal_options['horizons'],
                                               roi_hw=self.temporal_options['roi_hw'])
        width, height = self.tracker.search_region_feat_size
        self.reconstructor = SharedReconstructor(**common, grid_hw=(height, width))
        self.fusion = ReconstructionFusion(**common)
        self.stage = 'temporal'
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == 'temporal':
            self.tracker.backbone.eval()
            self.reference.train(False)
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        architecture = state_dict.get('tracker.architecture')
        if architecture is None or not torch.equal(architecture.cpu(), self.tracker.architecture.cpu()):
            raise ValueError('RGB-T checkpoint architecture/LoRA configuration mismatch')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.reference is not None:
            # Recursive nn.Module loading bypasses the child's public load_state_dict.
            self.reference._space_id = self.reference._fingerprint()
            self.reference.requires_grad_(False).eval()
        return result

    def _reference_call(self, method, image, *args, **kwargs):
        # Reference outputs must not silently depend on a caller's autocast context.
        enabled = self.reference_precision != 'float32'
        dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(self.reference_precision, torch.bfloat16)
        with torch.autocast(image.device.type, enabled=enabled, dtype=dtype):
            return getattr(self.reference, method)(image.float(), *args, **kwargs)

    def observe_reference(self, image, *args, **kwargs):
        return self._reference_call('observe', image, *args, **kwargs)

    def supervise_reference(self, image, *args, **kwargs):
        return self._reference_call('supervise', image, *args, **kwargs)

    def forward(self, z, x, d, z_feat_mask, d_feat_mask, modality_available,
                crop_boxes, frame_indices, histories=None, reconstruction_steps=3):
        if reconstruction_steps not in (0, 1, 3, 5):
            raise ValueError('Supported reconstruction budgets: 0, 1, 3, 5')
        batch = x.shape[0]
        if x.ndim != 4 or x.shape[1] != 6 or z.shape[1] != 6 or d.shape[1] != 6:
            raise ValueError('RGB-T images must have six channels ordered RGB, TIR')
        if modality_available.shape != (batch, 2) or not torch.all((modality_available == 0) | (modality_available == 1)):
            raise ValueError('Availability must have binary shape (B,2), ordered RGB,TIR')
        available = modality_available.bool()
        if not available.any(1).all():
            raise ValueError('Both-missing frames must bypass encoding and retain the previous prediction')
        if crop_boxes.shape != (batch, 4) or frame_indices.shape != (batch,):
            raise ValueError('Each sample needs its original-image crop box and exact frame index')
        histories = [None] * batch if histories is None else histories
        if len(histories) != batch:
            raise ValueError('One causal history is required per sample')
        if z_feat_mask.ndim == 3:
            z_feat_mask = z_feat_mask[:, None].expand(-1, 2, -1, -1)
        if d_feat_mask.ndim == 3:
            d_feat_mask = d_feat_mask[:, None].expand(-1, 2, -1, -1)
        outputs, samples = [None] * batch, [None] * batch
        for state in ((True, False), (False, True), (True, True)):
            selected = (available == available.new_tensor(state)).all(1).nonzero().flatten()
            if not len(selected):
                continue
            crops = {}
            for m, name in enumerate(MODALITIES):
                channels = slice(m * 3, (m + 1) * 3)
                crops[f'z_{name}'] = TrackingCrop(z[selected, channels], 'z', name, z_feat_mask[selected, m])
                crops[f'd_{name}'] = TrackingCrop(d[selected, channels], 'd', name, d_feat_mask[selected, m])
                if state[m]:
                    crops[f'x_{name}'] = TrackingCrop(x[selected, channels], 'x', name)
            joint = self.tracker.backbone.encode_crops(crops)
            joint_search = torch.stack([joint[f'x_{name}'] for m, name in enumerate(MODALITIES) if state[m]]).mean(0)
            base = self.tracker.predict_from_features(joint_search)
            anchor_boxes, _ = decode_prediction(base, crop_boxes[selected])
            observations = {}
            future = None
            reconstructed = {}
            missing = None
            fused = joint_search
            fusion_gates = joint_search.new_zeros(len(selected), 1)
            if self.stage == 'temporal':
                for m, name in enumerate(MODALITIES):
                    if state[m]:
                        observations[name] = self.observe_reference(
                            x[selected, m * 3:(m + 1) * 3], name, frame_indices[selected], crop_boxes[selected])
                past = [histories[index] for index in selected.tolist()]
                for local, item in enumerate(past):
                    if item is not None:
                        if item.space_id != self.reference.space_id:
                            raise ValueError('History belongs to a different frozen feature space')
                        if len(item.frame_indices) and not (item.frame_indices < frame_indices[selected[local]]).all():
                            raise ValueError('Future or current-frame entries cannot enter causal history')
                actual = list(observations.values())
                future = self.future_predictor(actual, past, anchor_boxes)
                if not all(state):
                    missing = MODALITIES[1 if state[0] else 0]
                    if reconstruction_steps:
                        reconstructed = self.reconstructor(actual, past, future, missing, reconstruction_steps)
                        fused, fusion_gates = self.fusion(joint_search, reconstructed[reconstruction_steps], future, missing)
            decoded = self.tracker.predict_from_features(fused)
            for local, index in enumerate(selected.tolist()):
                aux = dict(index=index, crop_box=crop_boxes[index:index + 1],
                           frame_index=frame_indices[index:index + 1], anchor_box=anchor_boxes[local:local + 1],
                           observations=[], reconstruction={}, missing_modality=missing,
                           fusion_gate=fusion_gates[local:local + 1], future=None)
                if self.stage == 'temporal':
                    actual = [observation.select(local) for observation in observations.values()]
                    single_future = FuturePrediction(future.appearance[local:local + 1],
                                                     future.motion[local:local + 1], future.visibility[local:local + 1])
                    aux.update(observations=actual, future=single_future,
                               reconstruction={step: value[local:local + 1] for step, value in reconstructed.items()})
                outputs[index] = {key: value[local:local + 1] for key, value in decoded.items()}
                samples[index] = aux
        return dict(score_map=torch.cat([o['score_map'] for o in outputs]),
                    boxes=torch.cat([o['boxes'] for o in outputs]), samples=samples)
