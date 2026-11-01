import torch

from trackit.data.methods.rgbt_temporal import crop_from_box, crop_image, normalize, foreground_mask
from .model import decode_prediction
from .pipeline import new_history
from .modules import MODALITIES


class RGBTSequenceTracker:
    """One causal track. Hidden current images and future annotations have no API here."""
    def __init__(self, model, *, reconstruction_steps=3, template_threshold=.84):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.steps = reconstruction_steps
        self.template_threshold = template_threshold
        self.initialized = False

    @torch.no_grad()
    def initialize(self, image, initial_box_xyxy, frame_index=0, track_id='track'):
        if image is None or image.ndim != 3 or image.shape[0] != 6 or not torch.isfinite(image).all():
            raise ValueError('Initialization requires both real registered RGB/TIR images')
        self.frame_hw = tuple(image.shape[-2:])
        height, width = self.frame_hw
        self.last_box = torch.as_tensor(initial_box_xyxy, dtype=torch.float32) / torch.tensor([width, height, width, height])
        if not (self.last_box[2:] > self.last_box[:2]).all():
            raise ValueError('Initialization box must have positive extent')
        self.last_confidence = 1.
        self.last_frame = int(frame_index)
        self.track_id = track_id
        tw, th = self.model.tracker.template_feat_size
        sw, sh = self.model.tracker.search_region_feat_size
        self.template_hw, self.search_hw = (th * 16, tw * 16), (sh * 16, sw * 16)
        crop = crop_from_box(self.last_box, self.frame_hw, 2.)
        self.z = normalize(crop_image(image, crop, self.template_hw))[None].to(self.device)
        mask = foreground_mask(self.last_box, crop, (th, tw))[None, None].repeat(1, 2, 1, 1)
        self.z_mask = mask.to(self.device)
        self.d, self.d_mask = self.z.clone(), self.z_mask.clone()
        self.history = None
        if self.model.stage == 'temporal':
            self.history = new_history(self.model)
            self.history.start_track(track_id, initialization_frame_index=self.last_frame)
            search_crop = crop_from_box(self.last_box, self.frame_hw, 4.)
            search = normalize(crop_image(image, search_crop, self.search_hw))[None].to(self.device)
            for m, modality in enumerate(MODALITIES):
                observation = self.model.observe_reference(search[:, m * 3:(m + 1) * 3], modality,
                                                            [self.last_frame], search_crop[None].to(self.device))
                self.history.record(track_id, observation, self.last_box, now_frame_index=self.last_frame,
                                    confidence=1., initial=True, box_source='initial_ground_truth')
        self.initialized = True

    @torch.no_grad()
    def track(self, image, frame_index, available=(True, True)):
        if not self.initialized or int(frame_index) <= self.last_frame:
            raise ValueError('Initialize first, then advance frame indices monotonically')
        if len(available) != 2 or any(value not in (0, 1, False, True) for value in available):
            raise ValueError('Availability is ordered RGB,TIR')
        self.last_frame = int(frame_index)
        if not any(available):
            # No decoder, encoder, reconstruction, or cache/template write is executed.
            return self.pixel_box(), self.last_confidence, dict(both_missing=True, reconstruction_steps=0)
        if image is None or tuple(image.shape[-2:]) != self.frame_hw:
            raise ValueError('An available image with the initialized frame geometry is required')
        crop = crop_from_box(self.last_box, self.frame_hw, 4.)
        search = normalize(crop_image(image, crop, self.search_hw))[None].to(self.device)
        history = (self.history.read(self.track_id, now_frame_index=self.last_frame, device=self.device)
                   if self.history is not None else None)
        output = self.model(self.z, search, self.d, self.z_mask, self.d_mask,
                            torch.tensor([available], device=self.device), crop[None].to(self.device),
                            torch.tensor([self.last_frame], device=self.device), [history], self.steps)
        boxes, confidence = decode_prediction(output, crop[None].to(self.device))
        self.last_box, self.last_confidence = boxes[0].cpu(), float(confidence[0])
        sample = output['samples'][0]
        accepted = self.last_confidence > self.template_threshold
        if accepted:
            template_crop = crop_from_box(self.last_box, self.frame_hw, 2.)
            updated = normalize(crop_image(image, template_crop, self.template_hw))[None].to(self.device)
            th, tw = self.d_mask.shape[-2:]
            mask = foreground_mask(self.last_box, template_crop, (th, tw)).to(self.device)
            for m, real in enumerate(available):
                if real:
                    self.d[:, m * 3:(m + 1) * 3].copy_(updated[:, m * 3:(m + 1) * 3])
                    self.d_mask[:, m].copy_(mask)
        written = 0
        if self.history is not None and accepted:
            for observation in sample['observations']:
                written += self.history.record(self.track_id, observation, self.last_box,
                                                now_frame_index=self.last_frame, confidence=self.last_confidence)
        diagnostics = dict(both_missing=False, available=list(map(bool, available)),
                           reconstruction_steps=self.steps if sample['reconstruction'] else 0,
                           fusion_gate=float(sample['fusion_gate'].mean()), history_writes=written,
                           template_updated=accepted)
        return self.pixel_box(), self.last_confidence, diagnostics

    def pixel_box(self):
        height, width = self.frame_hw
        return (self.last_box * torch.tensor([width, height, width, height])).numpy().copy()
