import torch
from torch import nn
from torch.nn import functional as F

from trackit.criteria.methods.box_with_score_map import SimpleCriteria
from trackit.criteria.modules.iou_loss import GIoULoss
from ..dinov3_rgb.history import CausalFeatureHistory
from ..dinov3_rgb.alignment import reconstruction_target, future_target, masked_cosine_loss
from .model import decode_prediction
from .modules import MODALITIES


def new_history(model):
    return CausalFeatureHistory(model.reference.space_id, model.tracker.backbone.embed_dim,
                               roi_hw=model.temporal_options['roi_hw'])


def forward_clip(model, inputs, reconstruction_steps=5):
    """Only this dictionary enters the causal forward path. No future/hidden labels."""
    batch, length = inputs['context'].shape[:2]
    histories = [new_history(model) for _ in range(batch)] if model.stage == 'temporal' else None
    if histories is not None:
        for index, history in enumerate(histories):
            history.start_track(index, initialization_frame_index=int(inputs['frame_indices'][index, 0]))
        for m, modality in enumerate(MODALITIES):
            observations = model.observe_reference(inputs['context'][:, 0, m * 3:(m + 1) * 3], modality,
                                                    inputs['frame_indices'][:, 0], inputs['crop_box'])
            for index, history in enumerate(histories):
                history.record(index, observations.select(index), inputs['initial_box'][index],
                               now_frame_index=int(inputs['frame_indices'][index, 0]), confidence=1.,
                               initial=True, box_source='initial_ground_truth')
    def frame(t):
        reads = ([history.read(index, now_frame_index=int(inputs['frame_indices'][index, t]),
                               device=inputs['z'].device) for index, history in enumerate(histories)]
                 if histories is not None else None)
        return model(z=inputs['z'], x=inputs['context'][:, t], d=inputs['z'],
                     z_feat_mask=inputs['template_mask'], d_feat_mask=inputs['template_mask'],
                     modality_available=inputs['availability'][:, t], crop_boxes=inputs['crop_box'],
                     frame_indices=inputs['frame_indices'][:, t], histories=reads,
                     reconstruction_steps=reconstruction_steps)
    if histories is not None:
        # Warm-up frames produce real observations and predicted boxes, never GT ROIs.
        device_type = inputs['z'].device.type
        # A no-grad first use must not cache detached casts of trainable weights
        # for the later grad-enabled current frame inside the SAME autocast scope.
        with torch.no_grad(), torch.autocast(device_type, enabled=torch.is_autocast_enabled(device_type),
                                             dtype=torch.get_autocast_dtype(device_type), cache_enabled=False):
            for t in range(1, length - 1):
                output = frame(t)
                boxes, confidence = decode_prediction(output, inputs['crop_box'])
                for index, history in enumerate(histories):
                    for observation in output['samples'][index]['observations']:
                        history.record(index, observation, boxes[index],
                                       now_frame_index=int(inputs['frame_indices'][index, t]),
                                       confidence=float(confidence[index]))
    return frame(length - 1)


def tracking_targets(boxes, grid_hw):
    height, width = grid_hw
    yy, xx = torch.meshgrid((torch.arange(height, device=boxes.device) + .5) / height,
                            (torch.arange(width, device=boxes.device) + .5) / width, indexing='ij')
    centers = torch.stack((xx.flatten(), yy.flatten()), 1)
    positive = ((centers[None] >= boxes[:, None, :2]) & (centers[None] <= boxes[:, None, 2:])).all(-1)
    for index in range(len(boxes)):
        if not positive[index].any():
            middle = ((boxes[index, :2] + boxes[index, 2:]) / 2).clamp(0, 1)
            positive[index, (centers - middle).square().sum(-1).argmin()] = True
    row, column = positive.nonzero(as_tuple=True)
    return dict(boxes=boxes, num_positive_samples=boxes.new_tensor([len(row)]),
                positive_sample_batch_dim_indices=row, positive_sample_map_dim_indices=column)


class MissingTrackingLoss(nn.Module):
    def __init__(self, reconstruction_weight=.5, appearance_weight=.5, motion_weight=.1, visibility_weight=.05):
        super().__init__()
        self.tracking = SimpleCriteria(nn.BCEWithLogitsLoss(reduction='none'), GIoULoss(), True,
                                       1., 1., 'bce', 'giou', log_diagnostics=True)
        self.weights = dict(reconstruction=reconstruction_weight, future_appearance=appearance_weight,
                            future_motion=motion_weight, future_visibility=visibility_weight)

    def forward(self, model, output, targets):
        height, width = output['score_map'].shape[-2:]
        tracking = self.tracking(output, tracking_targets(targets['current_local_box'], (height, width)))
        total = tracking.loss
        metrics = {**tracking.metrics, **tracking.extra_metrics}
        if model.stage == 'adapt':
            metrics['loss'] = float(total.detach())
            return total, metrics
        terms = {name: [] for name in self.weights}
        valid_reconstruction, valid_appearance, valid_motion = 0, 0, 0
        samples = output['samples']
        crops = torch.cat([sample['crop_box'] for sample in samples])
        frames = torch.cat([sample['frame_index'] for sample in samples])
        anchors = torch.cat([sample['anchor_box'] for sample in samples])
        gates = []
        # Batch labels by modality/horizon. Isolated attention never crosses samples.
        for m, modality in enumerate(MODALITIES):
            selected = [index for index, sample in enumerate(samples)
                        if sample['missing_modality'] == modality and sample['reconstruction']]
            if selected:
                supervised = model.supervise_reference(targets['current_image'][selected, m * 3:(m + 1) * 3],
                                                        modality, frames[selected], crops[selected])
                target = reconstruction_target(supervised, frames[selected], crops[selected])
                valid_reconstruction += int(target.valid_mask.sum())
                terms['reconstruction'].append(torch.stack([
                    masked_cosine_loss(torch.cat([samples[index]['reconstruction'][step] for index in selected]),
                                       target.tokens, target.valid_mask)
                    for step in samples[selected[0]]['reconstruction']]).mean())
                gates.extend(samples[index]['fusion_gate'].detach().mean() for index in selected)
        for h, horizon in enumerate(model.temporal_options['horizons']):
            target_index = horizon - 1
            for m, modality in enumerate(MODALITIES):
                # Privileged images are read only after the causal forward has finished.
                features = model.supervise_reference(
                    targets['future_images'][:, target_index, m * 3:(m + 1) * 3], modality,
                    targets['future_indices'][:, target_index], crops)
                label = future_target(features, anchor_frame_indices=frames, anchor_crop_boxes=crops,
                                      anchor_boxes=anchors, future_boxes=targets['future_boxes'][:, target_index],
                                      horizon=horizon, roi_hw=model.temporal_options['roi_hw'])
                valid_appearance += int(label.appearance.valid_mask.sum())
                valid_motion += int(label.motion_valid.sum())
                terms['future_appearance'].append(masked_cosine_loss(
                    torch.cat([sample['future'].appearance[:, h, m] for sample in samples]),
                    label.appearance.tokens, label.appearance.valid_mask))
                motion = F.smooth_l1_loss(torch.cat([sample['future'].motion[:, h, m] for sample in samples]).float(),
                                         label.motion, reduction='none').mean(-1)
                terms['future_motion'].append((motion * label.motion_valid).sum() / label.motion_valid.sum().clamp_min(1))
                # Predict whether appearance is usable inside the anchored search view.
                terms['future_visibility'].append(F.binary_cross_entropy_with_logits(
                    torch.cat([sample['future'].visibility[:, h, m] for sample in samples]).float(),
                    label.appearance_valid.float()))
        for name, weight in self.weights.items():
            if terms[name]:
                value = torch.stack(terms[name]).mean()
                total = total + weight * value
                metrics[f'Loss/{name}'] = float(value.detach())
        metrics.update(loss=float(total.detach()), reconstruction_valid_tokens=valid_reconstruction,
                       future_valid_roi_tokens=valid_appearance, future_valid_motion_targets=valid_motion,
                       fusion_gate=float(torch.stack(gates).mean()) if gates else 0.)
        return total, metrics
