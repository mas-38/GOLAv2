from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from test_dinov3_rgb import model, inputs
from trackit.models.methods.dinov3_rgb.backbone import TrackingCrop
from trackit.models.methods.dinov3_rgb.reference import FrozenReferenceEncoder
from trackit.models.methods.dinov3_rgb.feature_contract import FeatureObservation, FeatureSupervision
from trackit.models.methods.dinov3_rgb.alignment import (
    align_target_roi, reconstruction_target, future_target, masked_cosine_loss,
    crop_boxes_from_siamfc, siamfc_from_crop_boxes)
from trackit.models.methods.dinov3_rgb.history import CausalFeatureHistory


def plane_features(crop=(0., 0., 1., 1.), frame=10, modality='rgb', object_box=None):
    """A known coordinate field for testing geometry independently of the encoder."""
    crop = torch.tensor([crop])
    y, x = torch.meshgrid((torch.arange(8) + .5) / 8, (torch.arange(8) + .5) / 8, indexing='ij')
    points = torch.stack((x, y), dim=-1)
    image_points = crop[0, :2] + points * (crop[0, 2:] - crop[0, :2])
    if object_box is not None:
        box = torch.tensor(object_box)
        image_points = (image_points - box[:2]) / (box[2:] - box[:2])
    return FeatureSupervision(image_points.reshape(1, 64, 2), (8, 8), crop,
                              torch.tensor([frame]), modality, 'test-reference-space')


def as_observation(features, frame=None):
    return FeatureObservation(features.tokens, features.grid_hw, features.crop_boxes,
                              features.frame_indices if frame is None else torch.tensor([frame]),
                              features.modality, features.space_id)


class TemporalFeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stage_exports_preserve_predictions_and_checkpoint_keys(self):
        net = model().eval()
        state_keys = set(net.state_dict())
        data = inputs(1)
        before = net(**data)
        bundle = net.extract_feature_bundle(**data, include_isolated=True)
        self.assertEqual(bundle.grids, {'z': (2, 2), 'd': (2, 2), 'x': (4, 4)})
        self.assertEqual(set(bundle.independent), {'z', 'd', 'x'})
        self.assertEqual(bundle.isolated['x'].shape, (1, 16, 32))
        torch.testing.assert_close(net.predict_from_features(bundle.joint['x'])['boxes'], before['boxes'])
        self.assertEqual(set(net.state_dict()), state_keys)
        with self.assertRaises(ValueError):
            net.predict_from_features(bundle.isolated['z'])

    def test_isolated_features_never_attend_to_other_current_modalities(self):
        net = model().eval()
        net.backbone.add_modality('tir')
        image = inputs(1)['x']
        first = net.backbone.encode_isolated_crops({
            'rgb': TrackingCrop(image, 'x', 'rgb'),
            'tir': TrackingCrop(torch.randn_like(image), 'x', 'tir'),
        })
        second = net.backbone.encode_isolated_crops({
            'rgb': TrackingCrop(image, 'x', 'rgb'),
            'tir': TrackingCrop(torch.full_like(image, float('nan')), 'x', 'tir'),
        })
        torch.testing.assert_close(first['rgb'], second['rgb'])
        solo = net.backbone.encode_isolated_crops({'rgb': TrackingCrop(image, 'x', 'rgb')})
        torch.testing.assert_close(first['rgb'], solo['rgb'])

    def test_reference_is_frozen_deterministic_and_versioned(self):
        net = model().train()
        net.backbone.blocks[0].adapters['rgb'].q.dropout.p = .5
        net.backbone.rope_embed.rescale_coords = 2
        reference = FrozenReferenceEncoder.from_tracker(net).train()
        self.assertFalse(reference.training)
        self.assertFalse(any(module.training for module in reference.modules()))
        self.assertFalse(any(p.requires_grad for p in reference.parameters()))
        image = inputs(1)['x']
        first = reference.observe(image, 'rgb', [10], [[0, 0, 1, 1]])
        second = reference.observe(image, 'rgb', [10], [[0, 0, 1, 1]])
        torch.testing.assert_close(first.tokens, second.tokens)
        self.assertFalse(first.tokens.requires_grad)
        old_id = reference.space_id
        with torch.no_grad():
            net.backbone.role_embed.add_(.1)
        unchanged = reference.observe(image, 'rgb', [10], [[0, 0, 1, 1]])
        torch.testing.assert_close(first.tokens, unchanged.tokens)
        self.assertEqual(reference.space_id, old_id)
        self.assertNotEqual(FrozenReferenceEncoder.from_tracker(net).space_id, old_id)
        different_norm = FrozenReferenceEncoder(reference.encoder, search_grid_hw=(4, 4),
                                                normalization_profile='different-preprocessing')
        self.assertNotEqual(different_norm.space_id, old_id)
        changed_coordinates = FrozenReferenceEncoder(reference.encoder, search_grid_hw=(4, 4))
        changed_coordinates.encoder.rope_embed.normalize_coords = 'min'
        self.assertNotEqual(FrozenReferenceEncoder(changed_coordinates.encoder,
                                                 search_grid_hw=(4, 4)).space_id, old_id)
        restored = FrozenReferenceEncoder.from_tracker(net)
        restored.load_state_dict(reference.state_dict())
        self.assertEqual(restored.space_id, reference.space_id)
        with patch.object(reference.encoder.patch_embed, 'forward') as embed:
            with self.assertRaises(ValueError):
                reference.observe(torch.full_like(image, float('nan')), 'rgb', [10], [[0, 0, 1, 1]],
                                  real_available=False)
            embed.assert_not_called()

    def test_context_reuses_solo_features_but_only_search_can_be_observed(self):
        net = model().eval()
        reference = FrozenReferenceEncoder.from_tracker(net)
        data = inputs(1)
        crops = {'z': TrackingCrop(data['z'], 'z', foreground_mask=data['z_feat_mask']),
                 'x': TrackingCrop(data['x'], 'x')}
        context = reference.encode_context(crops)
        observation = reference.observation_from_context(context, 'x', [3], [[0, 0, 1, 1]])
        direct = reference.observe(data['x'], 'rgb', [3], [[0, 0, 1, 1]])
        torch.testing.assert_close(observation.tokens, direct.tokens)
        self.assertIsInstance(observation, FeatureObservation)
        # Frozen cached K/V must remain usable by a trainable downstream module;
        # inference-mode tensors would fail when backward needs to save them.
        projection = torch.nn.Linear(32, 4)
        projection(observation.tokens).sum().backward()
        self.assertTrue(torch.isfinite(projection.weight.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        supervision = reference.supervise(data['x'], 'rgb', [4], [[0, 0, 1, 1]])
        self.assertIsInstance(supervision, FeatureSupervision)
        self.assertNotIsInstance(supervision, FeatureObservation)
        with self.assertRaises(ValueError):
            reference.observation_from_context(context, 'z', [3], [[0, 0, 1, 1]])

    def test_roi_coordinates_and_crop_padding_are_explicit(self):
        feature = plane_features(crop=(.2, .1, .8, .7))
        box = torch.tensor([[.35, .25, .65, .55]])
        aligned = align_target_roi(feature, box)
        y, x = torch.meshgrid((torch.arange(4) + .5) / 4, (torch.arange(4) + .5) / 4, indexing='ij')
        expected = box[0, :2] + torch.stack((x, y), dim=-1) * (box[0, 2:] - box[0, :2])
        torch.testing.assert_close(aligned.tokens, expected.reshape(1, 16, 2), atol=1e-6, rtol=1e-6)
        self.assertTrue(aligned.valid_mask.all())
        # Changing both image-space crop and ROI origin preserves local sampling.
        delta = torch.tensor([[.05, .05, .05, .05]])
        shifted = replace(feature, crop_boxes=feature.crop_boxes + delta)
        torch.testing.assert_close(align_target_roi(shifted, box + delta).tokens, aligned.tokens)
        padded = plane_features(crop=(-.5, -.5, .5, .5))
        label = reconstruction_target(padded, [10], padded.crop_boxes)
        self.assertEqual(label.valid_mask.sum().item(), 16)  # 4x4 of an 8x8 map is real image.

    def test_crop_metadata_round_trip_handles_padding_and_frame_resolution(self):
        params = torch.tensor([[[2., 2.], [-100., -40.]]])
        boxes = crop_boxes_from_siamfc(params, [640, 480], [256, 256])
        expected = torch.tensor([[50/640, 20/480, 178/640, 148/480]])
        torch.testing.assert_close(boxes, expected)
        future_params = siamfc_from_crop_boxes(boxes, [1280, 960], [256, 256])
        torch.testing.assert_close(crop_boxes_from_siamfc(future_params, [1280, 960], [256, 256]), boxes)
        with self.assertRaises(ValueError):
            crop_boxes_from_siamfc([[[-1, 1], [0, 0]]], [640, 480], [256, 256])

    def test_future_appearance_alignment_keeps_motion_separate(self):
        current_box = [.2, .2, .4, .4]
        next_box = [.3, .25, .5, .45]
        first = plane_features(frame=10, object_box=current_box)
        second = plane_features(frame=11, object_box=next_box)
        target = future_target(second, anchor_frame_indices=[10], anchor_crop_boxes=first.crop_boxes,
                               anchor_boxes=torch.tensor([current_box]), future_boxes=torch.tensor([next_box]),
                               horizon=1)
        torch.testing.assert_close(target.appearance.tokens, align_target_roi(first, torch.tensor([current_box])).tokens,
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(target.motion, torch.tensor([[.5, .25, 0., 0.]]), atol=1e-6, rtol=1e-6)
        self.assertTrue(target.appearance_valid.item())
        self.assertTrue(target.motion_valid.item())
        self.assertFalse(target.motion.requires_grad)
        self.assertEqual(target.anchor_box_source, 'baseline_prediction')
        with self.assertRaises(ValueError):
            future_target(second, anchor_frame_indices=[10], anchor_crop_boxes=first.crop_boxes,
                          anchor_boxes=torch.tensor([current_box]), future_boxes=torch.tensor([next_box]),
                          horizon=1, anchor_box_source='final_fused_prediction')
        with self.assertRaises(ValueError):
            future_target(second, anchor_frame_indices=[10], anchor_crop_boxes=first.crop_boxes,
                          anchor_boxes=torch.tensor([current_box]), future_boxes=torch.tensor([next_box]), horizon=2)
        with self.assertRaises(ValueError):
            recentered = replace(second, crop_boxes=torch.tensor([[.1, .1, .9, .9]]))
            future_target(recentered, anchor_frame_indices=[10], anchor_crop_boxes=first.crop_boxes,
                          anchor_boxes=torch.tensor([current_box]), future_boxes=torch.tensor([next_box]), horizon=1)

    def test_future_out_of_view_occlusion_and_masked_loss(self):
        feature = plane_features(crop=(0., 0., .5, .5), frame=12)
        target = future_target(feature, anchor_frame_indices=[10], anchor_crop_boxes=feature.crop_boxes,
                               anchor_boxes=torch.tensor([[.1, .1, .3, .3]]),
                               future_boxes=torch.tensor([[.7, .7, .9, .9]]), horizon=2)
        self.assertFalse(target.appearance_valid.item())
        self.assertFalse(target.appearance.valid_mask.any())
        self.assertTrue(target.motion_valid.item())
        occluded = future_target(feature, anchor_frame_indices=[10], anchor_crop_boxes=feature.crop_boxes,
                                 anchor_boxes=torch.tensor([[.1, .1, .3, .3]]),
                                 future_boxes=torch.full((1, 4), float('nan')), horizon=2, visible=[False])
        self.assertFalse(occluded.motion_valid.item())
        prediction = torch.full_like(occluded.appearance.tokens, float('nan'), requires_grad=True)
        loss = masked_cosine_loss(prediction, occluded.appearance.tokens, occluded.appearance.valid_mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(prediction.grad.count_nonzero().item(), 0)
        valid = torch.ones(1, 16, dtype=torch.bool)
        data = torch.randn(1, 16, 8)
        self.assertAlmostEqual(masked_cosine_loss(data, data, valid).item(), 0, places=6)

    def test_reconstruction_is_current_frame_privileged_supervision_only(self):
        feature = plane_features(frame=10, modality='tir')
        label = reconstruction_target(feature, [10], feature.crop_boxes)
        torch.testing.assert_close(label.tokens, feature.tokens)
        self.assertEqual(label.modality, 'tir')
        with self.assertRaises(ValueError):
            reconstruction_target(feature, [9], feature.crop_boxes)
        with self.assertRaises(TypeError):
            reconstruction_target(as_observation(feature), [10], feature.crop_boxes)

    def test_history_is_causal_bounded_detached_and_per_track_per_modality(self):
        feature = plane_features(frame=0)
        history = CausalFeatureHistory(feature.space_id, 2, capacity_per_modality=3,
                                       min_confidence=.84, max_recent_age=5)
        history.start_track('a')
        box = [[.2, .2, .8, .8]]
        self.assertTrue(history.record('a', as_observation(feature), box, now_frame_index=0,
                                       confidence=1., initial=True, box_source='initial_ground_truth'))
        for frame in (1, 2, 3):
            self.assertTrue(history.record('a', as_observation(feature, frame), box,
                                           now_frame_index=frame, confidence=.9))
        cached = history.read('a', now_frame_index=4)
        self.assertEqual(cached.frame_indices.tolist(), [0, 2, 3])
        self.assertEqual(cached.ages.tolist(), [4, 2, 1])
        self.assertEqual(cached.tokens.shape, (3, 16, 2))
        self.assertFalse(cached.tokens.requires_grad)
        cached.tokens.fill_(-100)
        self.assertGreater(history.read('a', now_frame_index=4).tokens.min().item(), 0)
        self.assertEqual(history.read('a', now_frame_index=2).frame_indices.tolist(), [0])
        self.assertEqual(history.read('a', now_frame_index=20).frame_indices.tolist(), [0])
        self.assertEqual(history.read('other', now_frame_index=4).tokens.shape[0], 0)
        tir = replace(as_observation(feature, 3), modality='tir')
        self.assertTrue(history.record('a', tir, box, now_frame_index=3, confidence=.9))
        self.assertEqual(history.read('a', now_frame_index=4, modalities=['tir']).modalities, ('tir',))
        self.assertFalse(history.record('a', as_observation(feature, 4), box,
                                        now_frame_index=4, confidence=.8))
        history.reset('a')
        self.assertEqual(history.read('a', now_frame_index=5).tokens.numel(), 0)

    def test_history_rejects_supervision_future_writes_and_space_changes(self):
        feature = plane_features(frame=3)
        history = CausalFeatureHistory(feature.space_id, 2)
        history.start_track('a')
        kwargs = dict(track_id='a', target_box=[[.2, .2, .8, .8]], now_frame_index=3, confidence=.9)
        with self.assertRaises(TypeError):
            history.record(observation=feature, **kwargs)
        with self.assertRaises(ValueError):
            history.record(observation=as_observation(feature, 4), **kwargs)
        with self.assertRaises(ValueError):
            history.record(observation=replace(as_observation(feature), space_id='different'), **kwargs)
        with self.assertRaises(ValueError):
            history.record(observation=as_observation(feature), box_source='future_ground_truth', **kwargs)
        self.assertTrue(history.record(observation=as_observation(feature), **kwargs))
        with self.assertRaises(ValueError):
            history.record(observation=as_observation(feature), **kwargs)
        with self.assertRaises(ValueError):
            history.record(observation=as_observation(feature), initial=True,
                           box_source='initial_ground_truth', **kwargs)


if __name__ == '__main__':
    unittest.main()
