from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from safetensors.torch import save_file
import torch

from trackit.models.backbone.dinov3.builder import import_official_dinov3
from trackit.models.methods.dinov3_rgb.model import DINOv3RGBTracker
from trackit.models.methods.dinov3_missing.model import DINOv3MissingTracker
from trackit.models.methods.dinov3_missing.runtime import RGBTSequenceTracker
from trackit.models.methods.dinov3_missing.pipeline import forward_clip, MissingTrackingLoss, new_history
from trackit.models.methods.dinov3_missing.checkpoint import (
    build_model, file_sha256, save_checkpoint, read_checkpoint, restore_training_state, training_signature)
from trackit.runner.training.rgbt_temporal import make_optimization, update_parameters, train_stage
from trackit.data.methods.rgbt_temporal import LasHeRSequence, LasHeRTemporalClips, collate_clips, crop_image


def small_vit(**_):
    module = import_official_dinov3('../dinov3')
    vit = module.DinoVisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=4,
                                       num_heads=4, n_storage_tokens=4, mask_k_bias=True,
                                       layerscale_init=1e-5, pos_embed_rope_dtype='fp32')
    vit.init_weights()
    return vit


def small_model(temporal=False):
    tracker = DINOv3RGBTracker(small_vit(), (2, 2), (4, 4), independent_depth=2,
                              rank=2, alpha=4, dropout=.1)
    tracker.backbone.add_modality('tir')
    model = DINOv3MissingTracker(tracker, hidden=16, heads=4, roi_hw=(2, 2))
    if temporal:
        model.activate_temporal('synthetic-unit-test-adaptation')
    return model


def clip(batch=2):
    generator = torch.Generator().manual_seed(772)
    inputs = dict(z=torch.randn(batch, 6, 32, 32, generator=generator),
                  context=torch.randn(batch, 4, 6, 64, 64, generator=generator),
                  availability=torch.tensor([[[1, 1], [1, 0], [0, 1], [1, 0]]] * batch).bool(),
                  template_mask=torch.ones(batch, 2, 2, dtype=torch.long),
                  crop_box=torch.tensor([[0., 0., 1., 1.]] * batch), frame_indices=torch.arange(4).expand(batch, -1),
                  initial_box=torch.tensor([[.2, .2, .7, .7]] * batch))
    for t in range(4):
        for m in range(2):
            inputs['context'][~inputs['availability'][:, t, m], t, m * 3:(m + 1) * 3] = float('nan')
    targets = dict(current_local_box=torch.tensor([[.2, .2, .7, .7]] * batch),
                   current_image=torch.randn(batch, 6, 64, 64, generator=generator),
                   future_images=torch.randn(batch, 2, 6, 64, 64, generator=generator),
                   future_boxes=torch.tensor([[[.2, .2, .7, .7], [.25, .2, .75, .7]]] * batch),
                   future_indices=torch.tensor([[4, 5]] * batch))
    return inputs, targets


def config(directory):
    initial = Path(directory) / 'rgbt.safetensors'
    save_file(small_model().tracker.state_dict(), str(initial))
    return dict(model=dict(official_source='../dinov3', official_revision='unit-test',
                           initialization=str(initial), initialization_sha256=file_sha256(initial),
                           template_feat_size=[2, 2], search_feat_size=[4, 4],
                           tracking=dict(independent_depth=2, rank=2, alpha=4, dropout=.1),
                           temporal=dict(hidden=16, heads=4, horizons=[1, 2], roi_hw=[2, 2])),
                stages={name:dict(epochs=2, samples_per_epoch=12, batch_size=2, lr=.001, min_lr=.00001,
                                  warmup_epochs=1, weight_decay=.05, head_lr_multiplier=.1 if name == 'temporal' else 1.)
                        for name in ('adapt', 'temporal')},
                data=dict(root='synthetic', history_frames=3, workers=0), seed=42,
                runtime=dict(precision='float32', max_grad_norm=1., log_interval=100,
                             checkpoint_interval=2, train_reconstruction_steps=5,
                             inference_reconstruction_steps=3, template_update_threshold=.84,
                             torch_threads=2), loss={}, wandb=dict(enabled=False), version='synthetic-test',
                evaluation=dict(synthetic_missing_persistence=.8, modes=['complete']))


class SyntheticClips(torch.utils.data.Dataset):
    def __init__(self, _, *, samples, **kwargs):
        self.samples, self.epoch, self.names = samples, 0, ['synthetic']

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        inputs, targets = clip(1)
        inputs = {key: value[0] for key, value in inputs.items()}
        targets = {key: value[0] for key, value in targets.items()}
        inputs['context'] += (index + self.epoch) * .001
        return dict(inputs=inputs, targets=targets)


class DINOv3MissingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cv2.setNumThreads(0)

    def test_missing_current_pixels_are_never_encoded(self):
        model = small_model().eval()
        inputs, _ = clip(3)
        inputs['availability'][:, -1] = torch.tensor([[1, 0], [0, 1], [1, 1]])
        inputs['context'][:, -1] = torch.randn(3, 6, 64, 64)
        before = forward_clip(model, inputs)
        inputs['context'][0, -1, 3:] = float('nan')
        inputs['context'][1, -1, :3] = float('nan')
        after = forward_clip(model, inputs)
        self.assertTrue(torch.isfinite(after['boxes']).all())
        torch.testing.assert_close(before['boxes'], after['boxes'], rtol=0, atol=0)
        torch.testing.assert_close(before['score_map'], after['score_map'], rtol=0, atol=0)

    def test_all_temporal_losses_reach_modules_but_not_the_frozen_backbone(self):
        model = small_model(temporal=True).train()
        inputs, targets = clip()
        output = forward_clip(model, inputs, 5)
        loss, metrics = MissingTrackingLoss()(model, output, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics['reconstruction_valid_tokens'], 0)
        self.assertGreater(metrics['future_valid_roi_tokens'], 0)
        for name in ('future_predictor', 'reconstructor', 'fusion'):
            parameters = list(getattr(model, name).parameters())
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), name)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
        self.assertTrue(all(p.grad is None for p in model.reference.parameters()))
        self.assertTrue(all(p.grad is None for p in model.tracker.backbone.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.tracker.head.parameters()))
        self.assertFalse(model.reference.training)
        self.assertFalse(model.tracker.backbone.training)

    def test_shared_five_step_cell_and_zero_budget_bypass(self):
        model = small_model(temporal=True).eval()
        inputs, _ = clip(1)
        with patch.object(model.reconstructor.current_attention, 'forward',
                          wraps=model.reconstructor.current_attention.forward) as call:
            output = forward_clip(model, inputs, 5)
        # Three calls (five iterations each): two past frames and the current frame.
        self.assertEqual(call.call_count, 15)
        self.assertEqual(set(output['samples'][0]['reconstruction']), {1, 3, 5})
        with patch.object(model.reconstructor, 'forward', wraps=model.reconstructor.forward) as call:
            direct = forward_clip(model, inputs, 0)
        self.assertEqual(call.call_count, 0)
        self.assertFalse(direct['samples'][0]['reconstruction'])
        # Zero-initialized Phi preserves direct tracking until its residual learns.
        torch.testing.assert_close(output['boxes'], direct['boxes'], rtol=0, atol=0)

    def test_privileged_supervision_changes_loss_without_changing_inference(self):
        model = small_model(temporal=True).eval()
        inputs, targets = clip(1)
        before = forward_clip(model, inputs)
        first_loss, _ = MissingTrackingLoss()(model, before, targets)
        targets['future_images'] = torch.randn_like(targets['future_images']) * 3
        targets['current_image'] = torch.randn_like(targets['current_image']) * 3
        after = forward_clip(model, inputs)
        second_loss, _ = MissingTrackingLoss()(model, after, targets)
        torch.testing.assert_close(before['samples'][0]['future'].appearance,
                                   after['samples'][0]['future'].appearance, rtol=0, atol=0)
        torch.testing.assert_close(before['boxes'], after['boxes'], rtol=0, atol=0)
        self.assertNotEqual(float(first_loss.detach()), float(second_loss.detach()))

    def test_online_both_missing_and_per_modality_template_retention(self):
        model = small_model(temporal=True).eval()
        tracker = RGBTSequenceTracker(model, reconstruction_steps=3, template_threshold=0.)
        image = torch.rand(6, 96, 128)
        tracker.initialize(image, np.array([25., 20., 70., 75.]))
        old_tir = tracker.d[:, 3:].clone()
        old_mask = tracker.d_mask[:, 1].clone()
        previous = tracker.pixel_box()
        with patch.object(model, 'forward', wraps=model.forward) as call:
            box, _, diagnostics = tracker.track(None, 1, (False, False))
        self.assertEqual(call.call_count, 0)
        np.testing.assert_array_equal(box, previous)
        self.assertTrue(diagnostics['both_missing'])
        image[3:] = float('nan')
        with patch.object(model.reference, 'observe', wraps=model.reference.observe) as call:
            box, _, diagnostics = tracker.track(image, 2, (True, False))
        self.assertTrue(np.isfinite(box).all())
        self.assertEqual([c.args[1] for c in call.call_args_list], ['rgb'])
        torch.testing.assert_close(tracker.d[:, 3:], old_tir, rtol=0, atol=0)
        torch.testing.assert_close(tracker.d_mask[:, 1], old_mask, rtol=0, atol=0)

    def test_future_and_wrong_space_history_are_rejected(self):
        model = small_model(temporal=True).eval()
        inputs, _ = clip(1)
        history = new_history(model)
        history.start_track('track', initialization_frame_index=0)
        observation = model.reference.observe(inputs['context'][:, 0, :3], 'rgb', [0], inputs['crop_box'])
        history.record('track', observation, inputs['initial_box'][0], now_frame_index=0,
                        confidence=1., initial=True, box_source='initial_ground_truth')
        read = history.read('track', now_frame_index=3)
        kwargs = dict(z=inputs['z'], x=inputs['context'][:, -1], d=inputs['z'],
                       z_feat_mask=inputs['template_mask'], d_feat_mask=inputs['template_mask'],
                       modality_available=inputs['availability'][:, -1], crop_boxes=inputs['crop_box'],
                       frame_indices=inputs['frame_indices'][:, -1])
        with self.assertRaisesRegex(ValueError, 'different frozen'):
            model(**kwargs, histories=[replace(read, space_id='wrong-space')])
        with self.assertRaisesRegex(ValueError, 'causal history'):
            model(**kwargs, histories=[replace(read, frame_indices=torch.tensor([3]))])

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_exact_optimizer_scheduler_scaler_rng_resume_for_both_stages(self):
        with TemporaryDirectory() as temporary, patch(
                'trackit.models.methods.dinov3_missing.checkpoint.build_dino_v3_backbone', side_effect=small_vit):
            cfg = config(temporary)
            for stage in ('adapt', 'temporal'):
                with self.subTest(stage=stage):
                    model = build_model(cfg)
                    if stage == 'temporal':
                        model.activate_temporal('synthetic-trained-adaptation-test')
                    model.train()
                    optimizer, scheduler = make_optimization(model, cfg['stages'][stage], device='cpu', steps_per_epoch=6)
                    scaler = torch.amp.GradScaler('cpu', init_scale=128, growth_interval=1)
                    criterion = MissingTrackingLoss()
                    inputs, targets = clip(1)
                    def step(net, opt, sched, amp):
                        opt.zero_grad(set_to_none=True)
                        output = forward_clip(net, inputs)
                        loss, _ = criterion(net, output, targets)
                        update_parameters(loss, net, opt, sched, amp, 1.)
                        return float(loss.detach()), random.random(), float(np.random.random())
                    for _ in range(2):
                        step(model, optimizer, scheduler, scaler)
                    signature = training_signature(cfg, stage, 'synthetic-manifest')
                    path = Path(temporary) / f'{stage}.pt'
                    save_checkpoint(path, model, optimizer, scheduler, scaler, config=cfg, epoch=0, next_batch=2,
                                    optimizer_updates=2, signature=signature)
                    checkpoint = read_checkpoint(path)
                    expected = [step(model, optimizer, scheduler, scaler) for _ in range(3)]
                    expected_weights = deepcopy(model.state_dict())
                    expected_opt = deepcopy(optimizer.state_dict())
                    expected_scheduler, expected_scaler = scheduler.state_dict(), scaler.state_dict()
                    restored = build_model(cfg, stage, path).train()
                    opt, sched = make_optimization(restored, cfg['stages'][stage], device='cpu', steps_per_epoch=6)
                    amp = torch.amp.GradScaler('cpu', init_scale=2, growth_interval=100)
                    self.assertEqual(restore_training_state(checkpoint, opt, sched, amp, signature), (0, 2, 2))
                    actual = [step(restored, opt, sched, amp) for _ in range(3)]
                    self.assertEqual(actual, expected)
                    self.assert_nested_equal(expected_weights, restored.state_dict())
                    self.assert_nested_equal(expected_opt, opt.state_dict())
                    self.assert_nested_equal(expected_scheduler, sched.state_dict())
                    self.assert_nested_equal(expected_scaler, amp.state_dict())
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        restore_training_state(checkpoint, opt, sched, amp, 'wrong-signature')

    def test_temporal_factory_rejects_rgb_only_initialization(self):
        with TemporaryDirectory() as temporary, patch(
                'trackit.models.methods.dinov3_missing.checkpoint.build_dino_v3_backbone', side_effect=small_vit):
            cfg = config(temporary)
            with self.assertRaisesRegex(ValueError, 'adapt TIR'):
                build_model(cfg, 'temporal')
            model = build_model(cfg)
            optimizer, scheduler = make_optimization(model, cfg['stages']['adapt'], device='cpu', steps_per_epoch=6)
            path = Path(temporary) / 'untrained.pt'
            save_checkpoint(path, model, optimizer, scheduler, torch.amp.GradScaler('cpu'),
                            config=cfg, epoch=0, next_batch=0, optimizer_updates=0, signature='test')
            with self.assertRaisesRegex(ValueError, 'untrained copied TIR'):
                build_model(cfg, 'temporal', path)

    def test_real_training_loop_resumes_mid_epoch_and_hands_off_adapted_reference(self):
        with TemporaryDirectory() as temporary, patch(
                'trackit.models.methods.dinov3_missing.checkpoint.build_dino_v3_backbone', side_effect=small_vit), patch(
                'trackit.runner.training.rgbt_temporal.LasHeRTemporalClips', SyntheticClips), redirect_stdout(io.StringIO()):
            cfg = config(temporary)
            for settings in cfg['stages'].values():
                settings['samples_per_epoch'] = 8
            baseline = train_stage(cfg, 'adapt', Path(temporary) / 'baseline', device='cpu')
            def interrupt_after_save(path, *args, **kwargs):
                save_checkpoint(path, *args, **kwargs)
                if kwargs['epoch'] == 0 and kwargs['next_batch'] == 2:
                    raise KeyboardInterrupt()
            interrupted = Path(temporary) / 'interrupted'
            with patch('trackit.runner.training.rgbt_temporal.save_checkpoint', side_effect=interrupt_after_save):
                with self.assertRaises(KeyboardInterrupt):
                    train_stage(cfg, 'adapt', interrupted, device='cpu')
            resumed = train_stage(cfg, 'adapt', interrupted, resume=interrupted / 'last.pt', device='cpu')
            expected, actual = read_checkpoint(baseline), read_checkpoint(resumed)
            for key in ('model', 'optimizer', 'scheduler', 'scaler', 'optimizer_updates', 'rng'):
                self.assert_nested_equal(expected[key], actual[key])
            temporal = train_stage(cfg, 'temporal', Path(temporary) / 'temporal',
                                   initialization_checkpoint=resumed, device='cpu')
            saved = read_checkpoint(temporal)
            self.assertEqual(saved['stage'], 'temporal')
            self.assertEqual(saved['adaptation_source'], file_sha256(resumed))
            self.assertEqual(saved['optimizer_updates'], 8)
            self.assertTrue(saved['reference_space_id'].startswith('dinov3-isolated-final-ln-search-v1:'))
            for name, value in actual['model'].items():
                if name.startswith('tracker.backbone.'):
                    torch.testing.assert_close(value, saved['model'][name], rtol=0, atol=0)
                    reference_key = name.replace('tracker.backbone.', 'reference.encoder.')
                    torch.testing.assert_close(value, saved['model'][reference_key], rtol=0, atol=0)

    def test_evaluation_entry_uses_causal_runtime_and_exports_all_rows(self):
        from evaluate_dinov3_rgbt import evaluate
        with TemporaryDirectory() as temporary, patch(
                'trackit.models.methods.dinov3_missing.checkpoint.build_dino_v3_backbone', side_effect=small_vit), redirect_stdout(io.StringIO()):
            cfg = config(temporary)
            cfg['data']['root'] = temporary
            folder = Path(temporary) / 'testingset' / 'toy'
            for modality in ('visible', 'infrared'):
                (folder / modality).mkdir(parents=True)
                for frame in range(4):
                    cv2.imwrite(str(folder / modality / f'{frame:06d}.jpg'), np.full((64, 96, 3), 80, np.uint8))
            np.savetxt(folder / 'init.txt', [[25, 20, 20, 24]] * 4, delimiter=',')
            model = build_model(cfg)
            optimizer, scheduler = make_optimization(model, cfg['stages']['adapt'], device='cpu', steps_per_epoch=6)
            path = Path(temporary) / 'model.pt'
            save_checkpoint(path, model, optimizer, scheduler, torch.amp.GradScaler('cpu'),
                            config=cfg, epoch=1, next_batch=0, optimizer_updates=1, signature='test')
            marker = Path(temporary) / 'markers.json'
            marker.write_text(json.dumps(dict(toy=dict(data=[[1, 1], [1, 0], [0, 0], [0, 1]]))))
            output = Path(temporary) / 'eval'
            with patch('evaluate_dinov3_rgbt.split_names', return_value=['toy']):
                report = evaluate(cfg, path, output, ['schedule'], device='cpu', schedule=marker)
            self.assertEqual(report['sequences'], 1)
            self.assertTrue((output / 'report.json').exists())
            boxes = np.loadtxt(output / 'schedule/toy/toy.txt', delimiter=',')
            self.assertEqual(boxes.shape, (4, 4))
            np.testing.assert_array_equal(boxes[1], boxes[2])
            rows = [json.loads(line) for line in (output / 'schedule/toy/diagnostics.jsonl').read_text().splitlines()]
            self.assertTrue(rows[2]['both_missing'])

    def test_cli_defaults_to_preflight_and_cannot_submit_training(self):
        import finetune_dinov3_rgbt
        import yaml
        with TemporaryDirectory() as temporary:
            cfg = config(temporary)
            config_path = Path(temporary) / 'config.yaml'
            config_path.write_text(yaml.safe_dump(cfg))
            with patch('sys.argv', ['finetune_dinov3_rgbt.py', '--config', str(config_path), '--device', 'cpu']), patch(
                    'finetune_dinov3_rgbt.preflight') as preflight, patch(
                    'finetune_dinov3_rgbt.train_stage') as train:
                finetune_dinov3_rgbt.main()
                preflight.assert_called_once()
                train.assert_not_called()

    def test_dataset_uses_ordered_clips_and_fixed_initial_anchor(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'trainingset' / 'toy'
            (directory / 'visible').mkdir(parents=True)
            (directory / 'infrared').mkdir()
            for frame in range(8):
                for modality in ('visible', 'infrared'):
                    self.assertTrue(cv2.imwrite(str(directory / modality / f'{frame:06d}.jpg'),
                                                 np.full((64, 96, 3), 50 + frame, dtype=np.uint8)))
            boxes = np.array([[25 + frame, 20, 20, 24] for frame in range(8)])
            np.savetxt(directory / 'init.txt', boxes, delimiter=',', fmt='%d')
            with patch('trackit.data.methods.rgbt_temporal.split_names', return_value=['toy']):
                data = LasHeRTemporalClips(temporary, stage='temporal', samples=4, template_hw=(32, 32), search_hw=(64, 64))
                first = data[0]
                second = data[0]
                self.assertTrue(torch.equal(first['inputs']['frame_indices'], second['inputs']['frame_indices']))
                self.assertTrue(torch.equal(first['inputs']['crop_box'], second['inputs']['crop_box']))
                self.assertEqual(first['inputs']['context'].shape[0], 4)
                self.assertEqual(first['targets']['future_images'].shape[0], 2)
                current = first['inputs']['frame_indices'][-1]
                torch.testing.assert_close(first['targets']['future_indices'], current + torch.tensor([1, 2]))
                self.assertNotIn('future_images', first['inputs'])
                self.assertNotIn('current_box', first['inputs'])
                # C_t derives solely from the initial observation, independent of future GT.
                data.sequences[0].boxes[4:, :2] += 10
                data.sequences[0].boxes[4:, 2:] += 10
                altered = data[0]
                torch.testing.assert_close(first['inputs']['crop_box'], altered['inputs']['crop_box'])
                future_frame = int(first['targets']['future_indices'][0])
                data.sequences[0].boxes[future_frame] = 0
                occluded = data[0]
                self.assertEqual(int(occluded['targets']['future_indices'][0]), future_frame)
                self.assertTrue((occluded['targets']['future_boxes'][0] == 0).all())
            sequence = LasHeRSequence(temporary, 'toy')
            with patch('cv2.imread', wraps=cv2.imread) as read:
                image = sequence.image(0, (True, False))
                self.assertEqual(read.call_count, 1)
                self.assertTrue(torch.isnan(image[3:]).all())
            with patch('cv2.imread', wraps=cv2.imread) as read:
                self.assertIsNone(sequence.image(0, (False, False)))
                self.assertEqual(read.call_count, 0)

    def test_crop_geometry_preserves_pixel_centers(self):
        field = torch.arange(48).reshape(1, 6, 8).float().repeat(6, 1, 1)
        actual = crop_image(field, torch.tensor([0., 0., 1., 1.]), (6, 8))
        torch.testing.assert_close(actual, field, atol=1e-5, rtol=0)

    def test_reference_precision_is_explicit_and_independent_of_outer_autocast(self):
        model = small_model(temporal=True).eval()
        image = torch.randn(1, 3, 64, 64)
        crop = torch.tensor([[0., 0., 1., 1.]])
        expected = model.observe_reference(image, 'rgb', [0], crop)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = model.observe_reference(image, 'rgb', [0], crop)
        torch.testing.assert_close(expected.tokens, actual.tokens, rtol=0, atol=0)
        different = small_model()
        different.tracker.load_state_dict(model.tracker.state_dict(), strict=True)
        different.reference_precision = 'bfloat16'
        different.activate_temporal('synthetic-unit-test-adaptation')
        self.assertNotEqual(model.reference.space_id, different.reference.space_id)

    def test_invalid_future_annotations_are_masked_without_losing_current_tracking(self):
        model = small_model(temporal=True).eval()
        inputs, targets = clip(1)
        targets['future_boxes'][:, 0] = float('nan')
        loss, metrics = MissingTrackingLoss()(model, forward_clip(model, inputs), targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics['future_valid_motion_targets'], 2)

    def test_autocast_warmup_preserves_predictor_and_reconstructor_parameter_gradients(self):
        model = small_model(temporal=True).train()
        inputs, targets = clip(1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = forward_clip(model, inputs, 5)
            loss, _ = MissingTrackingLoss()(model, output, targets)
        loss.backward()
        for parameter in (model.future_predictor.queries, model.future_predictor.condition.project.weight,
                          model.reconstructor.initial.weight, model.fusion.output.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_batched_temporal_conditions_do_not_mix_tracks(self):
        model = small_model(temporal=True).eval()
        inputs, _ = clip(2)
        # Different current appearances and initialization ROIs produce different histories.
        inputs['context'][1, :, :3] *= 2
        inputs['initial_box'][1] = torch.tensor([.1, .1, .4, .4])
        together = forward_clip(model, inputs)
        for index in range(2):
            single = forward_clip(model, {key: value[index:index + 1] for key, value in inputs.items()})
            torch.testing.assert_close(together['boxes'][index:index + 1], single['boxes'], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(together['samples'][index]['future'].appearance,
                                       single['samples'][0]['future'].appearance, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(together['samples'][index]['reconstruction'][5],
                                       single['samples'][0]['reconstruction'][5], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
