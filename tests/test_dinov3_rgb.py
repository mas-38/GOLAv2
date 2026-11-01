import contextlib
from copy import deepcopy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from trackit.models.backbone.dinov3.builder import import_official_dinov3
from trackit.models.methods.dinov3_rgb.backbone import TrackingCrop
from trackit.models.methods.dinov3_rgb.lora import TrackingBlock
from trackit.models.methods.dinov3_rgb.model import DINOv3RGBTracker
from trackit.models.methods.dinov3_rgb.transfer import rgbt_initialization_state_dict
from trackit.core.boot.funcs.main.load_config import load_config
from trackit.runner.training.common.optimization.optimizer.per_parameter_options.apply import parse_optimizer_per_params_config


def config(mixin=None):
    return load_config(SimpleNamespace(config_path='config', method_name='DINOv3RGB', config_name='dinov3',
                                       mixin_config=mixin, weight_path=None, eval=False))


def model():
    module = import_official_dinov3('../dinov3')
    vit = module.DinoVisionTransformer(img_size=64, patch_size=16, embed_dim=32, depth=4,
                                     num_heads=4, n_storage_tokens=4, mask_k_bias=True,
                                     layerscale_init=1e-5, pos_embed_rope_dtype='fp32')
    vit.init_weights()
    return DINOv3RGBTracker(vit, (2, 2), (4, 4), independent_depth=2, rank=2, alpha=4, dropout=0)


def inputs(batch=2):
    return dict(z=torch.randn(batch, 3, 32, 32), d=torch.randn(batch, 3, 32, 32),
                x=torch.randn(batch, 3, 64, 64),
                z_feat_mask=torch.ones(batch, 2, 2, dtype=torch.long),
                d_feat_mask=torch.zeros(batch, 2, 2, dtype=torch.long))


class DINOv3RGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_lora_preserves_official_block_and_k_bias(self):
        net = model().eval()
        official = deepcopy(net.backbone.blocks[0].base)
        block = TrackingBlock(deepcopy(official), 'rgb', 2, 4, 0).eval()
        x = torch.randn(2, 9, 32)
        rope = net.backbone.rope_embed(H=2, W=2)
        torch.testing.assert_close(block(x, rope, 'rgb'), official(x, rope))
        normalized = official.norm1(x)
        # Even with nonzero Q/V updates, the K projection stays identical.
        with torch.no_grad():
            block.adapters['rgb'].q.B.normal_()
            block.adapters['rgb'].v.B.normal_()
        delta = block.adapters['rgb'](normalized)
        self.assertEqual(torch.count_nonzero(delta[..., 32:64]).item(), 0)
        self.assertTrue(torch.equal(block.base.attn.qkv.bias_mask, official.attn.qkv.bias_mask))

    def test_crop_prefixes_and_rope_remain_aligned(self):
        net = model().eval()
        data = inputs()
        attention = net.backbone.blocks[2].base.attn
        with patch.object(attention, 'compute_attention', wraps=attention.compute_attention) as call:
            features = net.forward_features(**data)
        self.assertEqual({k: v.shape for k, v in features.items()},
                         {'z': (2, 4, 32), 'd': (2, 4, 32), 'x': (2, 16, 32)})
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs['qkv'].shape[1], 3 * 5 + 4 + 4 + 16)
        self.assertEqual(kwargs['rope'][0].shape[-2], 24)
        output = net(**data)
        self.assertEqual(output['score_map'].shape, (2, 4, 4))
        self.assertEqual(output['boxes'].shape, (2, 4, 4, 4))
        with self.assertRaises(ValueError):
            net(**{**data, 'x': torch.randn(2, 6, 64, 64)})

    def test_real_tracking_loss_updates_adapters_and_head_only(self):
        from trackit.criteria.methods.box_with_score_map.builder import build_box_with_score_map_criteria
        net = model().train()
        frozen_before = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
        with contextlib.redirect_stdout(io.StringIO()):
            groups = parse_optimizer_per_params_config(net, None,
                        config()['run']['runner']['train']['optimization']['optimizer'])
            criterion = build_box_with_score_map_criteria(config()['run']['runner']['train']['criteria'])
        optimized = {id(p) for group in groups for p in group['params']}
        self.assertEqual(optimized, {id(p) for p in net.parameters() if p.requires_grad})
        optimizer = torch.optim.AdamW(groups, lr=1e-3)
        data = inputs()
        before = net.backbone.blocks[0].adapters['rgb'].q.B.detach().clone()
        for _ in range(2):
            target = dict(num_positive_samples=torch.tensor([2.]),
                          positive_sample_batch_dim_indices=torch.tensor([0, 1]),
                          positive_sample_map_dim_indices=torch.tensor([5, 10]),
                          boxes=torch.tensor([[.2, .2, .7, .7], [.3, .3, .8, .8]]))
            loss = criterion(net(**data), target).loss
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            for name, parameter in net.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
        self.assertFalse(torch.equal(before, net.backbone.blocks[0].adapters['rgb'].q.B))
        for name, parameter in net.named_parameters():
            if name in frozen_before:
                self.assertTrue(torch.equal(parameter, frozen_before[name]), name)

    def test_full_checkpoint_and_rgbt_initialization_round_trip(self):
        net = model().eval()
        # Nonzero adapters verify that learned updates survive export, not just initialization.
        with torch.no_grad():
            for block in net.backbone.blocks:
                next(iter(block.adapters.values())).q.B.normal_(std=.01)
        data = inputs()
        expected = net(**data)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.bin')
            save_file(net.state_dict(), path)
            state = load_file(path)
        restored = model().eval()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(**data)['boxes'], expected['boxes'])
        state_rgbt = rgbt_initialization_state_dict(state)
        restored.backbone.add_modality('tir')
        restored.load_state_dict(state_rgbt)
        torch.testing.assert_close(restored(**data)['boxes'], expected['boxes'])
        self.assertIsNot(restored.backbone.blocks[0].adapters['rgb'].q.B,
                         restored.backbone.blocks[0].adapters['tir'].q.B)
        features = restored.backbone.encode_crops({
            'z_rgb': TrackingCrop(data['z'], 'z', 'rgb', data['z_feat_mask']),
            'z_tir': TrackingCrop(data['z'], 'z', 'tir', data['z_feat_mask']),
            'x_rgb': TrackingCrop(data['x'], 'x', 'rgb'),
            # No current TIR crop is passed or embedded.
        })
        self.assertEqual(set(features), {'z_rgb', 'z_tir', 'x_rgb'})
        self.assertTrue(torch.isfinite(features['x_rgb']).all())
        incompatible = model()
        incompatible.architecture[6] += 1  # alpha mismatch, despite identical tensor shapes
        with self.assertRaises(ValueError):
            incompatible.load_state_dict(state)

    def test_rgb_online_template_is_decoded_and_labels_are_collated(self):
        from trackit.data.methods.siamese_tracker_train._types import SOTFrameInfo, SiameseTrainingPair
        from trackit.data.methods.siamese_tracker_train.transform.default.builder import _build_siamfc_cropping_parameter
        from trackit.data.methods.siamese_tracker_train.transform.default.processor import SiamTrackerTrainingPairProcessor, SiamTrackerTrainingPairProcessorBatchCollator
        from trackit.data.methods.siamese_tracker_train.transform.default.augmentation.builder import build_augmentation_pipeline
        from trackit.data.methods.siamese_tracker_train.transform.default.plugin.template_foreground_indicating_mask_gen import TemplateFeatMaskGenerator, template_feat_mask_data_collator
        from trackit.data.methods.siamese_tracker_train.transform.default.plugin.box_with_score_map_label_gen import BoxWithScoreMapLabelGenerator, box_with_score_map_label_collator
        from trackit.data.protocol.train_input import TrainData
        template = _build_siamfc_cropping_parameter({'area_factor': 2}, (32, 32), 'bilinear', False)
        search = _build_siamfc_cropping_parameter({'area_factor': 4}, (64, 64), 'bilinear', False)
        processor = SiamTrackerTrainingPairProcessor(template, search, build_augmentation_pipeline([]),
                    (.5, 1), (.5, 1), 'imagenet',
                    (TemplateFeatMaskGenerator((32, 32), (2, 2)),
                     BoxWithScoreMapLabelGenerator((4, 4), (64, 64))))
        decoded = []
        def image():
            decoded.append(True)
            return np.full((96, 96, 3), 128, dtype=np.uint8)
        frame = SOTFrameInfo(image, np.array([32., 32., 64., 64.]), True, None, None, None)
        pair = SiameseTrainingPair(True, True, frame, frame, frame)
        sample = processor(pair, np.random.default_rng(42))
        self.assertEqual(len(decoded), 1)  # All three roles use the decode cache.
        self.assertEqual(sample['d_cropped_image'].shape, (3, 32, 32))
        collated = TrainData()
        SiamTrackerTrainingPairProcessorBatchCollator((template_feat_mask_data_collator,
                    box_with_score_map_label_collator))([sample], collated)
        self.assertEqual(collated.input['d'].shape, (1, 3, 32, 32))
        self.assertEqual(collated.input['d_feat_mask'].shape, (1, 2, 2))
        self.assertGreater(collated.target['num_positive_samples'].item(), 0)

    def test_configuration_routes_rgb_data_and_optional_validation(self):
        cfg = config()
        self.assertEqual(cfg['common']['normalization'], 'imagenet')
        dataset = cfg['run']['data']['train']['source']['parameters']['datasets'][0]
        self.assertEqual((dataset['name'], dataset['type'], dataset['splits']), ('GOT10k', 'SOT', ['train']))
        self.assertEqual(set(cfg['run']['task']), {'train'})
        self.assertEqual(config(['smoke'])['run']['data']['train']['sampler']['samples_per_epoch'], 8)
        evaluation = config(['evaluation'])
        self.assertEqual(set(evaluation['run']['task']), {'eval'})
        self.assertEqual(evaluation['run']['data']['eval']['source']['parameters']['datasets'][0]['splits'], ['val'])

    def test_rng_checkpoint_is_safe_to_load_and_restores_randomness(self):
        import random
        from trackit.core.runtime.application.default.local_context import _get_rng_state, _load_rng_state
        state = _get_rng_state()
        expected = (np.random.random(), random.random(), torch.rand(3))
        file = io.BytesIO()
        torch.save(state, file)
        file.seek(0)
        _load_rng_state(torch.load(file, weights_only=True))
        self.assertEqual(np.random.random(), expected[0])
        self.assertEqual(random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

    def test_application_checkpoint_restores_unwrapped_component_state(self):
        from trackit.core.runtime.services import ServicesRegistry
        from trackit.core.runtime.services.checkpoint import get_state_from_registries, load_state_to_registries
        registry = ServicesRegistry().checkpoint
        component = {'iteration': 7}
        registry.register('component', lambda: dict(component), lambda state: component.update(state))
        saved = get_state_from_registries([('/train/', registry)])
        component['iteration'] = 99
        load_state_to_registries(saved, [('/train/', registry)])
        self.assertEqual(component['iteration'], 7)

    def test_fast_transfer_preserves_training_and_adds_validation(self):
        from trackit.core.runtime.utils.epoch_activation_criteria.builder import build_epoch_activation_criteria
        cfg = config(['fast_transfer'])
        self.assertEqual(cfg['model']['tracking']['rank'], 32)
        self.assertEqual(cfg['model']['tracking']['alpha'], 64)
        run = cfg['run']
        self.assertEqual(run['num_epochs'], 20)
        self.assertEqual(list(run['task']), ['train', 'eval'])
        self.assertEqual(set(run['data']), {'train', 'eval'})
        self.assertEqual(set(run['runner']), {'train', 'test'})
        self.assertEqual(run['runner']['train']['optimization']['grad_accumulation_steps'], 1)
        trigger = build_epoch_activation_criteria(run['task']['eval']['epoch_trigger'], 20)
        self.assertEqual([e + 1 for e in range(20) if trigger(e)], [2, 5, 10, 15, 20])
        self.assertEqual(run['checkpoint'][1]['metric'], 'val/average_overlap_GOT-10k-val')
        self.assertEqual(run['checkpoint'][0]['max_to_keep'], 20)

    def test_application_saves_best_weights_after_training_temporary_dump(self):
        from trackit.core.runtime.application.default import DefaultApplication
        from trackit.core.runtime.application.default.local_context import ApplicationContext, EpochIterator, GlobalIterationCounter
        from trackit.core.runtime.metric_logger.epoch_metric import get_current_epoch_metrics, disable_epoch_metrics
        from trackit.core.runtime.services import ServicesRegistry
        from trackit.core.runtime.utils.checkpoint.save.builder import build_checkpoint_dumper
        from unittest.mock import Mock

        # Exercise the application's train -> temporary save -> validation -> best
        # save ordering. Toy tasks perform no optimization or dataset iteration.
        weight = torch.zeros(1)
        manager = SimpleNamespace(version=0, state_dict=lambda: {'weight': weight.clone()})
        tasks = {name: SimpleNamespace(name=name, is_train=is_train, data_name=name,
                                      runner_name=name, epoch_activation_criteria=lambda e: True,
                                      services_registry=ServicesRegistry())
                 for name, is_train in (('train', True), ('eval', False))}
        data = {name: SimpleNamespace(name=name, services_registry=ServicesRegistry()) for name in tasks}
        runners = {name: SimpleNamespace(name=name, services_registry=ServicesRegistry()) for name in tasks}
        contexts = ApplicationContext(data, runners, tasks, EpochIterator(3), GlobalIterationCounter())
        metric = 'val/average_overlap_GOT-10k-val'

        def run_toy_task(model_manager, task, data_context, runner_context, epoch, iteration):
            if task.is_train:
                weight.fill_(epoch + 1)
                manager.version += 1
            else:
                get_current_epoch_metrics().update(epoch, {metric: (0.4, 0.6, 0.5)[epoch]})

        with TemporaryDirectory() as directory:
            dumper = build_checkpoint_dumper(config(['fast_transfer'])['run']['checkpoint'], directory, 3)
            application = DefaultApplication('toy', manager, None, Mock(), contexts, dumper)
            try:
                with patch('trackit.core.runtime.application.default.run_task', side_effect=run_toy_task), \
                        contextlib.redirect_stdout(io.StringIO()):
                    application.run()
                best = list(Path(directory).glob('best@*/model.bin'))
                self.assertEqual(len(best), 1)
                torch.testing.assert_close(load_file(str(best[0]))['weight'], torch.tensor([2.]))
                self.assertEqual(len(list(Path(directory).glob('epoch_*/model.bin'))), 3)
                self.assertEqual(torch.load(str(Path(directory) / 'epoch_2/state.pth'), weights_only=True)['epoch'], 3)
            finally:
                disable_epoch_metrics()


if __name__ == '__main__':
    unittest.main()
