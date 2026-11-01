import pickle
import unittest

import torch

from trackit.data.protocol.train_input import TrainData
from trackit.data.methods.siamese_tracker_train.worker import SiameseTrackerTrainingDataCollator
from trackit.data.methods.siamese_tracker_train.transform.default.plugin.builder import PluginRegistrationHelper
from trackit.miscellanies.torch.tensor_move_helper import move_torch_tensors_in_object_to_device


class TrainingDataIsolationTests(unittest.TestCase):
    def test_queued_batches_keep_their_own_inputs_and_targets(self):
        def collect(batch, output):
            output.input['x'] = batch[0]
            output.target['box'] = batch[0] + 1
        collator = SiameseTrackerTrainingDataCollator(collect)
        first = collator(([torch.tensor([1.])], None))
        second = collator(([torch.tensor([2.])], None))
        self.assertIsNot(first.input, second.input)
        self.assertIsNot(first.target, second.target)
        self.assertIsNot(first.miscellanies, second.miscellanies)
        self.assertEqual(first.input['x'].item(), 1.)
        self.assertEqual(first.target['box'].item(), 2.)
        restored = pickle.loads(pickle.dumps(first))
        moved = move_torch_tensors_in_object_to_device(restored, torch.device('cpu'))
        self.assertIsInstance(moved, TrainData)
        self.assertEqual(moved.input['x'].item(), 1.)
        self.assertIsNone(TrainData(input=None).input)

    def test_plugin_registrations_do_not_accumulate_across_builds(self):
        first, second = PluginRegistrationHelper(), PluginRegistrationHelper()
        first.register(transform=object(), data_collator=object(), host_data_pipeline=object())
        self.assertEqual(len(first.extra_transforms), 1)
        self.assertEqual(second.extra_transforms, [])
        self.assertEqual(second.extra_data_collators, [])
        self.assertEqual(second.extra_host_data_pipelines, [])

    def test_5090_overlay_keeps_budget_and_enables_fused_optimizer(self):
        from types import SimpleNamespace
        from trackit.core.boot.funcs.main.load_config import load_config
        config = load_config(SimpleNamespace(config_path='config', method_name='DINOv3RGB',
                             config_name='dinov3', mixin_config=['fast_transfer', 'rtx5090'],
                             weight_path=None, eval=False))
        run = config['run']
        self.assertEqual(run['num_epochs'], 20)
        self.assertEqual(run['data']['train']['sampler']['samples_per_epoch'], 131072)
        self.assertEqual(run['data']['train']['global_batch_size'], 64)
        self.assertEqual(run['data']['train']['num_workers'], 20)
        self.assertTrue(run['runner']['train']['optimization']['optimizer']['fused'])
        self.assertEqual(run['runner']['train']['optimization']['grad_accumulation_steps'], 1)
        self.assertEqual(run['performance']['opencv_num_threads'], 0)


if __name__ == '__main__':
    unittest.main()
