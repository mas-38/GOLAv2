"""CPU-only integration checks through the real application checkpoint path."""
import contextlib
from copy import deepcopy
import io
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from main import setup_arg_parser
from trackit.criteria import CriterionOutput
from trackit.data import DataPipeline
from trackit.data.protocol.train_input import TrainData
from trackit.models import ModelBuildingContext
from trackit.core.runtime.application.default.builder import build_default_application
from trackit.core.runtime.metric_logger.interface import MetricLoggerInterface
from trackit.runner.training.common.optimization.amp.parameter_updater import ParameterUpdater_WithAMPSupport


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Dropout(.25), nn.Linear(8, 3))

    def forward(self, x):
        return self.layers(x)


class TrainableCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(.2))

    def forward(self, output, target):
        return CriterionOutput((output - target).square().mean() * self.log_scale.exp() + .01 * self.log_scale.square())


def make_app(output, per_iteration=True, resume=None, weights=None, wd_schedule=False):
    optimization = dict(optimizer=dict(type='AdamW', lr=.002, weight_decay=.05),
                        grad_accumulation_steps=1, max_grad_norm=1., zero_grad_set_to_none=True,
                        lr_scheduler=dict(type='timm', sched='cosine', per_iteration=per_iteration,
                                          parameters=dict(lr_min=.00001, warmup_epochs=1, warmup_lr=.0001)),
                        auto_mixed_precision=dict(enabled=False))
    if wd_schedule:
        optimization['weight_decay_scheduler'] = dict(type='timm', sched='cosine', per_iteration=per_iteration,
                                                     parameters=dict(wd_end=.001))
    config = dict(name='resume-integration', run=dict(num_epochs=4, efficiency_assessment=dict(enabled=False),
                  checkpoint=[dict(type='regular', resumable=True, epoch_trigger=dict(interval=1))],
                  task=dict(train=dict(is_train=True, data='train', runner='train', logging=dict(interval=100))),
                  data=dict(train={}), runner=dict(train=dict(type='default_train', optimization=optimization,
                                                            torch_compile=dict(enabled=False),
                                                            detect_unused_parameters=False, criteria={}))))
    args = setup_arg_parser().parse_args(['test', 'test', '--device', 'cpu', '--disable_wandb', '--output_dir', str(output)])
    args.run_id = 'resume-integration'
    args.resume, args.weight_path = resume, [str(weights)] if weights else None

    def data_pipeline(data_name, data_config, context, model_config):
        context.variables.update(batch_size=4, num_iterations_per_epoch=3)
        rng = torch.Generator().manual_seed(915)
        batches = [TrainData(dict(x=torch.randn(4, 5, generator=rng)), torch.randn(4, 3, generator=rng)) for _ in range(3)]
        return DataPipeline(batches, None)

    def cpu_amp(options, max_grad_norm, device):
        assert device.type == 'cpu'
        scaler = torch.amp.GradScaler('cpu', init_scale=16., growth_interval=2)
        return ParameterUpdater_WithAMPSupport(scaler, max_grad_norm), contextlib.nullcontext

    with patch('trackit.core.runtime.application.default.builder.get_model_build_context',
               return_value=ModelBuildingContext(lambda _: SmallModel(), lambda _: 'small-model')), \
         patch('trackit.core.runtime.application.default.builder.build_data_pipeline', side_effect=data_pipeline), \
         patch('trackit.runner.training.default.builder.build_criterion', return_value=TrainableCriterion()), \
         patch('trackit.runner.training.common.optimization.builder.build_auto_mixed_precision_optimization_components',
               side_effect=cpu_amp), contextlib.redirect_stdout(io.StringIO()):
        return build_default_application(config, args, np.random.default_rng(1), np.random.default_rng(2),
                                         np.random.default_rng(3), None)


class Trajectory(MetricLoggerInterface):
    def __init__(self, app):
        self.app = app
        self.rows = []

    def log(self, metrics, force, step):
        trainer = self.app._all_context.runners['train'].runner
        self.rows.append(dict(metrics=dict(metrics), model=deepcopy(self.app._model_manager.state_dict()),
                              trainer=deepcopy(trainer.get_state())))

    def log_summary(self, metrics):
        pass


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_training_builder_registers_optimizer_scheduler_and_amp_state(self):
        with TemporaryDirectory() as directory:
            app = make_app(Path(directory)/'run')
            saved = app._all_context.state_dict()
            self.assertIn('/runner/train/trainer', saved['objects'])
            trainer_state, _ = saved['objects']['/runner/train/trainer']
            for key in ['optimizer', 'iteration', 'lr_scheduler_per_iteration', 'amp_param_updater', 'criterion']:
                self.assertIn(key, trainer_state)
            self.assertEqual(trainer_state['amp_param_updater']['scale'], 16.)

    def test_saved_application_resumes_same_next_and_subsequent_updates(self):
        for per_iteration, wd_schedule in product((True, False), repeat=2):
            with self.subTest(per_iteration=per_iteration, wd_schedule=wd_schedule), TemporaryDirectory() as directory:
                root = Path(directory)
                torch.manual_seed(42)
                baseline = make_app(root/'baseline', per_iteration, wd_schedule=wd_schedule)
                expected = Trajectory(baseline)
                baseline._all_context.tasks['train'].metric_logger.register_logger('trajectory', expected)
                with contextlib.redirect_stdout(io.StringIO()):
                    baseline.run()
                checkpoint = root/'baseline/checkpoint/epoch_1'
                state = torch.load(checkpoint/'state.pth', map_location='cpu', weights_only=True)
                self.assertIn('/runner/train/trainer', state['objects'])
                training_state, _ = state['objects']['/runner/train/trainer']
                self.assertTrue(training_state['optimizer']['state'])
                for moments in training_state['optimizer']['state'].values():
                    self.assertIn('exp_avg', moments)
                    self.assertIn('exp_avg_sq', moments)
                    self.assertEqual(moments['step'].item(), 6)
                self.assertEqual(training_state['iteration'], 6)
                self.assertGreater(training_state['amp_param_updater']['scale'], 16.)
                scheduler_key = 'lr_scheduler_per_iteration' if per_iteration else 'lr_scheduler'
                self.assertIn(scheduler_key, training_state)
                if wd_schedule:
                    self.assertIn('weight_decay_scheduler_per_iteration' if per_iteration else 'weight_decay_scheduler', training_state)
                restored = make_app(root/'resumed', per_iteration, str(checkpoint/'state.pth'), checkpoint/'model.bin', wd_schedule)
                actual = Trajectory(restored)
                restored._all_context.tasks['train'].metric_logger.register_logger('trajectory', actual)
                from trackit.core.runtime.utils.checkpoint.load import load_application_state

                def check_restored_state(setter, path):
                    load_application_state(setter, path)
                    self.assert_nested_equal(training_state, restored._all_context.runners['train'].runner.get_state())

                with patch('trackit.core.runtime.application.default.load_application_state', side_effect=check_restored_state), \
                        contextlib.redirect_stdout(io.StringIO()):
                    restored.run()
                self.assertEqual(len(actual.rows), 6)
                # Every resumed update, including its loss, LR, Adam moments and
                # enabled AMP scale/growth tracker, must match continuous training.
                for expected_row, actual_row in zip(expected.rows[6:], actual.rows):
                    self.assert_nested_equal(expected_row, actual_row)
                self.assertEqual(restored._all_context.epoch.get_current(), 4)
                self.assertEqual(restored._all_context.iteration.get_iteration(), 48)
                self.assert_nested_equal(baseline._all_context.runners['train'].runner.get_state(),
                                         restored._all_context.runners['train'].runner.get_state())

    def test_legacy_checkpoint_missing_trainer_is_rejected(self):
        with TemporaryDirectory() as directory:
            app = make_app(Path(directory)/'run')
            state = app._all_context.state_dict()
            state['objects'].pop('/runner/train/trainer', None)
            with self.assertRaisesRegex(RuntimeError, '/runner/train/trainer'):
                app._all_context.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
