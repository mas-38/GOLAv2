import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from analyze_dinov3_rgb_training import analyze, read_jsonl
from trackit.core.boot.funcs.main.local_run_logging import record_local_run
from trackit.core.runtime.metric_logger.adaptors.jsonl import JsonlMetricLogger


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_windows_and_epoch_means_include_every_batch_and_flush_tail(self):
        with TemporaryDirectory() as directory:
            logger = JsonlMetricLogger(directory, 'train', interval=2, prefix='train/')
            logger.on_epoch_begin(0, True)
            for step, loss in enumerate((1., 2., 100.)):
                logger.log({'loss': loss}, step=step * 16)
                # Different producers contribute to the same batch.
                logger.log({'lr': .001, 'timing/samples_in_batch': 16,
                            'timing/batch_seconds': (1., 3., 2.)[step]}, step=step * 16)
                logger.commit()
            logger.on_epoch_end(0, True)
            windows = read_jsonl(Path(directory) / 'metrics.jsonl')
            epoch = read_jsonl(Path(directory) / 'epoch_summary.jsonl')[0]
            self.assertEqual([r['batches'] for r in windows], [2, 1])
            self.assertEqual([r['metrics']['train/loss'] for r in windows], [1.5, 100.])
            self.assertAlmostEqual(epoch['metrics']['train/loss'], 103. / 3)
            self.assertEqual(epoch['batches'], 3)
            self.assertEqual(epoch['statistics']['train/loss']['max'], 100.)
            self.assertEqual(windows[-1]['sample_step'], 32)
            self.assertEqual(epoch['metrics']['train/timing/samples_per_second'], 8.)
            self.assertEqual(windows[0]['metrics']['train/timing/samples_per_second'], 8.)

    def test_nonfinite_values_are_explicit_and_json_stays_valid(self):
        with TemporaryDirectory() as directory:
            logger = JsonlMetricLogger(directory, 'train', interval=3)
            logger.on_epoch_begin(0, True)
            for value in (float('nan'), 2., float('inf')):
                logger.log({'grad_norm': value}, step=0)
                logger.commit()
            logger.on_epoch_end(0, True)
            records = read_jsonl(Path(directory) / 'metrics.jsonl')
            self.assertEqual(sum(r['kind'] == 'nonfinite' for r in records), 2)
            stats = records[-1]['statistics']['grad_norm']
            self.assertEqual(stats['nonfinite'], 2)
            self.assertEqual(stats['count'], 1)
            self.assertEqual(stats['mean'], 2.)
            self.assertNotIn('NaN', (Path(directory) / 'metrics.jsonl').read_text())

    def test_wandb_validation_history_does_not_reuse_explicit_internal_step(self):
        cloud = Mock()
        cloud.summary = {}
        with TemporaryDirectory() as directory:
            logger = JsonlMetricLogger(directory, 'eval', prefix='val/', wandb_instance=cloud)
            for epoch, ao in ((1, .4), (4, .5)):
                logger.on_epoch_begin(epoch, False)
                logger.log_summary({'average_overlap_GOT-10k-val': ao})
                logger.on_epoch_end(epoch, False)
            calls = cloud.log.call_args_list
            self.assertEqual([c.args[0]['epoch'] for c in calls], [2, 5])
            self.assertTrue(all('step' not in c.kwargs for c in calls))
            self.assertEqual(cloud.summary['val/average_overlap_GOT-10k-val'], .5)

    def test_builder_uses_one_cloud_publisher(self):
        from trackit.core.runtime.metric_logger.builder import build_metric_logger
        from trackit.core.runtime.services import ServicesRegistry
        with TemporaryDirectory() as directory:
            context = SimpleNamespace(device=torch.device('cpu'), services=ServicesRegistry(), wandb_instance=Mock())
            logger = build_metric_logger(dict(metric_prefix='train/', jsonl=dict(enabled=True, interval=50, sync_wandb=True)),
                                         context, directory, 'train')
            self.assertTrue(logger.has_logger('jsonl'))
            self.assertFalse(logger.has_logger('wandb'))
            self.assertIs(logger.get_logger('jsonl').wandb, context.wandb_instance)

    def test_run_snapshot_and_failure_are_saved_without_training(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory, run_id='diagnostics-test', device='cpu',
                                   root_path=str(Path(__file__).resolve().parents[1]))
            config = dict(run=dict(diagnostics=dict(enabled=True), num_epochs=20))
            with self.assertRaisesRegex(RuntimeError, 'diagnostic test failure'), contextlib.redirect_stdout(io.StringIO()):
                with record_local_run(args, config):
                    print('captured console output')
                    raise RuntimeError('diagnostic test failure')
            root = Path(directory) / 'diagnostics'
            info = json.loads((root / 'run_info.json').read_text())
            self.assertEqual(info['status'], 'failed')
            self.assertEqual(info['error_type'], 'RuntimeError')
            self.assertEqual(len(info['provenance']['git_commit']), 40)
            self.assertEqual(len(info['resolved_config_sha256']), 64)
            self.assertTrue((root / 'provenance.json').is_file())
            self.assertTrue((root / 'source_manifest.json').is_file())
            self.assertTrue((root / 'working_tree.patch').is_file())
            self.assertTrue((root / 'host_consts.yaml').is_file())
            self.assertEqual(json.loads((root / 'resolved_config.json').read_text())['run']['num_epochs'], 20)
            self.assertIn('captured console output', (root / 'console.log').read_text())
            self.assertIn('diagnostic test failure', (root / 'console.log').read_text())

    def test_analysis_handles_live_tail_and_selects_best_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train = JsonlMetricLogger(root / 'train', 'train', interval=1, prefix='train/')
            val = JsonlMetricLogger(root / 'eval', 'eval', prefix='val/')
            for epoch, ao in enumerate((.4, .6, .5)):
                train.on_epoch_begin(epoch, True)
                train.log({'loss': 3. - epoch, 'lr': .001}, step=epoch * 16)
                train.commit()
                train.on_epoch_end(epoch, True)
                val.on_epoch_begin(epoch, False)
                val.log_summary({'average_overlap_GOT-10k-val': ao})
                val.on_epoch_end(epoch, False)
            with (root / 'train/metrics.jsonl').open('a') as stream:
                stream.write('{"incomplete":')
            summary = analyze(root)
            self.assertEqual(summary['best_validation']['epoch'], 2)
            self.assertEqual(summary['validation_count'], 3)
            self.assertEqual(summary['completed_train_epochs'], [1, 2, 3])
            self.assertTrue((root / 'analysis/curves.png').is_file())
            self.assertTrue((root / 'analysis/train_windows.csv').is_file())

    def test_diagnostics_leave_loss_and_gradients_unchanged(self):
        from trackit.criteria.methods.box_with_score_map.builder import build_box_with_score_map_criteria
        config = dict(classification=dict(type='binary_cross_entropy', iou_aware_classification_score=True, weight=1.),
                      bbox_regression=dict(type='GIoU', weight=1.))
        with contextlib.redirect_stdout(io.StringIO()):
            plain = build_box_with_score_map_criteria(config)
            observed = build_box_with_score_map_criteria(dict(config, log_diagnostics=True))
        boxes = torch.tensor([.2, .2, .8, .8]).expand(2, 4, 4, 4).clone().requires_grad_()
        scores = torch.zeros(2, 4, 4, requires_grad=True)
        output = dict(score_map=scores, boxes=boxes)
        target = dict(num_positive_samples=torch.tensor([2.]), boxes=torch.tensor([[.2, .2, .8, .8]] * 2),
                      positive_sample_batch_dim_indices=torch.tensor([0, 1]),
                      positive_sample_map_dim_indices=torch.tensor([5, 10]))
        reference = plain(output, target)
        # The existing criterion reduces/clamps this counter in place. Give each
        # graph its own denominator, as separate real batches would receive.
        actual = observed(output, dict(target, num_positive_samples=target['num_positive_samples'].clone()))
        torch.testing.assert_close(actual.loss, reference.loss)
        torch.testing.assert_close(torch.autograd.grad(actual.loss, scores, retain_graph=True)[0],
                                   torch.autograd.grad(reference.loss, scores)[0])
        self.assertAlmostEqual(actual.extra_metrics['diagnostics/positive_iou'], 1., places=5)
        self.assertEqual(actual.extra_metrics['diagnostics/foreground_confidence'], .5)
        self.assertEqual(actual.extra_metrics['diagnostics/positive_patches'], 2.)

    def test_twenty_epoch_schedule_does_not_finish_at_epoch_ten(self):
        from main import setup_arg_parser
        from trackit.core.boot.funcs.main.load_config import load_config
        from trackit.runner.training.common.optimization.lr_scheduler.builder import build_lr_scheduler
        args = setup_arg_parser().parse_args(['DINOv3RGB', 'dinov3', '--mixin_config', 'fast_transfer'])
        args.config_path = 'config'
        config = load_config(args)
        options = config['run']['runner']['train']['optimization']
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=options['optimizer']['lr'])
        with contextlib.redirect_stdout(io.StringIO()):
            scheduler, _, _ = build_lr_scheduler(options, optimizer, options['optimizer']['lr'], 20, 8192, 1)
        self.assertEqual(scheduler.t_initial, 20 * 8192)
        self.assertGreater(scheduler._get_lr(10 * 8192)[0], 1.e-5)
        self.assertAlmostEqual(scheduler._get_lr(20 * 8192)[0], 1.e-6)


if __name__ == '__main__':
    unittest.main()
