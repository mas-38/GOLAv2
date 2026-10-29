import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from trackit.models.backbone.dinov2 import DinoVisionTransformer
from trackit.models.methods.dinov2_missing.model import DINOv2MissingTracker
from trackit.data.methods.siamese_tracker_eval.transform.missing import MissingModalitySchedule, decode_available_modalities
from trackit.data.methods.siamese_tracker_eval.transform.default import SiameseTrackerEval_DefaultDataTransform
from trackit.data.methods.siamese_tracker_eval import SiameseTrackerEvalDataWorker_Task, SiameseTrackerEvalDataWorker_FrameContext
from trackit.data.protocol import SequenceInfo
from trackit.data.protocol.eval_input import TrackerEvalData, TrackerEvalData_TaskDesc, TrackerEvalData_FrameData
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines import TrackerEvaluationPipeline_Context
from trackit.runner.evaluation.distributed.tracker_evaluator.default.pipelines.one_stream.builder import build_one_stream_tracker_pipeline
from trackit.runner.training.common.optimization.optimizer.per_parameter_options.apply import parse_optimizer_per_params_config
from trackit.core.boot.funcs.main.load_config import load_config


def config():
    args = SimpleNamespace(config_path='config', method_name='DINOv2Missing', config_name='dinov2',
                           mixin_config=None, weight_path=None, eval=False, missing_mask_file=None)
    return load_config(args)


def model():
    vit = DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=32, depth=2,
                                num_heads=4, block_chunks=0, acc='none')
    return DINOv2MissingTracker(vit, (2, 2), (4, 4))


def inputs(batch=3):
    return dict(z=torch.randn(batch, 6, 28, 28), d=torch.randn(batch, 6, 28, 28),
                x=torch.randn(batch, 6, 56, 56),
                z_feat_mask=torch.ones(batch, 2, 2, dtype=torch.long),
                d_feat_mask=torch.ones(batch, 2, 2, dtype=torch.long))


class MissingBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_missing_search_is_not_observed_and_batch_order_is_preserved(self):
        net = model().eval()
        data = inputs()
        flags = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bool)
        with torch.no_grad():
            expected = net(**data, modality_available=flags)
            changed = {k: v.clone() for k, v in data.items()}
            changed['x'][0, 3:] = float('nan')
            changed['x'][1, :3] = float('nan')
            actual = net(**changed, modality_available=flags)
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
                singles = torch.cat([net(**{k: v[i:i+1] for k,v in data.items()},
                                        modality_available=flags[i:i+1])[key] for i in range(3)])
                torch.testing.assert_close(actual[key], singles)
        self.assertEqual(actual['boxes'].shape, (3, 4, 4, 4))
        self.assertFalse(any('lora' in name.lower() or 'gola' in name.lower() for name in net.state_dict()))
        restored = model().eval()
        restored.load_state_dict(net.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(**data, modality_available=flags)['boxes'], expected['boxes'])

    def test_all_training_states_backpropagate_into_full_backbone(self):
        net = model().train()
        with contextlib.redirect_stdout(io.StringIO()):
            groups = parse_optimizer_per_params_config(net, None, config()['run']['runner']['train']['optimization']['optimizer'])
        optimizer = torch.optim.AdamW(groups, lr=1e-4)
        optimized = {id(p) for g in groups for p in g['params']}
        self.assertEqual(optimized, {id(p) for p in net.parameters() if p.requires_grad})
        for state, flags in enumerate(((1,0), (0,1), (1,1))):
            data = inputs(2)
            data['x'].requires_grad_()
            with patch('torch.randint', return_value=torch.tensor(state)):
                output = net(**data)
            explicit = net(**data, modality_available=torch.tensor([flags,flags]))
            torch.testing.assert_close(output['boxes'], explicit['boxes'])
            optimizer.zero_grad()
            loss = output['boxes'].square().mean() + output['score_map'].square().mean()
            loss.backward()
            weight = net.blocks[0].attn.qkv.weight
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
            if state < 2:
                missing = slice(3,6) if state == 0 else slice(0,3)
                self.assertEqual(data['x'].grad[:,missing].abs().sum().item(), 0)
            before = weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, weight))
        with self.assertRaises(ValueError):
            net(**inputs(1), modality_available=torch.zeros(1,2))

    def test_configuration_and_full_checkpoint_roundtrip(self):
        from trackit.models.methods.builder import get_model_build_context
        from trackit.models import ModelImplSuggestions
        import safetensors.torch
        cfg = config()
        self.assertNotIn('lora', cfg['model'])
        self.assertNotIn('gola_orthogonal_loss_weight', cfg['run']['runner']['train'])
        self.assertFalse(cfg['run']['runner']['train']['torch_compile']['enabled'])
        with patch('trackit.models.methods.dinov2_missing.builder.build_backbone',
                   side_effect=lambda *args, **kwargs: DinoVisionTransformer(img_size=56, patch_size=14,
                       embed_dim=32, depth=2, num_heads=4, block_chunks=0, acc='none')):
            cfg['common']['template_feat_size'] = [2,2]
            cfg['common']['search_region_feat_size'] = [4,4]
            context = get_model_build_context(cfg)
            first = context.create_fn(ModelImplSuggestions())
            second = context.create_fn(ModelImplSuggestions(optimize_for_inference=True))
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'model.bin')
            safetensors.torch.save_model(first, path)
            safetensors.torch.load_model(second, path, strict=True)
            self.assertIn('blocks.0.attn.qkv.weight', safetensors.torch.load_file(path))
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[key])
            args = SimpleNamespace(config_path='config', method_name='DINOv2Missing', config_name='dinov2',
                                   mixin_config=['evaluation'], weight_path=[path], eval=True,
                                   missing_mask_file=str(Path(tmp)/'mask.json'))
            evaluation = load_config(args)
            self.assertEqual(set(evaluation['run']['task']), {'eval'})
            self.assertTrue(Path(evaluation['run']['data']['eval']['transform']['missing_mask_file']).is_absolute())

    def test_schedule_and_decoder_fail_closed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'masks.json'
            path.write_text(json.dumps({'seq': {'data': [[1,1], [1,0], [0,1], [0,0]]}}))
            schedule = MissingModalitySchedule(path)
            self.assertEqual(schedule.get('seq',2),(False,True))
            with self.assertRaises(KeyError): schedule.get('absent',0)
            with self.assertRaises(IndexError): schedule.get('seq',4)
            path.write_text(json.dumps({'seq': {'data': [[0.5,1]]}}))
            with self.assertRaises(ValueError): MissingModalitySchedule(path)
        def missing(): raise AssertionError('Missing image was read')
        good = lambda: np.ones((8,8,3), dtype=np.uint8)
        self.assertEqual(decode_available_modalities([good,missing],(True,False)).shape,(8,8,6))
        self.assertIsNone(decode_available_modalities([missing,missing],(False,False)))

    def test_eval_mixed_missing_frames_template_images_and_masks(self):
        cfg = config()
        cfg['common'].update(template_size=[28,28], search_region_size=[56,56],
                             template_feat_size=[2,2], search_region_feat_size=[4,4], response_map_size=[4,4])
        pipe_cfg = cfg['run']['runner']['test']['evaluator']['pipeline']
        with contextlib.redirect_stdout(io.StringIO()):
            pipelines = build_one_stream_tracker_pipeline(pipe_cfg,cfg,torch.device('cpu'))
        shared = {}
        for pipeline in pipelines: pipeline.start(4,shared)
        main, initial_masks, online_masks = pipelines
        bbox = np.array([16.,16.,32.,32.])
        flags = ((False,False),(True,False),(False,True),(True,True))
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'schedule.json'
            path.write_text(json.dumps({str(i): {'data': [[0,0],list(flags[i]),[0,0],[1,1]]} for i in range(4)}))
            transform = SiameseTrackerEval_DefaultDataTransform((28,28),2.,False,'bilinear',False,'mm',missing_mask_file=path)
            def getter(value): return lambda: np.full((64,64,3),value,dtype=np.uint8)
            def missing(): raise AssertionError('Missing image decoded')
            init_tasks=[]
            for i in range(4):
                info=SequenceInfo('test',None,'test',str(i),4,None)
                init=SiameseTrackerEvalDataWorker_FrameContext(0,[getter(30),getter(90)],bbox)
                init_tasks.append(transform(SiameseTrackerEvalDataWorker_Task(i,info,init,None,False,str(i))))
            context=TrackerEvaluationPipeline_Context(TrackerEvalData(init_tasks,{}),{},shared)
            for pipeline in pipelines: pipeline.begin(context)
            for pipeline in pipelines: pipeline.prepare_initialization(context,{})
            old_images=[main.template_updater.get(i).clone() for i in range(4)]
            # Different valid historical masks make accidental updates observable.
            for i in range(4): online_masks.template_mask_cache.update(i,torch.zeros(2,2,2,dtype=torch.long))
            old_masks=[online_masks.template_mask_cache.get(i).clone() for i in range(4)]
            tasks=[]
            for i,available in enumerate(flags):
                getters=[getter(120) if available[0] else missing,getter(180) if available[1] else missing]
                frame=SiameseTrackerEvalDataWorker_FrameContext(1,getters,bbox)
                tasks.append(transform(SiameseTrackerEvalDataWorker_Task(i,None,None,frame,False,str(i))))
            context=TrackerEvaluationPipeline_Context(TrackerEvalData(tasks,{}),{},shared)
            params={}
            for pipeline in pipelines: pipeline.prepare_tracking(context,params)
            self.assertEqual(params['x'].shape[0],3)
            self.assertEqual(params['d_feat_mask'].shape,(3,2,2,2))
            self.assertEqual(context.temporary_objects['task_ids'],[1,2,3])
            with torch.no_grad(): outputs=model().eval()(**params)
            outputs['score_map'].fill_(10.)
            outputs['boxes'][...]=torch.tensor([.25,.25,.75,.75])
            for pipeline in pipelines: pipeline.on_tracked(outputs,context)
            self.assertTrue(context.result.is_all_submitted())
            np.testing.assert_array_equal(context.result.get(0).box,bbox)
            for pipeline in pipelines: pipeline.do_custom_update(None,None,context)
            for i,available in enumerate(flags):
                image=main.template_updater.get(i)
                masks=online_masks.template_mask_cache.get(i)
                for modality,valid in enumerate(available):
                    channels=slice(modality*3,(modality+1)*3)
                    self.assertEqual(torch.equal(image[channels],old_images[i][channels]),not valid)
                    self.assertEqual(torch.equal(masks[modality],old_masks[i][modality]),not valid)
            # Entire batch missing: no decoder/model input and no state mutation.
            snapshots=[main.template_updater.get(i).clone() for i in range(4)]
            tasks=[]
            for i in range(4):
                frame=SiameseTrackerEvalDataWorker_FrameContext(2,[missing,missing],bbox)
                tasks.append(transform(SiameseTrackerEvalDataWorker_Task(i,None,None,frame,False,str(i))))
            context2=TrackerEvaluationPipeline_Context(TrackerEvalData(tasks,{}),{},shared)
            params={}
            for pipeline in pipelines: pipeline.prepare_tracking(context2,params)
            self.assertEqual(params,{})
            for pipeline in pipelines: pipeline.on_tracked(None,context2)
            for pipeline in pipelines: pipeline.do_custom_update(None,None,context2)
            for i in range(4):
                torch.testing.assert_close(main.template_updater.get(i),snapshots[i])
                np.testing.assert_array_equal(context2.result.get(i).box,context.result.get(i).box)
            # Recovery updates both modalities; finalization clears all caches.
            tasks=[]
            for i in range(4):
                frame=SiameseTrackerEvalDataWorker_FrameContext(3,[getter(220),getter(240)],bbox)
                tasks.append(transform(SiameseTrackerEvalDataWorker_Task(i,None,None,frame,True,str(i))))
            final=TrackerEvaluationPipeline_Context(TrackerEvalData(tasks,{}),{},shared)
            params={}
            for pipeline in pipelines: pipeline.prepare_tracking(final,params)
            self.assertTrue(params['modality_available'].all())
            outputs={'score_map':torch.full((4,4,4),10.),'boxes':torch.tensor([.25,.25,.75,.75]).expand(4,4,4,4)}
            for pipeline in pipelines: pipeline.on_tracked(outputs,final)
            for pipeline in pipelines: pipeline.do_custom_update(None,None,final)
            for i in range(4):
                for modality in range(2):
                    self.assertFalse(torch.equal(main.template_updater.get(i)[modality*3:(modality+1)*3], snapshots[i][modality*3:(modality+1)*3]))
            for pipeline in pipelines: pipeline.end(final)
        for pipeline in pipelines: pipeline.stop(shared)


if __name__ == '__main__':
    unittest.main()
