from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from trackit.core.utils.bbox_mask_gen import get_foreground_bounding_box
from trackit.core.operator.numpy.bbox.validity import bbox_is_valid
from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params

from ....default import TrackerEvaluationPipeline, TrackerEvaluationPipeline_Context
from ....components.tensor_cache import CacheService, TensorCache


class OnlineTemplateFeatForegroundMaskGeneration(TrackerEvaluationPipeline):
    def __init__(self, template_size: Tuple[int, int], template_feat_size: Tuple[int, int],
                 template_area_factor: float, threshold: float, device: torch.device,
                 provide_during_tracking: bool = True, per_modality: bool = False):
        self.per_modality = per_modality
        self.template_size = template_size
        self.template_feat_size = template_feat_size
        self.template_area_factor = template_area_factor
        self.stride = template_size[0] / template_feat_size[0], template_size[1] / template_feat_size[1]
        self.threshold = threshold
        self.device = device
        self.background_value = 0
        self.foreground_value = 1
        self.provide_during_tracking = provide_during_tracking

    def start(self, max_batch_size: int, global_objects: dict):
        shape = (self.template_feat_size[1], self.template_feat_size[0])
        if self.per_modality:
            shape = (2, *shape)
        self.template_mask_cache = CacheService(max_batch_size, TensorCache(max_batch_size, shape, self.device, torch.long))

    def stop(self, global_objects: dict):
        del self.template_mask_cache

    def prepare_initialization(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        do_init_task_ids = []
        for task in context.input_data.tasks:
            if task.tracker_do_init_context is not None:
                current_init_context = task.tracker_do_init_context
                template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]),
                                           self.background_value, dtype=torch.long)
                template_cropped_bbox = get_foreground_bounding_box(current_init_context.gt_bbox,
                                                                    current_init_context.input_data[
                                                                        'curation_parameter'], self.stride)
                assert bbox_is_valid(template_cropped_bbox)
                template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
                template_mask[template_cropped_bbox[1]: template_cropped_bbox[3],
                template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
                if self.per_modality:
                    template_mask = template_mask.unsqueeze(0).repeat(2, 1, 1)
                self.template_mask_cache.put(task.id, template_mask.to(self.device))
                do_init_task_ids.append(task.id)
        if not self.provide_during_tracking:
            if len(do_init_task_ids) > 0:
                model_input_params['d_feat_mask'] = self.template_mask_cache.get_batch(do_init_task_ids)

    def prepare_tracking(self, context: TrackerEvaluationPipeline_Context, model_input_params: dict):
        if self.provide_during_tracking:
            do_track_task_ids = []
            active_ids = context.temporary_objects.get('task_ids')
            for task in context.input_data.tasks:
                if task.tracker_do_tracking_context is not None:
                    if active_ids is None or task.id in active_ids:
                        do_track_task_ids.append(task.id)

            if len(do_track_task_ids) > 0:
                model_input_params['d_feat_mask'] = self.template_mask_cache.get_batch(do_track_task_ids)

    def do_custom_update(self, model, raw_model: nn.Module, context: TrackerEvaluationPipeline_Context):
        for task in context.input_data.tasks:
            if task.tracker_do_tracking_context is None:
                continue
            available = task.tracker_do_tracking_context.input_data.get('modality_available', (True, True))
            if not any(available):
                continue
            if context.result.get(task.id).confidence > self.threshold:
                pred_bbox = context.result.get(task.id).box
                template_mask = torch.full((self.template_feat_size[1], self.template_feat_size[0]), self.background_value,
                                           dtype=torch.long)
                curation_parameter = get_siamfc_cropping_params(pred_bbox, self.template_area_factor, np.array(self.template_size))
                template_cropped_bbox = get_foreground_bounding_box(pred_bbox, curation_parameter, self.stride)
                assert bbox_is_valid(template_cropped_bbox)
                template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
                template_mask[template_cropped_bbox[1]: template_cropped_bbox[3],
                template_cropped_bbox[0]: template_cropped_bbox[2]] = self.foreground_value
                template_mask = template_mask.to(self.device)
                if self.per_modality:
                    masks = self.template_mask_cache.get(task.id).clone()
                    for modality, valid in enumerate(available):
                        if valid:
                            masks[modality] = template_mask
                    template_mask = masks
                self.template_mask_cache.update(task.id, template_mask)

    def end(self, context: TrackerEvaluationPipeline_Context):
        for task in context.input_data.tasks:
            if task.do_task_finalization:
                self.template_mask_cache.delete(task.id)
