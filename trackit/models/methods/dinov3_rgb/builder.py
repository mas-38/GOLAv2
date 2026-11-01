import torch

from trackit.models import ModelBuildingContext
from trackit.models.backbone.builder import build_backbone
from trackit.models.schema.input.dummy_data_generation import SampleInputDataGeneratorInterface
from .model import DINOv3RGBTracker


class RGBSampleInputGenerator(SampleInputDataGeneratorInterface):
    def __init__(self, common):
        self.common = common

    def get(self, batch_size, device):
        template_w, template_h = self.common['template_size']
        search_w, search_h = self.common['search_region_size']
        feat_w, feat_h = self.common['template_feat_size']
        return dict(z=torch.randn(batch_size, 3, template_h, template_w, device=device),
                    d=torch.randn(batch_size, 3, template_h, template_w, device=device),
                    x=torch.randn(batch_size, 3, search_h, search_w, device=device),
                    z_feat_mask=torch.ones(batch_size, feat_h, feat_w, device=device, dtype=torch.long),
                    d_feat_mask=torch.ones(batch_size, feat_h, feat_w, device=device, dtype=torch.long))

    def get_batch_size(self, _):
        return 1


def get_build_context(config):
    def build(advice):
        vit = build_backbone(config['model']['backbone'],
                             torch_jit_trace_compatible=advice.torch_jit_trace_compatible)
        common = config['common']
        return DINOv3RGBTracker(vit, common['template_feat_size'], common['search_region_feat_size'],
                               **config['model']['tracking'])
    return ModelBuildingContext(build, lambda _: 'DINOv3RGB', RGBSampleInputGenerator(config['common']))
