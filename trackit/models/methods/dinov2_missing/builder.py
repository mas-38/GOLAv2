from trackit.models import ModelBuildingContext
from trackit.models.backbone.builder import build_backbone
from .model import DINOv2MissingTracker
from .sample_data_generator import build_sample_input_data_generator


def get_build_context(config):
    def build(advice):
        backbone = build_backbone(config['model']['backbone'],
                                  torch_jit_trace_compatible=advice.torch_jit_trace_compatible)
        return DINOv2MissingTracker(backbone, config['common']['template_feat_size'],
                                   config['common']['search_region_feat_size'])
    return ModelBuildingContext(build, lambda advice: 'DINOv2Missing',
                                build_sample_input_data_generator(config))
