from torch import nn
from trackit.models import ModelBuildingContext


def get_model_build_context(config: dict) -> ModelBuildingContext:
    # Zekai Shao: Add GOLA build support
    if config['type'] == 'GOLA':
        from .GOLA.builder import get_GOLA_build_context
        build_context = get_GOLA_build_context(config)
    elif config['type'] == 'DINOv2Missing':
        from .dinov2_missing.builder import get_build_context
        build_context = get_build_context(config)
    elif config['type'] == 'DINOv3RGB':
        from .dinov3_rgb.builder import get_build_context
        build_context = get_build_context(config)
    else:
        raise NotImplementedError()
    if isinstance(build_context, nn.Module):
        model = build_context
        build_context = ModelBuildingContext(lambda _: model, lambda _: model.__class__.__name__, None)
    return build_context
