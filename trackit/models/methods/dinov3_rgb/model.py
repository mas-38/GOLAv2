import torch
from torch import nn

from ..dinov2_missing.head import MlpAnchorFreeHead
from .backbone import DINOv3TrackingBackbone, TrackingCrop


class DINOv3RGBTracker(nn.Module):
    def __init__(self, vit, template_feat_size, search_region_feat_size,
                 independent_depth=6, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.template_feat_size = tuple(template_feat_size)
        self.search_region_feat_size = tuple(search_region_feat_size)
        self.backbone = DINOv3TrackingBackbone(vit, independent_depth, rank, alpha, dropout)
        self.head = MlpAnchorFreeHead(vit.embed_dim, self.search_region_feat_size)
        # Persist structural/scaling settings alongside full pretrained and LoRA
        # tensors, so --weight_path cannot silently apply a different LoRA scale.
        self.register_buffer('architecture', torch.tensor([
            1, vit.embed_dim, len(vit.blocks), vit.patch_size, independent_depth,
            rank, round(alpha * 1000000), round(dropout * 1000000),
            *self.template_feat_size, *self.search_region_feat_size,
        ], dtype=torch.int64))

    def _validate_sizes(self, z, x, d):
        patch_size = self.backbone.patch_size
        for image, grid in ((z, self.template_feat_size), (d, self.template_feat_size),
                            (x, self.search_region_feat_size)):
            expected_hw = (grid[1] * patch_size, grid[0] * patch_size)
            if image.shape[-2:] != expected_hw:
                raise ValueError(f'Expected crop size {expected_hw}, got {image.shape[-2:]}')
    def extract_feature_bundle(self, z, x, d, z_feat_mask, d_feat_mask, **stage_options):
        self._validate_sizes(z, x, d)
        return self.backbone.encode_feature_bundle({
            'z': TrackingCrop(z, 'z', foreground_mask=z_feat_mask),
            'd': TrackingCrop(d, 'd', foreground_mask=d_feat_mask),
            'x': TrackingCrop(x, 'x'),
        }, **stage_options)

    def forward_features(self, z, x, d, z_feat_mask, d_feat_mask):
        self._validate_sizes(z, x, d)
        return self.backbone(z, x, d, z_feat_mask, d_feat_mask)

    def predict_from_features(self, search_features):
        expected = self.search_region_feat_size[0] * self.search_region_feat_size[1]
        if search_features.ndim != 3 or search_features.shape[1:] != (expected, self.backbone.embed_dim):
            raise ValueError('Prediction head requires dense joint-search features matching its configured grid')
        return self.head(search_features)

    def forward(self, z, x, d, z_feat_mask, d_feat_mask):
        return self.predict_from_features(self.forward_features(z, x, d, z_feat_mask, d_feat_mask)['x'])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'architecture' not in state_dict:
            raise ValueError('Expected a complete DINOv3RGB checkpoint with architecture metadata')
        if not torch.equal(self.architecture.cpu(), state_dict['architecture'].cpu()):
            raise ValueError('Checkpoint architecture/LoRA settings do not match this model configuration')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
