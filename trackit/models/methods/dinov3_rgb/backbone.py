from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from .lora import TrackingBlock


@dataclass(frozen=True)
class TrackingCrop:
    image: Tensor
    role: str  # z: initial template; d: online template; x: current search
    modality: str = 'rgb'
    foreground_mask: Optional[Tensor] = None


@dataclass(frozen=True)
class TrackingFeatureBundle:
    # Grid sizes use (height, width), unlike the tracker's config (width, height).
    grids: Mapping[str, tuple[int, int]]
    independent: Mapping[str, Tensor]
    isolated: Mapping[str, Tensor]
    joint: Mapping[str, Tensor]


class DINOv3TrackingBackbone(nn.Module):
    """Independent crop encoding followed by joint template/search encoding.

    encode_crops returns one dense patch-token tensor per real input crop.
    A future RGB-T tracker can omit unavailable current-search crops, retain
    historical templates, and consume these features before calling any head.
    """
    def __init__(self, vit, independent_depth=6, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not 0 <= independent_depth < len(vit.blocks):
            raise ValueError('independent_depth must be in [0, number of blocks)')
        vit.requires_grad_(False)
        self.embed_dim = vit.embed_dim
        self.patch_size = vit.patch_size
        self.independent_depth = independent_depth
        self.patch_embed = vit.patch_embed
        self.rope_embed = vit.rope_embed
        self.cls_token = vit.cls_token
        self.storage_tokens = vit.storage_tokens
        self.num_prefix_tokens = 1 + vit.n_storage_tokens
        self.norm = vit.norm
        self.blocks = nn.ModuleList([
            TrackingBlock(block, 'rgb' if i < independent_depth else 'shared', rank, alpha, dropout)
            for i, block in enumerate(vit.blocks)
        ])
        # z background/foreground, d background/foreground, search.
        self.role_embed = nn.Parameter(torch.empty(5, self.embed_dim))
        nn.init.trunc_normal_(self.role_embed, std=0.02)
        self.modality_embed = nn.ParameterDict({'rgb': nn.Parameter(torch.zeros(self.embed_dim))})

    def add_modality(self, name, copy_from='rgb'):
        """Call after loading the RGB checkpoint, before creating the optimizer/DDP.

        Only the shallow adapters and modality embedding are duplicated. All
        pretrained weights, deep tracking adapters, and role embeddings stay shared.
        """
        if name in self.modality_embed or not name or '.' in name:
            raise ValueError(f'Invalid or existing modality: {name}')
        if copy_from not in self.modality_embed:
            raise ValueError(f'Unknown source modality: {copy_from}')
        self.modality_embed[name] = nn.Parameter(self.modality_embed[copy_from].detach().clone())
        for block in self.blocks[:self.independent_depth]:
            block.adapters[name] = deepcopy(block.adapters[copy_from])

    def _prepare_crop(self, crop):
        image = crop.image
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Each real crop must have shape (B, 3, H, W)')
        if image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError('Crop dimensions must be multiples of the DINOv3 patch size')
        if crop.modality not in self.modality_embed:
            raise ValueError(f'Unknown modality {crop.modality}; call add_modality first')
        patches = self.patch_embed(image)
        batch, height, width, dim = patches.shape
        patches = patches.reshape(batch, height * width, dim)
        if crop.role in ('z', 'd'):
            mask = crop.foreground_mask
            if mask is None or mask.shape != (batch, height, width):
                raise ValueError('Template foreground mask must match its patch grid')
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError('Template foreground masks must be binary')
            offset = 0 if crop.role == 'z' else 2
            role_embedding = self.role_embed[offset:offset + 2][mask.flatten(1).long()]
        elif crop.role == 'x':
            role_embedding = self.role_embed[4]
        else:
            raise ValueError(f'Unknown crop role: {crop.role}')
        patches = patches + role_embedding + self.modality_embed[crop.modality]
        prefix = torch.cat((self.cls_token, self.storage_tokens), dim=1).expand(batch, -1, -1)
        return torch.cat((prefix, patches), dim=1), self.rope_embed(H=height, W=width)

    def encode_feature_bundle(self, crops: Mapping[str, TrackingCrop], *,
                              include_independent=True, include_isolated=False, include_joint=True):
        """Expose separate feature spaces without changing weights or checkpoint keys.

        independent: normalized patches after the shallow crop-specific blocks.
        isolated: final normalized patches with NO attention to another crop.
        joint: final normalized patches after joint template/search attention.
        The isolated path is optional and runs extra deep-block computations.
        """
        if not crops:
            raise ValueError('At least one real crop is required')
        if not (include_independent or include_isolated or include_joint):
            raise ValueError('Request at least one feature stage')
        prepared = [self._prepare_crop(crop) for crop in crops.values()]
        tokens = [item[0] for item in prepared]
        ropes = [item[1] for item in prepared]
        if len({t.shape[0] for t in tokens}) != 1:
            raise ValueError('All crops must have the same batch size')
        for block in self.blocks[:self.independent_depth]:
            tokens = [block(t, rope, crop.modality)
                      for t, rope, crop in zip(tokens, ropes, crops.values())]

        prefix_count = self.num_prefix_tokens
        grids = {name: (crop.image.shape[-2] // self.patch_size, crop.image.shape[-1] // self.patch_size)
                 for name, crop in crops.items()}
        independent = ({name: self.norm(t[:, prefix_count:]) for name, t in zip(crops, tokens)}
                       if include_independent else {})
        isolated = {}
        if include_isolated:
            for name, token, rope in zip(crops, tokens, ropes):
                for block in self.blocks[self.independent_depth:]:
                    token = block(token, rope, 'shared')
                isolated[name] = self.norm(token[:, prefix_count:])
        if not include_joint:
            return TrackingFeatureBundle(grids, independent, isolated, {})
        lengths = [t.shape[1] - prefix_count for t in tokens]
        # Official attention exempts a single prefix segment from RoPE. Move all
        # CLS/register tokens to the front, followed by all crop patch segments.
        joint = torch.cat([t[:, :prefix_count] for t in tokens] +
                          [t[:, prefix_count:] for t in tokens], dim=1)
        joint_rope = tuple(torch.cat([rope[i] for rope in ropes], dim=-2) for i in range(2))
        for block in self.blocks[self.independent_depth:]:
            joint = block(joint, joint_rope, 'shared')
        patches = self.norm(joint[:, prefix_count * len(tokens):])
        return TrackingFeatureBundle(grids, independent, isolated,
                                     dict(zip(crops, patches.split(lengths, dim=1))))

    def encode_crops(self, crops: Mapping[str, TrackingCrop]):
        return self.encode_feature_bundle(crops, include_independent=False).joint

    def encode_isolated_crops(self, crops: Mapping[str, TrackingCrop]):
        return self.encode_feature_bundle(crops, include_independent=False,
                                          include_isolated=True, include_joint=False).isolated

    def forward(self, z, x, d, z_feat_mask, d_feat_mask):
        return self.encode_crops({
            'z': TrackingCrop(z, 'z', foreground_mask=z_feat_mask),
            'd': TrackingCrop(d, 'd', foreground_mask=d_feat_mask),
            'x': TrackingCrop(x, 'x'),
        })
