# Derived from the RGB-T tracking scaffold by Zekai Shao.
# Licensed under Apache-2.0. Pure DINOv2, without low-rank adaptation.

from typing import Tuple

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .patch_embed import PatchEmbedNoSizeCheck
from .head import MlpAnchorFreeHead


class DINOv2MissingTracker(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int]):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[
            1]
        self.z_size = template_feat_size
        self.d_size = template_feat_size
        self.x_size = search_region_feat_size

        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))

        self.token_type_embed = nn.Parameter(torch.empty(5, self.embed_dim))
        trunc_normal_(self.token_type_embed, std=.02)

        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

    def forward(self, z, x, d, z_feat_mask, d_feat_mask, modality_available=None):
        """SCDT order: [RGB, TIR]. Historical templates remain valid.

        Training samples one state per batch, uniformly over RGB-only,
        TIR-only and complete. Explicit masks override sampling for validation.
        Missing current-search tokens are never embedded or attended to.
        """
        batch = x.shape[0]
        if modality_available is None:
            if self.training:
                states = torch.tensor([[True, False], [False, True], [True, True]], device=x.device)
                modality_available = states[torch.randint(3, (), device=x.device)].expand(batch, -1)
            else:
                modality_available = torch.ones((batch, 2), dtype=torch.bool, device=x.device)
        if modality_available.shape != (batch, 2):
            raise ValueError('modality_available must have shape (B, 2), ordered [RGB, TIR]')
        if not torch.all((modality_available == 0) | (modality_available == 1)):
            raise ValueError('Modality flags must be binary')
        available = modality_available.to(device=x.device, dtype=torch.bool)
        if not available.any(dim=1).all():
            raise ValueError('Both-missing frames must bypass the model and reuse the previous prediction')
        features, indices = [], []
        for rgb, tir in ((True, False), (False, True), (True, True)):
            selected = ((available[:, 0] == rgb) & (available[:, 1] == tir)).nonzero().flatten()
            if selected.numel() == 0:
                continue
            zm = z_feat_mask[selected]
            dm = d_feat_mask[selected]
            # Training has one shared mask; inference keeps one mask per modality.
            if zm.ndim == 3:
                zm = zm.unsqueeze(1).expand(-1, 2, -1, -1)
            if dm.ndim == 3:
                dm = dm.unsqueeze(1).expand(-1, 2, -1, -1)
            chunks = []
            for modality in range(2):
                channels = slice(modality * 3, (modality + 1) * 3)
                chunks.append(self._template(z[selected, channels], zm[:, modality], 0))
                chunks.append(self._template(d[selected, channels], dm[:, modality], 2))
            # Put the selected prediction modality last. Complete frames use TIR.
            if rgb:
                chunks.append(self._search(x[selected, :3]))
            if tir:
                chunks.append(self._search(x[selected, 3:]))
            encoded = torch.cat(chunks, dim=1)
            for block in self.blocks:
                encoded = block(encoded)
            features.append(self.norm(encoded)[:, -self.x_size[0] * self.x_size[1]:])
            indices.append(selected)
        order = torch.cat(indices).argsort()
        return self.head(torch.cat(features)[order])

    def _template(self, image, mask, type_offset):
        tokens = self.patch_embed(image)
        width, height = self.z_size
        pos = self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)
        pos = pos[:, :height, :width].reshape(1, height * width, self.embed_dim)
        return tokens + pos + self.token_type_embed[type_offset:type_offset + 2][mask.flatten(1)]

    def _search(self, image):
        return self.patch_embed(image) + self.pos_embed + self.token_type_embed[4]
