from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


MODALITIES = ('rgb', 'tir')


@dataclass
class FuturePrediction:
    appearance: torch.Tensor  # B, horizons, modalities, ROI tokens, C
    motion: torch.Tensor      # B, horizons, modalities, 4
    visibility: torch.Tensor  # B, horizons, modalities (logits)


class TemporalConditioning(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.project = nn.Linear(dim, hidden)
        self.modality = nn.Embedding(2, hidden)
        self.metadata = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))

    def current(self, observations):
        return torch.cat([self.project(o.tokens) + self.modality.weight[MODALITIES.index(o.modality)]
                          for o in observations], dim=1)

    def history(self, history, device):
        histories = history if isinstance(history, list) else [history]
        raw, metadata, modalities, valid = [], [], [], []
        for item in histories:
            if item is None or not len(item.modalities):
                raw.append(self.null.new_empty(0, self.project.in_features))
                metadata.append(self.null.new_empty(0, 8))
                modalities.append(torch.empty(0, device=device, dtype=torch.long))
                valid.append(torch.empty(0, device=device, dtype=torch.bool))
                continue
            # Time, source modality, confidence and geometry remain explicit.
            meta = torch.cat((item.boxes.float(), item.ages.float().log1p()[:, None],
                              item.confidences[:, None], item.initial.float()[:, None],
                              item.valid_mask.float().mean(1, keepdim=True)), dim=1)
            length = item.tokens.shape[1]
            ids = torch.tensor([MODALITIES.index(m) for m in item.modalities], device=device)
            raw.append(item.tokens.flatten(0, 1))
            metadata.append(meta.repeat_interleave(length, dim=0))
            modalities.append(ids.repeat_interleave(length))
            valid.append(item.valid_mask.flatten())
        raw = pad_sequence(raw, batch_first=True)
        meta = pad_sequence(metadata, batch_first=True)
        ids = pad_sequence(modalities, batch_first=True)
        mask = pad_sequence(valid, batch_first=True)
        tokens = self.project(raw) + self.modality(ids) + self.metadata(meta)
        batch = len(histories)
        # One valid null token per track handles empty/all-masked ROI histories.
        return torch.cat((self.null.expand(batch, -1, -1), tokens), dim=1), torch.cat((
            torch.zeros(batch, 1, device=device, dtype=torch.bool), ~mask), dim=1)


class FuturePredictor(nn.Module):
    def __init__(self, dim=768, hidden=256, heads=8, horizons=(1, 2), roi_hw=(4, 4)):
        super().__init__()
        self.horizons, self.roi_hw = tuple(horizons), tuple(roi_hw)
        self.roi_length = roi_hw[0] * roi_hw[1]
        self.condition = TemporalConditioning(dim, hidden)
        self.queries = nn.Parameter(torch.empty(1, len(horizons) * 2 * self.roi_length, hidden))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.anchor = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.appearance = nn.Linear(hidden, dim)
        self.motion = nn.Linear(hidden, 4)
        self.visibility = nn.Linear(hidden, 1)

    def forward(self, observations, history, anchor_box):
        current = self.condition.current(observations)
        past, mask = self.condition.history(history, current.device)
        context = torch.cat((current, past), dim=1)
        batch = current.shape[0]
        padding = torch.cat((torch.zeros(batch, current.shape[1], device=current.device, dtype=torch.bool), mask), 1)
        query = self.queries + self.anchor(anchor_box.detach())[:, None]
        token = query + self.attention(query, context, context, key_padding_mask=padding, need_weights=False)[0]
        token = token + self.ffn(self.norm(token))
        token = self.norm(token).reshape(batch, len(self.horizons), 2, self.roi_length, -1)
        pooled = token.mean(dim=3)
        return FuturePrediction(self.appearance(token), self.motion(pooled), self.visibility(pooled).squeeze(-1))


class SharedReconstructor(nn.Module):
    """One recurrent cell, three condition attentions, readouts after 1/3/5 updates."""
    def __init__(self, dim=768, hidden=256, heads=8, max_steps=5, grid_hw=(16, 16)):
        super().__init__()
        self.condition = TemporalConditioning(dim, hidden)
        self.grid_hw = tuple(grid_hw)
        self.position = nn.Parameter(torch.empty(1, grid_hw[0] * grid_hw[1], hidden))
        nn.init.trunc_normal_(self.position, std=.02)
        self.initial = nn.Linear(dim, hidden)
        self.step = nn.Embedding(max_steps, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.current_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.history_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.future_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.future_motion = nn.Linear(4, hidden)
        self.gates = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim))

    def forward(self, observations, history, future, missing_modality, steps):
        if steps not in (1, 3, 5) or steps > self.step.num_embeddings:
            raise ValueError('Reconstruction uses a shared cell with 1, 3 or 5 updates')
        current = self.condition.current(observations)
        past, padding = self.condition.history(history, current.device)
        index = MODALITIES.index(missing_modality)
        prior = self.condition.project(future.appearance[:, :, index])
        prior = prior + self.future_motion(future.motion[:, :, index])[:, :, None]
        prior = prior * future.visibility[:, :, index].sigmoid()[:, :, None, None]
        prior = prior.flatten(1, 2)
        dense = torch.stack([o.tokens for o in observations]).mean(0)
        state = self.initial(dense) + self.position + self.condition.modality.weight[index]
        readouts = {}
        for step in range(steps):
            query = self.query_norm(state) + self.step.weight[step]
            a = self.current_attention(query, current, current, need_weights=False)[0]
            h = self.history_attention(query, past, past, key_padding_mask=padding, need_weights=False)[0]
            u = self.future_attention(query, prior, prior, need_weights=False)[0]
            gates = self.gates(torch.cat((query, h, u), dim=-1)).sigmoid()
            state = state + a + gates[..., :1] * h + gates[..., 1:] * u
            state = state + self.ffn(self.ffn_norm(state))
            if step + 1 in (1, 3, 5):
                readouts[step + 1] = self.readout(state)
        return readouts


class ReconstructionFusion(nn.Module):
    """Map isolated-reference reconstruction to a gated joint tracking residual."""
    def __init__(self, dim=768, hidden=256, heads=8):
        super().__init__()
        self.query = nn.Linear(dim, hidden)
        self.key_value = nn.Linear(dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.output = nn.Linear(hidden, dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2 + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, joint, reconstructed, future, missing_modality):
        query, condition = self.query(joint), self.key_value(reconstructed)
        residual = self.output(self.attention(query, condition, condition, need_weights=False)[0])
        motion = future.motion[:, :, MODALITIES.index(missing_modality)].mean(1)
        gate = self.gate(torch.cat((joint.mean(1), reconstructed.mean(1), motion), dim=-1)).sigmoid()
        return joint + gate[:, None] * residual, gate
