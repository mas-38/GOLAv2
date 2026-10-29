"""SCDT JSON protocol: {sequence: {"data": [[RGB, TIR], ...]}}."""
import json
import numpy as np


class MissingModalitySchedule:
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.sequences = {}
        for name, record in data.items():
            flags = np.asarray(record['data'])
            if flags.ndim != 2 or flags.shape[1] != 2 or not len(flags):
                raise ValueError(f'{name}: expected nonempty N x 2 modality flags')
            if not np.isin(flags, [0, 1]).all():
                raise ValueError(f'{name}: modality flags must be binary')
            self.sequences[name] = flags.astype(bool)

    def get(self, name, index):
        if name not in self.sequences:
            raise KeyError(f'Missing modality schedule for sequence {name!r}')
        flags = self.sequences[name]
        if index < 0 or index >= len(flags):
            raise IndexError(f'{name}: frame {index} outside schedule length {len(flags)}')
        return tuple(bool(v) for v in flags[index])


def decode_available_modalities(getters, available):
    """Never read a missing image. Copy the valid image only as storage padding."""
    rgb, tir = available
    if not rgb and not tir:
        return None
    if not isinstance(getters, list) or len(getters) != 2:
        raise ValueError('Missing-modality evaluation requires RGB-T image getters')
    visible = getters[0]() if rgb else None
    thermal = getters[1]() if tir else None
    if visible is None:
        visible = thermal
    if thermal is None:
        thermal = visible
    return np.concatenate([visible, thermal], axis=-1)
