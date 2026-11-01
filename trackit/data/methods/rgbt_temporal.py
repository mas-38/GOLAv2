"""LasHeR clips: causal inputs and privileged supervision use separate dictionaries."""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


MEAN = torch.tensor([.485, .456, .406, .449, .449, .449])[:, None, None]
STD = torch.tensor([.229, .224, .225, .226, .226, .226])[:, None, None]


def split_names(root, split):
    root = Path(root)
    train = (root / 'trainingsetList.txt').read_text().splitlines()
    test = (root / 'testingsetList.txt').read_text().splitlines()
    if len(train) != 979 or len(test) != 245 or set(train) & set(test):
        raise ValueError('Expected disjoint official LasHeR 979-train / 245-test manifests')
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Duplicate sequences in LasHeR split manifests')
    if split not in ('train', 'test'):
        raise ValueError('Use the user-selected official train or test split')
    return train if split == 'train' else test


class LasHeRSequence:
    def __init__(self, root, name, split='train'):
        self.name = name
        directory = Path(root) / ('trainingset' if split == 'train' else 'testingset') / name
        self.rgb = sorted((directory / 'visible').glob('*.jpg'))
        self.tir = sorted((directory / 'infrared').glob('*.jpg'))
        # Some distributed copies use PNG. Never silently truncate via zip().
        if not self.rgb:
            self.rgb = sorted((directory / 'visible').glob('*.png'))
        if not self.tir:
            self.tir = sorted((directory / 'infrared').glob('*.png'))
        self.boxes = np.loadtxt(directory / 'init.txt', delimiter=',', ndmin=2).astype(np.float32)
        self.boxes[:, 2:] += self.boxes[:, :2]
        if len(self.rgb) != len(self.tir) or len(self.rgb) != len(self.boxes) or not len(self.rgb):
            raise ValueError(f'{name}: RGB/TIR/annotation lengths differ')
        if self.boxes.shape[1] != 4:
            raise ValueError(f'{name}: expected XYWH annotation rows')

    def image(self, index, available=(True, True)):
        images = []
        for enabled, paths in zip(available, (self.rgb, self.tir)):
            if not enabled:
                images.append(None)
                continue
            image = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f'Cannot decode {paths[index]}')
            images.append(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.)
        shapes = {tuple(image.shape) for image in images if image is not None}
        if not shapes:
            return None
        if len(shapes) != 1:
            raise ValueError(f'{self.name}: registered RGB/TIR image sizes differ')
        shape = next(iter(shapes))
        # Padding is never a modality observation and must not reach its encoder.
        return torch.cat([image if image is not None else torch.full(shape, float('nan')) for image in images])


def crop_image(image, crop_box, output_hw):
    """Subpixel crop, normalized original-image XYXY, channel-mean padding."""
    box = torch.as_tensor(crop_box, dtype=torch.float32, device=image.device)
    height, width = output_hw
    y = (torch.arange(height, device=image.device, dtype=torch.float32) + .5) / height
    x = (torch.arange(width, device=image.device, dtype=torch.float32) + .5) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack((xx, yy), -1) * (box[2:] - box[:2]) + box[:2]
    # Process real channels independently; NaN padding remains NaN in missing slots.
    mean = image.mean((-2, -1), keepdim=True)
    return (F.grid_sample((image - mean)[None], (points * 2 - 1)[None], mode='bilinear',
                          padding_mode='zeros', align_corners=False)[0] + mean)


def crop_from_box(box, image_hw, area_factor):
    height, width = image_hw
    box = torch.as_tensor(box, dtype=torch.float32)
    extent = (box[2:] - box[:2]).clamp_min(1e-6)
    pixel_extent = extent * box.new_tensor([width, height])
    side = (pixel_extent + (area_factor - 1) * pixel_extent.mean()).prod().sqrt()
    half = side / box.new_tensor([width, height]) / 2
    center = (box[:2] + box[2:]) / 2
    return torch.cat((center - half, center + half))


def foreground_mask(box, crop, grid_hw):
    height, width = grid_hw
    local = (box - crop[:2].repeat(2)) / (crop[2:] - crop[:2]).repeat(2)
    yy, xx = torch.meshgrid((torch.arange(height) + .5) / height,
                            (torch.arange(width) + .5) / width, indexing='ij')
    return ((xx >= local[0]) & (xx <= local[2]) & (yy >= local[1]) & (yy <= local[3])).long()


def normalize(image):
    return (image - MEAN.to(image)) / STD.to(image)


def availability_schedule(length, rng, persistence=.8):
    states = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    flags = np.ones((length, 2), dtype=bool)
    current = int(rng.integers(3))
    for index in range(1, length):
        if rng.random() > persistence:
            current = int(rng.integers(3))
        flags[index] = states[current]
    return torch.from_numpy(flags)


class LasHeRTemporalClips(Dataset):
    def __init__(self, root, *, stage, samples, seed=42, template_hw=(128, 128),
                 search_hw=(256, 256), history_frames=3, horizons=(1, 2)):
        if stage not in ('adapt', 'temporal'):
            raise ValueError(stage)
        self.root, self.stage, self.samples, self.seed = str(root), stage, int(samples), int(seed)
        self.template_hw, self.search_hw = tuple(template_hw), tuple(search_hw)
        self.history_frames = history_frames if stage == 'temporal' else 1
        self.horizons = tuple(horizons) if stage == 'temporal' else ()
        self.epoch = 0
        self.names = split_names(root, 'train')
        self.sequences = [LasHeRSequence(root, name) for name in self.names]
        minimum = self.history_frames + 1 + max(self.horizons, default=0)
        if any(len(sequence.rgb) < minimum for sequence in self.sequences):
            raise ValueError('A LasHeR sequence is too short for the configured temporal window')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        # Sampling does not depend on worker count or prefetch state; resumable by index.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        for attempt in range(32):
            sequence = self.sequences[int(rng.integers(len(self.sequences)))]
            length = len(sequence.rgb)
            if self.stage == 'adapt':
                start = int(rng.integers(length - 1))
                current = min(start + int(rng.integers(1, 101)), length - 1)
                frames = [start, current]
            else:
                total = self.history_frames + 1 + max(self.horizons)
                start = int(rng.integers(length - total + 1))
                frames = list(range(start, start + total))
            images = [sequence.image(frame) for frame in frames]
            height, width = images[0].shape[-2:]
            if any(image.shape != images[0].shape for image in images):
                raise ValueError(f'{sequence.name}: frame size changes inside a clip')
            sizes = torch.tensor([width, height, width, height])
            boxes = torch.from_numpy(sequence.boxes[frames]) / sizes
            required = boxes[[0, self.history_frames]]
            if not torch.isfinite(required).all() or not (required[:, 2:] > required[:, :2]).all():
                continue
            initial = boxes[0]
            search_crop = crop_from_box(initial, (height, width), 4.)
            # One crop anchored at the initial real observation, held fixed across the clip.
            # No current/future ground truth controls the predictor's coordinate system.
            local = (boxes[self.history_frames] - search_crop[:2].repeat(2)) / (search_crop[2:] - search_crop[:2]).repeat(2)
            intersection = (torch.minimum(local[2:], torch.ones(2)) - torch.maximum(local[:2], torch.zeros(2))).clamp_min(0)
            if not (intersection > 0).all():
                continue
            template_crop = crop_from_box(initial, (height, width), 2.)
            template = normalize(crop_image(images[0], template_crop, self.template_hw))
            z_mask = foreground_mask(initial, template_crop, (self.template_hw[0] // 16, self.template_hw[1] // 16))
            views = torch.stack([normalize(crop_image(image, search_crop, self.search_hw)) for image in images])
            causal_length = self.history_frames + 1
            flags = availability_schedule(causal_length, rng)
            causal = views[:causal_length].clone()
            for t in range(causal_length):
                for m in range(2):
                    if not flags[t, m]:
                        causal[t, m * 3:(m + 1) * 3] = float('nan')
            return dict(inputs=dict(z=template, context=causal, availability=flags,
                                    template_mask=z_mask, crop_box=search_crop,
                                    frame_indices=torch.tensor(frames[:causal_length]), initial_box=initial),
                        targets=dict(current_image=views[self.history_frames], current_box=boxes[self.history_frames],
                                     future_images=views[causal_length:], future_boxes=boxes[causal_length:],
                                     future_indices=torch.tensor(frames[causal_length:]),
                                     current_local_box=local, crop_box=search_crop),
                        sequence=sequence.name)
        raise RuntimeError(f'Failed to sample a visible causal clip after {attempt + 1} attempts')


def collate_clips(samples):
    batch = {section: {key: torch.stack([sample[section][key] for sample in samples])
                       for key in samples[0][section]} for section in ('inputs', 'targets')}
    batch['metadata'] = dict(sequences=[sample.get('sequence', 'synthetic') for sample in samples])
    return batch


def manifest_digest(names):
    return hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest()


def audit_lasher(root):
    from PIL import Image
    report = {}
    for split in ('train', 'test'):
        names = split_names(root, split)
        frames, invalid_boxes = 0, 0
        for name in names:
            sequence = LasHeRSequence(root, name, split)
            frames += len(sequence.rgb)
            invalid_boxes += int((~np.isfinite(sequence.boxes).all(1) |
                                  (sequence.boxes[:, 2:] <= sequence.boxes[:, :2]).any(1)).sum())
            for index in {0, len(sequence.rgb) - 1}:
                with Image.open(sequence.rgb[index]) as rgb, Image.open(sequence.tir[index]) as tir:
                    if rgb.size != tir.size:
                        raise ValueError(f'{name}: RGB/TIR geometry differs at frame {index}')
        report[split] = dict(sequences=len(names), paired_frames=frames, invalid_annotation_rows=invalid_boxes,
                             manifest_sha256=manifest_digest(names),
                             checks='All sequence frame/annotation counts; paired image headers at first/last frame')
    return report
