"""SIIM-ACR pneumothorax segmentation (https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation).

Expected layout::

    <root>/images/<resolution>/dicom/<image_id>.png
    <root>/images/<resolution>/mask/<image_id>.png
"""

import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _scan_samples(root, resolution, positive_only):
    image_dir = Path(root) / 'images' / str(resolution) / 'dicom'
    mask_dir = Path(root) / 'images' / str(resolution) / 'mask'
    if not image_dir.exists():
        raise ValueError(f'Image directory not found: {image_dir}')
    if not mask_dir.exists():
        raise ValueError(f'Mask directory not found: {mask_dir}')

    samples = []
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith('.png'):
            continue
        mask_path = mask_dir / name
        if not mask_path.exists():
            continue
        has_lesion = np.array(Image.open(mask_path)).max() > 0
        if positive_only and not has_lesion:
            continue
        samples.append({
            'image_id': name[:-len('.png')],
            'image_path': str(image_dir / name),
            'mask_path': str(mask_path),
            'has_lesion': has_lesion,
        })
    return samples


class SIIMDataset(Dataset):
    """Returns ``(image [3, H, W], mask [1, H, W])``."""

    def __init__(self, samples, target_size=(512, 512)):
        self.samples = samples
        self.target_size = target_size
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485], std=[0.229]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample['image_path']).convert('L'))
        image = np.stack([image] * 3, axis=-1)
        mask = (np.array(Image.open(sample['mask_path'])) > 127).astype(np.float32)

        out = self.transform(image=image, mask=mask)
        image = torch.from_numpy(np.transpose(out['image'], (2, 0, 1)).astype(np.float32))
        mask = torch.from_numpy(out['mask'][None].astype(np.float32))
        return image, mask


def create_siim_datasets(args):
    """Stratified random split of the SIIM training set into an AL pool and a held-out set."""
    samples = _scan_samples(args.siim_root, resolution=1024, positive_only=args.siim_positive_only)

    random.seed(args.seed)
    np.random.seed(args.seed)
    order = list(range(len(samples)))
    random.shuffle(order)

    positive = [i for i in order if samples[i]['has_lesion']]
    negative = [i for i in order if not samples[i]['has_lesion']]
    n_val_pos = max(1, int(args.num_validation_samples * len(positive) / len(samples)))
    n_val_neg = args.num_validation_samples - n_val_pos
    val_indices = positive[:n_val_pos] + negative[:n_val_neg]
    train_indices = [i for i in order if i not in val_indices]

    pool = SIIMDataset([samples[i] for i in train_indices])
    val = SIIMDataset([samples[i] for i in val_indices])
    print(f'SIIM-ACR: pool={len(pool)}, held-out={len(val)}')
    return pool, val
