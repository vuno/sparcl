"""DDR fundus lesion segmentation (https://github.com/nkicsl/DDR-dataset).

Expected layout::

    <root>/{train,valid,test}/image/<id>.jpg
    <root>/{train,test}/label/{EX,HE,MA,SE}/<id>.tif
    <root>/valid/segmentation label/{EX,HE,MA,SE}/<id>.tif
"""

import glob
import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

LESIONS = ('EX', 'HE', 'MA', 'SE')


def _scan_samples(root, split, lesion, positive_only):
    root = Path(root)
    image_dir = root / split / 'image'
    if not image_dir.exists():
        raise ValueError(f'Image directory not found: {image_dir}')
    mask_dir = None
    for label_name in ('label', 'segmentation label'):
        if (root / split / label_name / lesion).exists():
            mask_dir = root / split / label_name / lesion
            break
    if mask_dir is None:
        raise ValueError(f'No {lesion} masks found under {root / split}')

    samples = []
    for image_path in sorted(glob.glob(str(image_dir / '*.jpg'))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = mask_dir / f'{stem}.tif'
        if not mask_path.exists():
            continue
        has_lesion = bool(np.array(Image.open(mask_path)).max() > 0)
        if positive_only and not has_lesion:
            continue
        samples.append({'image_id': stem, 'image_path': image_path,
                        'mask_path': str(mask_path), 'has_lesion': has_lesion})
    return samples


class DDRDataset(Dataset):
    """Returns ``(image [3, H, W], mask [1, H, W])``."""

    def __init__(self, samples, target_size=(512, 512)):
        self.samples = samples
        self.target_size = target_size
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(sample['image_path']).convert('RGB'))
        mask = (np.array(Image.open(sample['mask_path'])) > 0).astype(np.float32)
        out = self.transform(image=image, mask=mask)
        image = torch.from_numpy(np.transpose(out['image'], (2, 0, 1)).astype(np.float32))
        mask = torch.from_numpy(out['mask'][None].astype(np.float32))
        return image, mask


def create_ddr_datasets(args):
    """AL pool = DDR training split, held-out set = stratified subsample of the DDR validation split."""
    lesion = args.ddr_lesion.upper()
    if lesion not in LESIONS:
        raise ValueError(f'ddr_lesion must be one of {LESIONS}, got {lesion}')

    pool = DDRDataset(_scan_samples(args.ddr_root, 'train', lesion, args.ddr_positive_only))
    val_samples = _scan_samples(args.ddr_root, 'valid', lesion, args.ddr_positive_only)

    random.seed(args.seed)
    np.random.seed(args.seed)
    order = list(range(len(val_samples)))
    random.shuffle(order)
    n_val = args.num_validation_samples
    if n_val and n_val < len(order):
        positive = [i for i in order if val_samples[i]['has_lesion']]
        negative = [i for i in order if not val_samples[i]['has_lesion']]
        n_pos = max(1, int(n_val * len(positive) / max(1, len(val_samples))))
        keep = positive[:n_pos] + negative[:n_val - n_pos]
    else:
        keep = order
    val = DDRDataset([val_samples[i] for i in keep])
    print(f'DDR ({lesion}): pool={len(pool)}, held-out={len(val)}')
    return pool, val
