"""ChestX-Det10 instance-level box detection (https://arxiv.org/abs/2006.10550).

Expected layout::

    <root>/train-old/<file_name>.png    <root>/train.json
    <root>/test_data/<file_name>.png    <root>/test.json
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .vindr import detection_transforms, expand_lesions, scale_boxes, to_detection_sample

LESION_MAPPING = {
    'atelectasis': ['Atelectasis'],
    'calcification': ['Calcification'],
    'consolidation': ['Consolidation'],
    'effusion': ['Effusion'],
    'emphysema': ['Emphysema'],
    'fibrosis': ['Fibrosis'],
    'fracture': ['Fracture'],
    'mass': ['Mass'],
    'nodule': ['Nodule'],
    'pneumothorax': ['Pneumothorax'],
}

LESION_GROUPS = {
    'concentrated': ['calcification', 'pneumothorax', 'consolidation', 'effusion', 'atelectasis'],
    'dispersed': ['nodule', 'mass', 'fracture', 'fibrosis', 'emphysema'],
}


class ChestXDet10Dataset(Dataset):
    """Returns ``(image [3, H, W], {'boxes': [N, 4], 'labels': [N]})``."""

    def __init__(self, data_root, split='train', target_lesion='all', target_size=(1024, 1024),
                 include_negative=False):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.include_negative = include_negative
        self.target_lesions, self.target_classes = expand_lesions(
            target_lesion, LESION_MAPPING, LESION_GROUPS)

        if split == 'train':
            self.image_dir = self.data_root / 'train-old'
            self.annotation_file = self.data_root / 'train.json'
        else:
            self.image_dir = self.data_root / 'test_data'
            self.annotation_file = self.data_root / 'test.json'
        self.annotations = self._load_annotations()
        self.image_ids = self._filter_images()
        self.transforms = detection_transforms(target_size)
        self.augment = True
        self._orig_sizes = {}
        print(f'ChestX-Det10 {split}: {len(self.image_ids)} images, classes={self.target_classes}')

    def _load_annotations(self):
        if not self.annotation_file.exists():
            raise FileNotFoundError(f'Annotation file not found: {self.annotation_file}')
        with open(self.annotation_file, 'r') as f:
            entries = json.load(f)
        return {e['file_name']: {'syms': e.get('syms', []), 'boxes': e.get('boxes', [])} for e in entries}

    def _filter_images(self):
        image_ids = []
        for file_name, ann in self.annotations.items():
            if not (self.image_dir / file_name).exists():
                continue
            if any(sym in self.target_classes for sym in ann['syms']) or self.include_negative:
                image_ids.append(file_name)
        return image_ids

    def _target_boxes(self, file_name):
        ann = self.annotations[file_name]
        return [tuple(box[:4]) for sym, box in zip(ann['syms'], ann['boxes']) if sym in self.target_classes]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        file_name = self.image_ids[idx]
        image = cv2.imread(str(self.image_dir / file_name))
        if image is None:
            print(f'Failed to load {file_name}')
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
            boxes = []
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = self._target_boxes(file_name)
        return to_detection_sample(image, boxes, image.shape[0], image.shape[1], self.transforms[self.augment])

    def gt_boxes(self, idx):
        """Ground-truth boxes of item ``idx`` in ``target_size`` pixel coordinates, without augmentation."""
        file_name = self.image_ids[idx]
        if file_name not in self._orig_sizes:
            with Image.open(self.image_dir / file_name) as im:
                self._orig_sizes[file_name] = (im.height, im.width)
        orig_h, orig_w = self._orig_sizes[file_name]
        return scale_boxes(self._target_boxes(file_name), orig_h, orig_w, self.target_size)


def create_chestxdet10_datasets(args):
    """AL pool = official training split, held-out set = official test split."""
    kwargs = dict(data_root=args.chestxdet10_root, target_lesion=args.target_lesion,
                  include_negative=args.include_negative)
    pool = ChestXDet10Dataset(split='train', **kwargs)
    val = ChestXDet10Dataset(split='test', **kwargs)
    return pool, val
