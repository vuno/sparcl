"""VinDr-CXR bounding-box detection (https://vindr.ai/datasets/cxr).

Expected layout::

    <root>/dicom/{train,test}/<image_id>.dicom
    <root>/old/annotations_{train,test}.csv
"""

import csv
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

LESION_MAPPING = {
    'calcification': ['Calcification'],
    'nodule': ['Nodule/Mass'],
    'cardiomegaly': ['Cardiomegaly'],
    'pleural_effusion': ['Pleural effusion'],
    'consolidation': ['Consolidation'],
    'pneumothorax': ['Pneumothorax'],
    'aortic_enlargement': ['Aortic enlargement'],
    'infiltration': ['Infiltration'],
    'lung_opacity': ['Lung Opacity'],
}

LESION_GROUPS = {
    'concentrated': ['calcification', 'aortic_enlargement', 'pneumothorax',
                     'consolidation', 'pleural_effusion', 'cardiomegaly'],
    'dispersed': ['nodule', 'infiltration', 'lung_opacity'],
}


def detection_transforms(target_size):
    """``{True: augmented, False: resize + normalize}`` box-aware transforms."""
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3, min_area=100)
    resize = A.Resize(target_size[0], target_size[1])
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmented = A.Compose([
        resize,
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        normalize,
    ], bbox_params=bbox_params)
    return {True: augmented, False: A.Compose([resize, normalize], bbox_params=bbox_params)}


@contextmanager
def augmentation_disabled(dataset):
    """Temporarily serve un-augmented items from a detection dataset."""
    previous = dataset.augment
    dataset.augment = False
    try:
        yield dataset
    finally:
        dataset.augment = previous


def to_detection_sample(image, boxes, orig_h, orig_w, transform):
    """Clip boxes, apply the transform and return ``(image tensor, target dict)``."""
    bbox_list, label_list = [], []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        if x2 > x1 and y2 > y1:
            bbox_list.append([x1, y1, x2, y2])
            label_list.append(1)

    out = transform(image=image, bboxes=bbox_list, labels=label_list)
    image = torch.from_numpy(out['image']).permute(2, 0, 1).float()
    if len(out['bboxes']) > 0:
        target = {'boxes': torch.tensor(out['bboxes'], dtype=torch.float32),
                  'labels': torch.tensor(out['labels'], dtype=torch.int64)}
    else:
        target = {'boxes': torch.zeros((0, 4), dtype=torch.float32),
                  'labels': torch.zeros((0,), dtype=torch.int64)}
    return image, target


def scale_boxes(boxes, orig_h, orig_w, target_size):
    """Clip boxes to the original image and rescale them to ``target_size`` (no augmentation)."""
    sy, sx = target_size[0] / orig_h, target_size[1] / orig_w
    scaled = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        if x2 > x1 and y2 > y1:
            scaled.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return np.asarray(scaled, dtype=np.float64).reshape(-1, 4)


def expand_lesions(target_lesion, mapping, groups):
    names = [target_lesion] if isinstance(target_lesion, str) else list(target_lesion)
    lesions = []
    for name in (n.lower() for n in names):
        if name in groups:
            lesions.extend(groups[name])
        elif name == 'all':
            lesions.extend(mapping.keys())
        elif name in mapping:
            lesions.append(name)
        else:
            raise ValueError(f'Unknown target lesion: {name}. Available: {list(mapping)}')
    lesions = list(dict.fromkeys(lesions))
    classes = list(dict.fromkeys(c for lesion in lesions for c in mapping[lesion]))
    return lesions, classes


class VinDrCXRDataset(Dataset):
    """Returns ``(image [3, H, W], {'boxes': [N, 4], 'labels': [N]})``."""

    def __init__(self, data_root, split='train', target_lesion='dispersed', target_size=(1024, 1024),
                 min_radiologist_agreement=1, include_negative=False):
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.min_radiologist_agreement = min_radiologist_agreement
        self.include_negative = include_negative
        self.target_lesions, self.target_classes = expand_lesions(
            target_lesion, LESION_MAPPING, LESION_GROUPS)

        self.dicom_dir = self.data_root / 'dicom' / split
        self.annotation_file = self.data_root / 'old' / f'annotations_{split}.csv'
        self.annotations = self._load_annotations()
        self.image_ids = self._filter_images()
        self.transforms = detection_transforms(target_size)
        self.augment = True
        self._orig_sizes = {}
        print(f'VinDr-CXR {split}: {len(self.image_ids)} images, classes={self.target_classes}')

    def _load_annotations(self):
        if not self.annotation_file.exists():
            raise FileNotFoundError(f'Annotation file not found: {self.annotation_file}')
        annotations = defaultdict(list)
        with open(self.annotation_file, 'r') as f:
            for row in csv.DictReader(f):
                bbox = None
                if row.get('x_min') and row.get('y_min') and row.get('x_max') and row.get('y_max'):
                    try:
                        bbox = (float(row['x_min']), float(row['y_min']),
                                float(row['x_max']), float(row['y_max']))
                    except ValueError:
                        bbox = None
                annotations[row['image_id']].append({'class_name': row['class_name'], 'bbox': bbox})
        return annotations

    def _target_boxes(self, image_id):
        return [ann['bbox'] for ann in self.annotations[image_id]
                if ann['class_name'] in self.target_classes and ann['bbox'] is not None]

    def _filter_images(self):
        positive, negative = [], []
        for image_id in self.annotations:
            if not (self.dicom_dir / f'{image_id}.dicom').exists():
                continue
            if len(self._target_boxes(image_id)) >= self.min_radiologist_agreement:
                positive.append(image_id)
            elif self.include_negative:
                negative.append(image_id)
        return positive + negative

    def _load_dicom(self, image_id):
        ds = pydicom.dcmread(str(self.dicom_dir / f'{image_id}.dicom'))
        image = ds.pixel_array
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            center, width = ds.WindowCenter, ds.WindowWidth
            if isinstance(center, pydicom.multival.MultiValue):
                center = center[0]
            if isinstance(width, pydicom.multival.MultiValue):
                width = width[0]
            lo, hi = center - width / 2, center + width / 2
            image = np.clip(image, lo, hi)
            image = ((image - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        if getattr(ds, 'PhotometricInterpretation', None) == 'MONOCHROME1':
            image = 255 - image
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return image, ds.Rows, ds.Columns

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        try:
            image, orig_h, orig_w = self._load_dicom(image_id)
            boxes = self._target_boxes(image_id)
        except Exception as e:
            print(f'Failed to load {image_id}: {e}')
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.target_size
            boxes = []
        return to_detection_sample(image, boxes, orig_h, orig_w, self.transforms[self.augment])

    def gt_boxes(self, idx):
        """Ground-truth boxes of item ``idx`` in ``target_size`` pixel coordinates, without augmentation."""
        image_id = self.image_ids[idx]
        if image_id not in self._orig_sizes:
            ds = pydicom.dcmread(str(self.dicom_dir / f'{image_id}.dicom'), stop_before_pixels=True)
            self._orig_sizes[image_id] = (int(ds.Rows), int(ds.Columns))
        orig_h, orig_w = self._orig_sizes[image_id]
        return scale_boxes(self._target_boxes(image_id), orig_h, orig_w, self.target_size)


def create_vindr_datasets(args):
    """AL pool = official training split, held-out set = official test split."""
    kwargs = dict(data_root=args.vindr_root, target_lesion=args.target_lesion,
                  min_radiologist_agreement=args.min_radiologist_agreement,
                  include_negative=args.include_negative)
    pool = VinDrCXRDataset(split='train', **kwargs)
    val = VinDrCXRDataset(split='test', **kwargs)
    return pool, val
