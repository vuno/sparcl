from .chestxdet10 import ChestXDet10Dataset, create_chestxdet10_datasets
from .ddr import DDRDataset, create_ddr_datasets
from .siim import SIIMDataset, create_siim_datasets
from .vindr import VinDrCXRDataset, augmentation_disabled, create_vindr_datasets

DATASETS = {
    'siim': ('segmentation', create_siim_datasets),
    'ddr': ('segmentation', create_ddr_datasets),
    'vindr': ('detection', create_vindr_datasets),
    'chestxdet10': ('detection', create_chestxdet10_datasets),
}


def task_type(dataset_name):
    return DATASETS[dataset_name][0]


def create_al_datasets(args):
    """Return ``(pool_dataset, heldout_dataset)`` for ``args.dataset``."""
    return DATASETS[args.dataset][1](args)


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


__all__ = ['ChestXDet10Dataset', 'DDRDataset', 'SIIMDataset', 'VinDrCXRDataset',
           'augmentation_disabled', 'create_al_datasets', 'detection_collate_fn', 'task_type']
