from .coreset import select_coreset
from .loss_prediction import select_loss_prediction
from .paal import PAAL
from .simple import (select_area_random, select_area_uncertainty, select_diversity, select_random,
                     select_uncertainty)
from .sparcl import (box_lesion_distribution, mask_lesion_distribution, score_detection, score_segmentation,
                     select_sparcl)
from .taudis import select_taudis
from .usimc import select_usimc

SEGMENTATION_ONLY = {'loss_prediction', 'paal'}
STRATEGIES = ['sparcl', 'random', 'area_random', 'uncertainty', 'uncertainty_area', 'coreset', 'diversity',
              'taudis', 'usimc', 'loss_prediction', 'paal']

__all__ = ['PAAL', 'SEGMENTATION_ONLY', 'STRATEGIES', 'box_lesion_distribution', 'mask_lesion_distribution',
           'score_detection', 'score_segmentation', 'select_area_random', 'select_area_uncertainty',
           'select_coreset', 'select_diversity', 'select_loss_prediction', 'select_random', 'select_sparcl',
           'select_taudis', 'select_uncertainty', 'select_usimc']
