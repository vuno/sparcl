from .detection import DetectionModel, create_detection_model
from .paal_predictor import AccuracyPredictor
from .segmentation import (LossPredictionModule, SegmentationModel,
                           SegmentationModelWithLossPrediction, create_segmentation_model)


def create_model(args, device, checkpoint_path=None):
    if args.task == 'detection':
        return create_detection_model(args, device, checkpoint_path)
    return create_segmentation_model(args, device, checkpoint_path)


__all__ = ['AccuracyPredictor', 'DetectionModel', 'LossPredictionModule', 'SegmentationModel',
           'SegmentationModelWithLossPrediction', 'create_model']
