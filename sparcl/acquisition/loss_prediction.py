"""Learning Loss for Active Learning (Yoo & Kweon, CVPR 2019): query the images with the highest predicted loss."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def select_loss_prediction(unlabeled, budget, model, dataset, device, batch_size, seed=None):
    if len(unlabeled) < budget:
        return list(unlabeled)
    if not hasattr(model, 'loss_prediction_module'):
        rng = np.random.RandomState(seed if seed is not None else 42)
        return rng.choice(unlabeled, size=budget, replace=False).tolist()

    model.eval()
    loader = DataLoader(Subset(dataset, unlabeled), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    predicted = []
    for images, _ in loader:
        _, loss = model(images.to(device), return_loss_prediction=True)
        predicted.extend(loss.cpu().numpy().flatten())
    top = np.argsort(np.array(predicted))[-budget:][::-1]
    return [unlabeled[i] for i in top]
