import numpy as np
import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        if self.transform:
            image, target = self.transform(image=image, target=target)
        return image, target
