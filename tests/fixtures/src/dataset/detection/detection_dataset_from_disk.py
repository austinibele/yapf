import numpy as np
import torch
from torch.utils.data import Dataset


class DetectionDatasetFromDisk(Dataset):
    """
    Inputs:
        image_paths (List[Path]): list of paths to image files.
        target_paths (List[Path]): list of paths to target files.
        image_loader (Callable): A callable which loads the image from the image path.
        target_loader (Callable): A callable which loads the target from the target path.
    """
    def __init__(self, image_paths, target_paths, image_loader, target_loader, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_loader = image_loader
        self.target_loader = target_loader
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_loader(self.image_paths[idx])
        target = self.target_loader(self.target_paths[idx])
        if self.transform:
            image, target = self.transform(image=image, target=target)
        return image, target
