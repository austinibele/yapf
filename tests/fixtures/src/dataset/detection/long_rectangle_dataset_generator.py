import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter
from src.type_validation.pytorch_type_validator import PytorchTypeValidator


class LongRectangleDatasetGenerator:
    def __init__(self, background_size=(128, 128), n_rectangles=2):
        self.background_size = background_size
        self.max_rectangle_size = (4, 80)
        self.n_rectangles = n_rectangles

    def _make_targets_from_bboxes(self, bboxes):
        targets = []
        for bbox in bboxes:
            target = {}
            target["boxes"] = bbox
            target["labels"] = torch.from_numpy(np.zeros(shape=len(bbox))).to(
                torch.int64
            )
            targets.append(target)
        return targets

    @property
    def background(self):
        return (
            np.ones(
                shape=(self.background_size[0], self.background_size[1], 3),
                dtype=np.uint8,
            )
            * 255
        )

    def rectangle(self, x1, y1, x2, y2):
        return np.zeros(shape=((x2 - x1), (y2 - y1), 3), dtype=np.uint8)

    def _get_coordinates(self):
        x_min = int(
            (
                self.background_size[0]
                - self.max_rectangle_size[0]
            )
            * random.random()
        )
        y_min = int(
            (
                self.background_size[1]
                - self.max_rectangle_size[1]
            )
            * random.random()
        )
        return x_min, y_min

    def _paste_rectangles(self, image):
        bboxes = []
        for i in range(self.n_rectangles):
            x_min, y_min = self._get_coordinates()
            x_1 = x_min
            y_1 = y_min
            x_2 = x_min + int(
                self.max_rectangle_size[0] * (0.5 + 0.5 * random.random())
            )
            y_2 = y_min + int(
                self.max_rectangle_size[1] * (0.5 + 0.5 * random.random())
            )
            image[x_1:x_2, y_1:y_2, :] = self.rectangle(x1=x_1, x2=x_2, y1=y_1, y2=y_2)
            bboxes.append(torch.FloatTensor((y_1, x_1, y_2, x_2)))
        bboxes = torch.stack(bboxes, dim=0)
        return image, bboxes

    def make_image(self):
        image = self.background
        image, bboxes = self._paste_rectangles(image=image)
        image = PytorchTypeValidator.coerce_image_to_correct_type(image)
        return image, bboxes

    def __call__(self, num_samples):
        images, bboxes = [], []
        for i in range(num_samples):
            image, boxes = self.make_image()
            images.append(image)
            bboxes.append(boxes)
        targets = self._make_targets_from_bboxes(bboxes)
        return images, targets
