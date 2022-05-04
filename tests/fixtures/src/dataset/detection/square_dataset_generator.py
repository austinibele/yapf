import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter
from src.type_validation.pytorch_type_validator import PytorchTypeValidator


class SquareDatasetGenerator:
    def __init__(self, background_size=(128, 128), n_squares=2, n_hollow_squares=2):
        self.background_size = background_size
        self.max_square_size = (20, 20)
        self.max_hollow_square_size = (20, 20)
        self.n_squares = n_squares
        self.n_hollow_squares = n_hollow_squares

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

    def square(self, x1, y1, x2, y2):
        return np.zeros(shape=((x2 - x1), (y2 - y1), 3), dtype=np.uint8)

    def _get_coordinates(self):
        x_min = int(
            (
                self.background_size[0]
                - max(self.max_hollow_square_size[0], self.max_square_size[0])
            )
            * random.random()
        )
        y_min = int(
            (
                self.background_size[1]
                - max(self.max_hollow_square_size[1], self.max_square_size[1])
            )
            * random.random()
        )
        return x_min, y_min

    def _paste_squares(self, image):
        bboxes = []
        for i in range(self.n_squares):
            x_min, y_min = self._get_coordinates()
            x_1 = x_min
            y_1 = y_min
            x_2 = x_min + int(self.max_square_size[0] * (0.5 + 0.5 * random.random()))
            y_2 = y_min + int(self.max_square_size[1] * (0.5 + 0.5 * random.random()))
            image[x_1:x_2, y_1:y_2, :] = self.square(x1=x_1, x2=x_2, y1=y_1, y2=y_2)
            bboxes.append(torch.FloatTensor((y_1, x_1, y_2, x_2)))
        bboxes = torch.stack(bboxes, dim=0)
        return image, bboxes

    def _paste_hollow_squares(self, image):
        for i in range(self.n_hollow_squares):
            x_min, y_min = self._get_coordinates()

            x_1 = x_min
            y_1 = y_min
            x_2 = x_min + int(
                self.max_hollow_square_size[0] * (0.5 + 0.5 * random.random())
            )
            y_2 = y_min + int(
                self.max_hollow_square_size[1] * (0.5 + 0.5 * random.random())
            )

            hollow_square = np.zeros(
                shape=(
                    (x_2 - x_1),
                    (y_2 - y_1),
                    3,
                ),
                dtype=np.uint8,
            )
            hollow_square[
                hollow_square.shape[0] // 2 - 2 : hollow_square.shape[0] // 2 + 3,
                hollow_square.shape[1] // 2 - 2 : hollow_square.shape[1] // 2 + 3,
                :,
            ] = (
                np.ones(shape=(5, 5, 3)) * 255
            )
            blurred_hollow_square = gaussian_filter(hollow_square, sigma=0)
            image[
                x_min : x_min + (x_2 - x_1),
                y_min : y_min + (y_2 - y_1),
                :,
            ] = blurred_hollow_square
        return image

    def make_image(self):
        image = self.background
        image, bboxes = self._paste_squares(image=image)
        image = self._paste_hollow_squares(image=image)
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
