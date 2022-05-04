import unittest
import torch
from src.visualization.small_object_plotter import SmallObjectPlotter
from src.dataset.detection.square_dataset_generator import SquareDatasetGenerator
from src.dataset.detection.detection_dataset import DetectionDataset


class TestSquareDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.sdg = SquareDatasetGenerator()
        self.plotter = SmallObjectPlotter()

    def test_box_location(self):
        images, targets = self.sdg(num_samples=2)
        assert torch.is_tensor(images[0])
        assert images[0].shape[0] == 3

        ds = DetectionDataset(images=images, targets=targets)
        image, target = ds[0]
        assert torch.is_tensor(image)
        assert image.shape[0] == 3
        _ = self.plotter.draw(image=image, prediction=target)


if __name__ == "__main__":
    test = TestSquareDatasetGenerator()
    test.setUp()
    test.test_box_location()
