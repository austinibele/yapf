from src.dataset.detection.detection_dataset import DetectionDataset
from src.dataset.detection.square_dataset_generator import SquareDatasetGenerator
from src.dataset.detection.long_rectangle_dataset_generator import (
    LongRectangleDatasetGenerator,
)
from src.dataset.detection.dot_dataset_generator import DotDatasetGenerator
from src.type_validation.pytorch_type_validator import PytorchTypeValidator


class DatasetBuilder:
    @classmethod
    def create_square_dataset(cls, num_samples):
        square_image_generator = SquareDatasetGenerator()
        images, targets = square_image_generator(num_samples=num_samples)

        if not PytorchTypeValidator.images_are_correct_type(images):
            images = PytorchTypeValidator.ensure_images_are_correct_type(images)
            assert PytorchTypeValidator.images_are_correct_type(images)

        # TODO ensure targets are correct type

        dataset = DetectionDataset(images=images, targets=targets)
        return dataset

    @classmethod
    def create_long_rectangle_dataset(cls, num_samples):
        long_rectange_image_generator = LongRectangleDatasetGenerator()
        images, targets = long_rectange_image_generator(num_samples=num_samples)

        if not PytorchTypeValidator.images_are_correct_type(images):
            images = PytorchTypeValidator.ensure_images_are_correct_type(images)
            assert PytorchTypeValidator.images_are_correct_type(images)

        # TODO ensure targets are correct type

        dataset = DetectionDataset(images=images, targets=targets)
        return dataset

    @classmethod
    def create_dot_dataset(cls, num_samples):
        dot_image_generator = DotDatasetGenerator()
        images, targets = dot_image_generator(num_samples=num_samples)

        if not PytorchTypeValidator.images_are_correct_type(images):
            images = PytorchTypeValidator.ensure_images_are_correct_type(images)
            assert PytorchTypeValidator.images_are_correct_type(images)

        # TODO ensure targets are correct type

        dataset = DetectionDataset(images=images, targets=targets)
        return dataset
