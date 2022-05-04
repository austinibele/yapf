import torch
import pytest
from src.dataset.detection.detection_dataset_from_disk import DetectionDatasetFromDisk
from src.dataset.target_loader import TargetLoader
from src.dataset.image_loader import ImageLoader


@pytest.fixture
def image_paths():
    return [
        "tests/dataset/detection_dataset_from_disk_fixtures/img_0001.jpeg",
        "tests/dataset/detection_dataset_from_disk_fixtures/img_0002.jpeg",
    ]


@pytest.fixture
def target_paths():

    return [
        "tests/dataset/detection_dataset_from_disk_fixtures/target_0001.json",
        "tests/dataset/detection_dataset_from_disk_fixtures/target_0002.json",
    ]


def test_init(image_paths, target_paths):
    dataset = DetectionDatasetFromDisk(
        image_paths=image_paths,
        target_paths=target_paths,
        image_loader=ImageLoader.load_image_to_torch,
        target_loader=TargetLoader.retinanet_target_from_json,
    )
    assert len(dataset.image_paths) == 2
    assert len(dataset.target_paths) == 2


def test_getitem(image_paths, target_paths):
    dataset = DetectionDatasetFromDisk(
        image_paths=image_paths,
        target_paths=target_paths,
        image_loader=ImageLoader.load_image_to_torch,
        target_loader=TargetLoader.retinanet_target_from_json,
    )
    image, target = dataset[0]
    assert torch.is_tensor(image)
    assert "boxes" and "labels" in target.keys()
    assert torch.is_tensor(target["boxes"])


if __name__ == "__main__":
    test_init(image_paths, target_paths)
    test_getitem()
