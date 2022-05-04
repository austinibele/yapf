import torch
import pytest
from src.dataset.image_loader import ImageLoader


@pytest.fixture
def test_image_path():
    return "tests/dataset/detection_dataset_from_disk_fixtures/img_0001.jpeg"


def test_load_image_to_torch(test_image_path):
    img = ImageLoader.load_image_to_torch(image_path=test_image_path)
    assert torch.is_tensor(img)
