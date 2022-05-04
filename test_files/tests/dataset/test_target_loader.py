import pytest
import torch
from src.dataset.target_loader import TargetLoader

@pytest.fixture
def test_target_path():
    return "tests/dataset/detection_dataset_from_disk_fixtures/target_0001.json"


def test_retinanet_target_from_json(test_target_path):
    target = TargetLoader.retinanet_target_from_json(target_path=test_target_path)
    assert torch.is_tensor(target['boxes'])
    assert target['boxes'].dtype == torch.float32
    assert torch.is_tensor(target['labels'])
    assert target['labels'].dtype == torch.int64

if __name__ == '__main__':
    test_retinanet_target_from_json(test_target_path)