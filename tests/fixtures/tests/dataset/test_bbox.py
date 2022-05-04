import torch
import numpy as np
from copy import deepcopy
import pytest
from src.dataset.bbox import Bbox


@pytest.fixture
def box_tensor():
    return torch.tensor([104, 61, 476, 257])


@pytest.fixture
def box_np(box_tensor):
    return deepcopy(box_tensor).numpy()


def test_init_torch(box_tensor):
    bbox = Bbox(bbox=box_tensor)
    assert bbox.x_1 == 104
    assert bbox.y_1 == 61
    assert bbox.x_2 == 476
    assert bbox.y_2 == 257


def test_init_np(box_np):
    bbox = Bbox(box_np)
    assert bbox.x_1 == 104
    assert bbox.y_1 == 61
    assert bbox.x_2 == 476
    assert bbox.y_2 == 257


def test_init_optional_args(box_tensor):
    bbox = Bbox(bbox=box_tensor, class_id=0, class_name="object", score=0.4)
    assert bbox.class_id == 0
    assert bbox.class_name == "object"
    assert bbox.score == 0.4


def test_COCO(box_tensor):
    bbox = Bbox(box_tensor)
    assert (bbox.COCO, (104, 61, 476, 257))


def test_Pascal_VOC(box_tensor):
    bbox = Bbox(box_tensor)
    assert (bbox.Pascal_VOC, (104, 61, (476 - 104), (257 - 61)))


def test_mean_leg_length(box_tensor):
    bbox = Bbox(box_tensor)
    assert round(float(bbox.mean_leg_length), 2) == round(
        float(((476 - 104) * (257 - 61)) ** 0.5), 2
    )


def test_aspect_ratio(box_tensor):
    bbox = Bbox(box_tensor)
    assert bbox.aspect_ratio == (257 - 61) / (476 - 104)


def test_centroid():
    bbox = Bbox(torch.tensor([5, 5, 10, 10]))
    assert bbox.centroid == (7.5, 7.5)
