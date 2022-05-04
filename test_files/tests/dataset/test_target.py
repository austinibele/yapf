from cv2 import _OUTPUT_ARRAY_DEPTH_MASK_16F
import torch
import pytest
from src.dataset.target import Target
from src.dataset.bbox import Bbox


@pytest.fixture
def bboxes():
    bbox1 = Bbox(bbox=torch.tensor([1, 2, 3, 4]), class_id=0)
    bbox2 = Bbox(bbox=torch.tensor([2, 5, 7, 9]), class_id=0)
    return [bbox1, bbox2]


@pytest.fixture
def bboxes_with_names():
    bbox1 = Bbox(bbox=torch.tensor([1, 2, 3, 4]), class_id=0, class_name="object")
    bbox2 = Bbox(bbox=torch.tensor([2, 5, 7, 9]), class_id=0, class_name="object")
    return [bbox1, bbox2]


@pytest.fixture
def target(bboxes):
    return Target(bboxes=bboxes)


@pytest.fixture
def target_with_names(bboxes_with_names):
    return Target(bboxes=bboxes_with_names)


@pytest.fixture
def output():
    return {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]]),
        "labels": torch.tensor([0, 0]),
    }


@pytest.fixture
def output_with_names():
    return {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]]),
        "labels": torch.tensor([0, 0]),
        "names": ["object", "object"],
    }


def test_init(bboxes):
    target = Target(bboxes=bboxes)
    assert target.bboxes == bboxes


def test_boxes_as_torch(bboxes, target):
    bboxes = target.bboxes_as_torch(box_format="COCO")
    assert bboxes.shape == (2, 4)
    assert bboxes.dtype == torch.float32
    assert bboxes[0].tolist() == torch.tensor([1, 2, 3, 4]).to(torch.float32).tolist()
    bboxes = target.bboxes_as_torch(box_format="Pascal_VOC")
    assert bboxes.shape == (2, 4)
    assert bboxes.dtype == torch.float32
    assert bboxes[0].tolist() == torch.tensor([1, 2, 2, 2]).to(torch.float32).tolist()


def test_labels_as_torch(target):
    labels = target.labels_as_torch
    assert labels.dtype == torch.int64
    assert labels.tolist() == torch.tensor([0, 0]).to(torch.int64).tolist()


def test_to_retinanet_target(target, output):
    assert target.to_retinanet_target.keys() == output.keys()
    for k, v in target.to_retinanet_target.items():
        assert torch.isclose(output[k].sum(), v.sum())


def test_to_retinanet_target_with_names(target_with_names, output_with_names):
    assert target_with_names.to_retinanet_target.keys() == output_with_names.keys()
    for k, v in target_with_names.to_retinanet_target.items():
        if k == "names":
            assert output_with_names[k] == v
        else:
            assert torch.isclose(output_with_names[k].sum(), v.sum())

def test_from_retinanet_dict(output_with_names):
    out = Target.from_retinanet_dict(retinanet_dict=output_with_names)
    assert isinstance(out, Target)
    assert out.bboxes[0].bbox.sum() == torch.tensor([1,2,3,4]).sum()
    assert out.bboxes[0].class_id == 0
    assert out.bboxes[0].class_name == "object"
    assert out.bboxes[1].bbox.sum() == torch.tensor([2,5,7,9]).sum()
    assert out.bboxes[1].class_id == 0
    assert out.bboxes[1].class_name == "object"


def test_from_retinanet_dict_no_names(output):
    out = Target.from_retinanet_dict(retinanet_dict=output)
    assert isinstance(out, Target)
    assert out.bboxes[0].bbox.sum() == torch.tensor([1,2,3,4]).sum()
    assert out.bboxes[0].class_id == 0
    assert out.bboxes[1].bbox.sum() == torch.tensor([2,5,7,9]).sum()
    assert out.bboxes[1].class_id == 0

if __name__ == '__main__':
    test_from_retinanet_dict(output_with_names=output_with_names)
    test_from_retinanet_dict_no_names(output=output)