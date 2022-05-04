import torch
import pytest
from src.dataset.prediction import Prediction
from src.dataset.bbox import Bbox
from tests.dataset.test_target import output_with_names


@pytest.fixture
def bboxes():
    bbox1 = Bbox(bbox=torch.tensor([1, 2, 3, 4]), class_id=0, score=0.23)
    bbox2 = Bbox(bbox=torch.tensor([2, 5, 7, 9]), class_id=0, score=0.3)
    return [bbox1, bbox2]


@pytest.fixture
def bboxes_with_names():
    bbox1 = Bbox(
        bbox=torch.tensor([1, 2, 3, 4]), class_id=0, class_name="object", score=0.23
    )
    bbox2 = Bbox(
        bbox=torch.tensor([2, 5, 7, 9]), class_id=0, class_name="object", score=0.3
    )
    return [bbox1, bbox2]


@pytest.fixture
def prediction(bboxes):
    return Prediction(bboxes=bboxes)


@pytest.fixture
def prediction_with_names(bboxes_with_names):
    return Prediction(bboxes=bboxes_with_names)


@pytest.fixture
def output():
    return {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]]),
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.2300, 0.3000]),
    }


@pytest.fixture
def output_with_names():
    return {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 5.0, 7.0, 9.0]]),
        "labels": torch.tensor([0, 0]),
        "names": ["object", "object"],
        "scores": torch.tensor([0.2300, 0.3000]),
    }


def test_init(bboxes):
    prediction = Prediction(bboxes=bboxes)
    assert prediction.bboxes == bboxes


def test_boxes_as_torch(bboxes, prediction):
    bboxes = prediction.bboxes_as_torch(box_format="COCO")
    assert bboxes.shape == (2, 4)
    assert bboxes.dtype == torch.float32
    assert bboxes[0].tolist() == torch.tensor([1, 2, 3, 4]).to(torch.float32).tolist()
    bboxes = prediction.bboxes_as_torch(box_format="Pascal_VOC")
    assert bboxes.shape == (2, 4)
    assert bboxes.dtype == torch.float32
    assert bboxes[0].tolist() == torch.tensor([1, 2, 2, 2]).to(torch.float32).tolist()


def test_labels_as_torch(prediction):
    labels = prediction.labels_as_torch
    assert labels.dtype == torch.int64
    assert labels.tolist() == torch.tensor([0, 0]).to(torch.int64).tolist()


def test_to_retinanet_prediction(prediction, output):
    assert prediction.to_retinanet_prediction.keys() == output.keys()
    for k, v in prediction.to_retinanet_prediction.items():
        assert torch.isclose(output[k].sum(), v.sum())


def test_to_retinanet_prediction(prediction_with_names, output_with_names):
    assert (
        prediction_with_names.to_retinanet_prediction.keys() == output_with_names.keys()
    )
    for k, v in prediction_with_names.to_retinanet_prediction.items():
        if k == "names":
            assert output_with_names[k] == v
        else:
            assert torch.isclose(output_with_names[k].sum(), v.sum())



def test_from_retinanet_dict(output_with_names):
    out = Prediction.from_retinanet_dict(retinanet_dict=output_with_names)
    assert isinstance(out, Prediction)
    assert out.bboxes[0].bbox.sum() == torch.tensor([1,2,3,4]).sum()
    assert out.bboxes[0].class_id == 0
    assert out.bboxes[0].class_name == "object"
    assert out.bboxes[0].score == 0.23
    assert out.bboxes[1].bbox.sum() == torch.tensor([2,5,7,9]).sum()
    assert out.bboxes[1].class_id == 0
    assert out.bboxes[1].class_name == "object"
    assert out.bboxes[1].score == 0.30

def test_from_retinanet_dict_no_names(output):
    out = Prediction.from_retinanet_dict(retinanet_dict=output)
    assert isinstance(out, Prediction)
    assert out.bboxes[0].bbox.sum() == torch.tensor([1,2,3,4]).sum()
    assert out.bboxes[0].class_id == 0
    assert out.bboxes[0].score == 0.23
    assert out.bboxes[1].bbox.sum() == torch.tensor([2,5,7,9]).sum()
    assert out.bboxes[1].class_id == 0
    assert out.bboxes[1].score == 0.30
