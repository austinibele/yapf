import torch
from src.dataset.bbox import Bbox
from src.type_validation.bbox_converter import FORMATS
from typing import List
from src.dataset.target_base import TargetBase


class Prediction(TargetBase):
    def __init__(
        self,
        bboxes: List[Bbox] = None,
    ):
        self.bboxes = bboxes

    @property
    def to_retinanet_prediction(self):
        temp = {}
        temp["boxes"] = self.bboxes_as_torch(box_format="COCO")
        temp["labels"] = self.labels_as_torch
        temp["names"] = self.names
        temp["scores"] = self.scores_as_torch

        pred = {}
        for k, v in temp.items():
            if not any(x is None for x in v):
                pred[k] = v
        return pred

    @property
    def scores_as_torch(self):
        """Returns labels as a torch int64 tensor"""
        return torch.tensor([bbox.score for bbox in self.bboxes])

    @classmethod
    def from_retinanet_dict(cls, retinanet_dict):
        label_map = {"boxes": 'bbox', "labels": 'class_id', "names": 'class_name', "scores":"score"}
        bboxes = []
        for i in range(len(tuple(retinanet_dict.values())[0])):
            bbox_dict = {label_map[k]: v[i] for k, v in retinanet_dict.items()}
            bboxes.append(Bbox(**bbox_dict))
        return Prediction(bboxes=bboxes)
        