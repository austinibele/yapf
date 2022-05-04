import numpy as np
import torch
from src.dataset.bbox import Bbox
from src.type_validation.bbox_converter import FORMATS
from typing import List
from src.dataset.target_base import TargetBase


class Target(TargetBase):
    def __init__(self, bboxes: List[Bbox]):
        self.bboxes = bboxes

    @property
    def to_retinanet_target(self):
        temp = {}
        temp["boxes"] = self.bboxes_as_torch(box_format="COCO")
        temp["labels"] = self.labels_as_torch
        temp["names"] = self.names

        target = {}
        for k, v in temp.items():
            if not any(x is None for x in v):
                target[k] = v
        return target

    @classmethod
    def from_retinanet_dict(cls, retinanet_dict):
        label_map = {"boxes": 'bbox', "labels": 'class_id', "names": 'class_name', "scores":"score"}
        bboxes = []
        for i in range(len(tuple(retinanet_dict.values())[0])):
            bbox_dict = {label_map[k]: v[i] for k, v in retinanet_dict.items()}
            bboxes.append(Bbox(**bbox_dict))
        return Target(bboxes=bboxes)
        