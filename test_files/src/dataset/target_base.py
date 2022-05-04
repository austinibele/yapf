import numpy as np
import torch
from src.dataset.bbox import Bbox
from src.type_validation.bbox_converter import FORMATS
from typing import List


class TargetBase:
    def __init__(self, bboxes: List[Bbox]):
        self.bboxes = bboxes

    @property
    def to_retinanet_target(self):
        raise NotImplementedError

    def bboxes_as_torch(self, box_format):
        """Returns bboxes as a [N, 4] torch float32 tensor"""
        bboxes_list = self._get_bboxes(box_format)
        bboxes_list = [torch.tensor(bbox).to(torch.float32) for bbox in bboxes_list]
        return torch.stack(bboxes_list, dim=0)

    @property
    def labels_as_torch(self):
        """Returns labels as a torch int64 tensor"""
        labels = [bbox.class_id for bbox in self.bboxes]
        return torch.tensor(labels).to(torch.int64)

    @property
    def names(self):
        """Returns labels as a torch int64 tensor"""
        return [bbox.class_name for bbox in self.bboxes]

    def _get_bboxes(self, box_format):
        assert box_format in FORMATS
        bboxes_list = []
        if box_format == "COCO":
            for bbox in self.bboxes:
                bboxes_list.append(bbox.COCO)
        if box_format == "Pascal_VOC":
            for bbox in self.bboxes:
                bboxes_list.append(bbox.Pascal_VOC)
        return bboxes_list
