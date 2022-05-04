import re
import torch
import numpy as np
from typing import Union, Optional

FORMATS = ["COCO", "Pascal_VOC"]


class Bbox:
    """
    Accepts a COCO format bounding box as input
    """

    def __init__(
        self,
        bbox: Optional[Union[torch.tensor, np.ndarray]]=None,
        class_id: Optional[Union[int, torch.tensor]] = None,
        class_name: Optional[str] = None,
        score: Optional[Union[float, torch.tensor]] = None,
    ):
        self.bbox = bbox
        self.class_id = class_id
        self.class_name = class_name
        self.score = score


    @property
    def x_1(self):
        return self.bbox[0]

    @property
    def y_1(self):
        return self.bbox[1]

    @property
    def x_2(self):
        return self.bbox[2]

    @property
    def y_2(self):
         return self.bbox[3]

    @property
    def as_torch(self):
        return torch.FloatTensor([self.x_1, self.y_1, self.x_2, self.y_2])

    @property
    def COCO(self):
        #     return self._COCO

        # @COCO.setter
        # def COCO(self):
        return self.x_1, self.y_1, self.x_2, self.y_2

    @property
    def Pascal_VOC(self):
        #     return self._Pascal_VOC

        # @Pascal_VOC.setter
        # def Pascal_VOC(self):
        return self.x_1, self.y_1, self.x_2 - self.x_1, self.y_2 - self.y_1

    @property
    def mean_leg_length(self):
        #     """Returns Sqrt(H*W) , or the mean length of any leg"""
        #     return self._mean_leg_length

        # @mean_leg_length.setter
        # def mean_leg_length(self):
        return ((self.x_2 - self.x_1) * (self.y_2 - self.y_1)) ** 0.5

    @property
    def aspect_ratio(self):
        """Returns Height / Width"""
        #     return self._aspect_ratio

        # @aspect_ratio.setter
        # def aspect_ratio(self):
        return (self.y_2 - self.y_1) / (self.x_2 - self.x_1)

    @property
    def centroid(self):
        """Returns the centroid in form (X, Y)"""
        #     return self._centroid

        # @centroid.setter
        # def centroid(self):
        x = self.x_1 + (self.x_2 - self.x_1) / 2
        y = self.y_1 + (self.y_2 - self.y_1) / 2
        return x, y
