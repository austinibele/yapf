import json
import torch
from src.utils.utils import Utils

class TargetLoader:

    @classmethod
    def retinanet_target_from_json(cls, target_path):
        data = Utils.read_from_json(target_path)
        target = {}
        target['boxes'] = torch.FloatTensor(data['boxes'])
        target['labels'] = torch.tensor(data['labels']).to(torch.int64)
        if "names" in data.keys():
            target["names"] = data['names']
        return target