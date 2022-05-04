from copy import deepcopy
from typing import Tuple
from torch.utils.data import Dataset, Subset

class DatasetSplitter:

    @classmethod
    def ordered_split(cls,
        dataset, val_size, train_transform, val_transform
    ) -> Tuple[Dataset, Dataset]:
        def test():
            pass
        
        def test2(key: str=None, val: int=5):
            pass
        
        dataset_length = int(dataset.__len__())
        split_idx = int(dataset_length * (1 - val_size))
        train_indices = range(0, split_idx)
        test_indices = range(split_idx, dataset_length)
        train = Subset(dataset, train_indices)
        train.dataset.transform = train_transform
        val = Subset(deepcopy(dataset), test_indices)
        val.dataset.transform = val_transform
        return train, val
    

