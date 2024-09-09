import os

from . import RawDataset


class BirdVSDroneBird(RawDataset.RawDataset):
    def __init__(self, image_dir: str, split: str = "train"):
        assert os.path.isdir(image_dir)
        items = []

        file_list = os.listdir(image_dir)
        file_list.sort()

        if split == "train":
            file_list = file_list[:int(len(file_list) * 0.8)]
        elif split == "val":
            file_list = file_list[int(len(file_list) * 0.8):]
        else:
            raise ValueError("split must be train or val")

        for img in file_list:
            items.append(RawDataset.DataItem(os.path.join(image_dir, img), []))

        super().__init__(items, [])
