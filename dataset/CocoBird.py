import os

import tqdm
from PIL import Image

from . import CocoDataset
from .RawDataset import DataItem


class CocoBird(CocoDataset.CocoDataset):

    def __init__(self, image_path: str, ann_path: str):

        super().__init__(image_path, ann_path, check_channel=False)

        bird_id = self.label_names.index("bird")

        bird_items = []
        for item in tqdm.tqdm(self.items):
            objs = []
            for obj in item.objs:
                if obj[0] == bird_id:
                    objs.append((0, obj[1]))
            if len(objs) > 0:
                img = Image.open(item.img)
                if (hasattr(img, "layers") and img.layers == 3) or img.mode == "RGB":
                    bird_items.append(DataItem(item.img, objs))

        self.items = bird_items
        self.label_names = ["bird"]
    pass


if __name__ == '__main__':
    dataset = CocoBird(r"D:\迅雷下载\train2017", r"D:\迅雷下载\annotations\instances_train2017.json")
    for i in range(len(dataset)):
        dataset.display(i)