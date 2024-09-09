import json
import os
from PIL import Image
import tqdm

from dataset.RawDataset import RawDataset, DataItem


class CocoDataset(RawDataset):
    def __init__(self, img_dir_path: str, ann_path: str, check_channel: bool = True):
        assert os.path.isdir(img_dir_path)
        self.img_dir_path = img_dir_path

        ann_file_json = json.load(open(ann_path, "rb"))
        images = ann_file_json["images"]
        annotations = ann_file_json["annotations"]

        # 重映射类别id
        categories = ann_file_json["categories"]
        label_names = []
        categories_map = {}
        for i, c in enumerate(categories):
            categories_map[c["id"]] = i
            label_names.append(c["name"])

        item_dict = {}
        for img in images:
            item_dict[img["id"]] = DataItem(os.path.join(img_dir_path, img["file_name"]), [])

        print("image num = ", len(item_dict))
        for ann in annotations:
            mapped_id = categories_map[ann["category_id"]]
            item_dict[ann["image_id"]].objs.append((mapped_id, ann["bbox"]))

        if check_channel:
            items = []
            for k, v in tqdm.tqdm(item_dict.items()):
                if len(v.objs) != 0:
                    img = Image.open(v.img)
                    if (hasattr(img, "layers") and img.layers == 3) or img.mode == "RGB":
                        items.append(v)
        else:
            items = list(item_dict.values())
        print("filtered image num = ", len(items))
        super().__init__(items, label_names)


if __name__ == '__main__':
    data = CocoDataset(r"D:\迅雷下载\train2017", r"D:\迅雷下载\annotations\instances_train2017.json", check_channel=False)
    for i in range(len(data.items)):
        data.display(i)
