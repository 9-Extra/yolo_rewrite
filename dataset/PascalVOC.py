import os

from dataset.RawDataset import RawDataset, DataItem
from typing import Literal
import xml.dom.minidom
import tqdm
import random

class _Parser:
    
    categories_map: dict[str, int]
    _next_id: int
    
    def __init__(self):
        self._next_id = -1
        self.categories_map = {}
        pass
    
    def _get_class_id(self, name: str):
        if name in self.categories_map:
            return self.categories_map[name]
        else:
            self._next_id += 1
            self.categories_map[name] = self._next_id
            return self._next_id
        
    def get_label_names(self):
        return list(self.categories_map.keys()) # 利用dict的有序性
            
    def parse_xml(self, xml_path: str) -> tuple[str, list]:
        dom = xml.dom.minidom.parse(xml_path)
        root = dom.documentElement
        filename = root.getElementsByTagName("filename")[0].childNodes[0].data
        objs = root.getElementsByTagName("object")
        res = []
        for obj in objs:
            id = self._get_class_id(obj.getElementsByTagName("name")[0].childNodes[0].data)
            bndbox = obj.getElementsByTagName("bndbox")[0]
            
            try:
                x = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data, 10)
                y = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data, 10)
                x2 = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data, 10)
                y2 = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data, 10)
            except ValueError:
                x = int(float(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data))
                y = int(float(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data))
                x2 = int(float(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data))
                y2 = int(float(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data))
                
            res.append((id, [x, y, x2 - x, y2 - y]))
        return filename, res


class PascalVOC(RawDataset):
    items: list[DataItem]

    def __init__(self, root: str, split: Literal["train", "val", "all"] = "train"):
        img_dir = os.path.join(root, "JPEGImages")
        ann_dir = os.path.join(root, "Annotations")
        
        parse = _Parser()

        self.items: list[DataItem] = []
        # 为了保证在不同系统上分割的训练集是一致的，故排序后再用固定随机数乱序
        xml_files = sorted(os.listdir(ann_dir))
        random.seed(42)
        random.shuffle(xml_files)
        for xml in tqdm.tqdm(xml_files):
            filename, res = parse.parse_xml(os.path.join(ann_dir, xml))
            if len(res) != 0:
                self.items.append(DataItem(os.path.join(img_dir, filename), res))
        pass
        
        sp = int(len(self.items) * 0.8)
        if split == "train":
            self.items = self.items[:sp]
        elif split == "val":
            self.items = self.items[sp:]
        elif split == "all":
            pass
        else:
            raise ValueError("split must be train or val or all")

        super().__init__(self.items, parse.get_label_names())