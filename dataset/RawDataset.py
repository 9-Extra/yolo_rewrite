import random
from typing import Iterable

import cv2
import pandas


class DataItem:
    img: str  # file_path
    objs: list[tuple[int, list]]  # class label, [x1, y1, x2, y2]

    def __init__(self, img: str, objs: list[tuple[int, list]]):
        self.img = img
        self.objs = objs


class RawDataset:
    items: list[DataItem]
    label_names: list[str]

    def __init__(self, items: list[DataItem], label_names: list[str]):
        self.items = items
        self.label_names = label_names

    def __getitem__(self, index) -> DataItem:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def ramdom_sample(self, num: int, ramdom_seed: object=None):
        assert num <= len(self.items)
        if ramdom_seed is not None:
            random.seed(ramdom_seed)
        
        samples = random.sample(self.items, num)
        return RawDataset(samples, self.label_names)

    def delete_object(self, *obj_names: str) -> "RawDataset":
        obj_names: set = set(obj_names)
        if len(obj_names) == 0:
            return
        assert obj_names.issubset(set(self.label_names))
        
        ori_id_map = list(range(len(self.label_names)))    
        new_label_names = self.label_names[:] # copy
        for d in obj_names:
            i = self.label_names.index(d)
            ori_id_map[i] = -1
            del new_label_names[i]
        
        # 重分配id
        id = 0
        for i in range(len(ori_id_map)):
            if ori_id_map[i] != -1:
                ori_id_map[i] = id
                id += 1

        new_items: list[DataItem] = []
        for item in self.items:
            new_objs: list[tuple[int, list]] = []
            for obj in item.objs:
                new_id = ori_id_map[obj[0]]
                if new_id != -1:
                    new_objs.append((new_id, obj[1]))
            if len(new_objs) != 0:
                new_items.append(DataItem(item.img, new_objs))
        
        return RawDataset(new_items, new_label_names)
                    
    def get_label_names(self) -> list[str]:
        return self.label_names

    def display(self, index: int):
        item = self.items[index]
        img = cv2.imread(item.img)
        for obj in item.objs:
            x, y, width, height = [int(x) for x in obj[1]]
            x1, y1, x2, y2 = x, y, x + width, y + height
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.putText(img, self.get_label_names()[obj[0]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                              2,
                              cv2.LINE_AA)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    def summary(self):
        print(f"共{len(self.items)}张图像")
        counter = [0 for _ in range(len(self.label_names))]
        for item in self.items:
            for obj in item.objs:
                counter[obj[0]] += 1
        summary = pandas.DataFrame({"label": self.label_names, "id": range(len(self.label_names)), "count": counter})
        print(summary.to_string(index=False))    


def delete_all_object(dataset: RawDataset):
    dataset.label_names.clear()
    for item in dataset.items:
        item.objs.clear()


def mix_raw_dataset(*datasets: RawDataset, label_names: list[str]=None) -> RawDataset:
    print("mixing datasets")
    for i, d in enumerate(datasets):
        print(f"dataset {i} num={len(d)} \nlabels: {d.label_names} ")

    if label_names is None:
        # 从数据集生成新label
        categories_map = {}
        id = 0
        for d in datasets:
            for name in d.label_names:
                if name not in categories_map:
                    categories_map[name] = id
                    id += 1
                    
        label_names = list(categories_map.keys())
    else:
        # 使用给定的label_names
        categories_map = {label: i for i, label in enumerate(label_names)}    

    final_items = []
    for d in datasets:
        for item in d.items:
            remapped_objs = []
            for i in range(len(item.objs)):
                name = d.label_names[item.objs[i][0]]
                box = item.objs[i][1]
                if name in categories_map: 
                    remapped_objs.append((categories_map[name], box))
            # if len(remapped_objs) != 0: 
            final_items.append(DataItem(item.img, remapped_objs))

    return RawDataset(final_items, label_names)


