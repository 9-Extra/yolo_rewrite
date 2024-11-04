import random

import cv2


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

    def ramdom_sample(self, num: int):
        assert num <= len(self.items)
        samples = random.sample(self.items, num)
        return RawDataset(samples, self.label_names)

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


def delete_all_object(dataset: RawDataset):
    dataset.label_names.clear()
    for item in dataset.items:
        item.objs.clear()


def mix_raw_dataset(datasets: list[RawDataset]) -> RawDataset:
    print("mixing datasets")
    for i, d in enumerate(datasets):
        print(f"dataset {i} num {len(d)} labels: {d.label_names} ")

    categories_map = {}
    id = 0
    for d in datasets:
        for name in d.label_names:
            if name not in categories_map:
                categories_map[name] = id
                id += 1

    final_items = []
    for d in datasets:
        for item in d.items:
            remapped_objs = []
            for i in range(len(item.objs)):
                name = d.label_names[item.objs[i][0]]
                box = item.objs[i][1]
                remapped_objs.append((categories_map[name], box))
            final_items.append(DataItem(item.img, remapped_objs))

    label_names = [""] * len(categories_map)
    for name, id in categories_map.items():
        label_names[id] = name

    return RawDataset(final_items, label_names)


