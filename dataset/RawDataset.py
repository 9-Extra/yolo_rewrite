import os

import cv2


class DataItem:
    img: str  # file_path
    objs: list[tuple[int, list]]  # class label, [x1, y1, x2, y2]

    def __init__(self, img: str, objs: list[tuple[int, list]]):
        self.img = img
        self.objs = objs


class RawDataset:
    def __getitem__(self, index) -> DataItem:
        raise NotImplementedError()

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        item = self.__getitem__(self._iter)
        self._iter += 1
        return item

    def __len__(self):
        raise NotImplementedError()

    def get_label_names(self):
        raise NotImplementedError()

    def display(self, index: int):
        item = self[index]
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
