import os

from dataset.RawDataset import RawDataset, DataItem
import xml.dom.minidom
import tqdm


def parse_xml(xml_path: str):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    filename = root.getElementsByTagName("filename")[0].childNodes[0].data
    objs = root.getElementsByTagName("object")
    res = []
    for obj in objs:
        bndbox = obj.getElementsByTagName("bndbox")[0]
        x = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
        y = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
        x2 = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
        y2 = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
        res.append((0, [x, y, x2 - x, y2 - y]))
    return filename, res


class DroneDataset(RawDataset):
    def __init__(self, root: str):
        img_dir = os.path.join(root, "Drone_TrainSet")
        ann_dir = os.path.join(root, "Drone_TrainSet_XMLs")

        self.items: list[DataItem] = []
        for xml in tqdm.tqdm(os.listdir(ann_dir)):
            filename, res = parse_xml(os.path.join(ann_dir, xml))
            self.items.append(DataItem(os.path.join(img_dir, filename), res))
        pass

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def get_label_names(self):
        return {0: "drone"}


if __name__ == '__main__':
    dataset = DroneDataset("G:/datasets/DroneTrainDataset")
    for i in range(len(dataset)):
        dataset.display(i)
    pass