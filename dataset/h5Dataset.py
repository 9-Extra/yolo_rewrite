import os
import h5py
import numpy
from torch.utils.data import Dataset


class H5DatasetYolo(Dataset):
    def __init__(self, path: str):
        assert os.path.isfile(path)
        
        self.h5f = h5py.File(path, "r")
        self.images: h5py.Dataset = self.h5f["image"]
        self.slices = [slice(x1, x2) for x1, x2 in self.h5f["bbox_idx"][:]]
        self.bboxes = self.h5f["bbox"][:]

    def __len__(self):
        return self.images.len()

    def __getitem__(self, idx):
        img = self.images[idx]
        s = self.slices[idx]
        bboxes = self.bboxes[s]
        return img, bboxes

    def get_label_names(self):
        return list(str(n, encoding="utf-8") for n in self.h5f["obj_name"])

    @staticmethod
    def collate_fn(batch):
        im, bboxes_batch = zip(*batch)  # transposed
        labels = []
        for i, bbox in enumerate(bboxes_batch):
            if len(bbox) != 0:
                indexed_bbox = numpy.empty_like(bbox, shape=(len(bbox), 6))
                indexed_bbox[:, 0] = i
                indexed_bbox[:, 1:] = numpy.stack(bbox)
                labels.append(indexed_bbox)
        labels = numpy.concatenate(labels, 0) if len(labels) else numpy.zeros((0, 6), dtype=numpy.float32)
        return numpy.stack(im, 0), labels


def get_label_names_from_file(path: str):
    with h5py.File(path, "r") as h5:
        obj_names = list(str(n, encoding="utf-8") for n in h5["obj_name"])

    return obj_names
