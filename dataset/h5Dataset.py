import os.path
import pickle
import h5py
import numpy
from torch.utils.data import Dataset


class ObjectRecord:
    label_names: list[str]
    objs: list[numpy.ndarray]

    def __init__(self, label_names: list[str], objs: list[numpy.ndarray]):
        self.label_names = label_names
        self.objs = objs

    def dump(self, file_path: str):
        obj = {"label_names": self.label_names, "objs": self.objs}
        pickle.dump(obj, open(file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str):
        j = pickle.load(open(file_path, "rb"))
        return ObjectRecord(j["label_names"], j["objs"])


class H5Dataset(Dataset):
    def __init__(self, path: str):
        h5_path = os.path.join(path, "data.h5")
        obj_record_path = os.path.join(path, "obj_record.pkl")
        self.h5f = h5py.File(h5_path, "r")
        self.images: h5py.Dataset = self.h5f["image"]
        self.obj_record = ObjectRecord.load(obj_record_path)

    def __len__(self):
        return self.images.len()

    def __getitem__(self, idx):
        img = self.images[idx]
        objs = self.obj_record.objs[idx]
        return img, objs

    @staticmethod
    def collate_fn(batch):
        im, label_batch = zip(*batch)  # transposed
        labels = []
        for i, lb in enumerate(label_batch):
            if len(lb) != 0:
                new_label = numpy.empty_like(lb, shape=(len(lb), 6))
                new_label[:, 0] = i
                new_label[:, 1:] = numpy.stack(lb)
                labels.append(new_label)
        return numpy.stack(im, 0), numpy.concatenate(labels, 0)

