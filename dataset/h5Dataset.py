import cv2
import h5py
import numpy
from torch.utils.data import Dataset


class H5DatasetYolo(Dataset):
    def __init__(self, path: str):
        # images: h5py.Dataset = h5f.create_dataset("image", (image_count, 3, *target_size), dtype=numpy.uint8)
        # bbox_idx: h5py.Dataset = h5f.create_dataset("bbox_idx", (image_count, 2), dtype=numpy.uint32)
        # sub_images: h5py.Dataset = h5f.create_dataset("sub_image", (bbox_count, 3, *sub_image_size), dtype=numpy.uint8)
        # bbox: h5py.Dataset = h5f.create_dataset("bbox", (bbox_count, 5), dtype=numpy.float32)

        self.h5f = h5py.File(path, "r")
        self.images: h5py.Dataset = self.h5f["image"]
        self.slices = [slice(x1, x2) for x1, x2 in self.h5f["bbox_idx"][:]]
        self.bboxes = self.h5f["bbox"][:]
        # self.sub_images = [sub_images[s] for s in self.slices]

    def __len__(self):
        return self.images.len()

    def __getitem__(self, idx):
        img = self.images[idx]
        s = self.slices[idx]
        bboxes = self.bboxes[s]
        return img, bboxes

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


class H5DatasetFindMe(Dataset):
    def __init__(self, path: str):
        # images: h5py.Dataset = h5f.create_dataset("image", (image_count, 3, *target_size), dtype=numpy.uint8)
        # bbox_idx: h5py.Dataset = h5f.create_dataset("bbox_idx", (image_count, 2), dtype=numpy.uint32)
        # sub_images: h5py.Dataset = h5f.create_dataset("sub_image", (bbox_count, 3, *sub_image_size), dtype=numpy.uint8)
        # bbox: h5py.Dataset = h5f.create_dataset("bbox", (bbox_count, 5), dtype=numpy.float32)

        self.h5f = h5py.File(path, "r")
        self.images: h5py.Dataset = self.h5f["image"]
        self.slices = [slice(x1, x2) for x1, x2 in self.h5f["bbox_idx"][:]]
        self.bboxes = self.h5f["bbox"][:]
        self.sub_images = self.h5f["sub_image"]
        # self.sub_images = [sub_images[s] for s in self.slices]

    def __len__(self):
        return self.images.len()

    def __getitem__(self, idx):
        img = self.images[idx]
        s = self.slices[idx]
        sub_images = self.sub_images[s]
        bboxes = self.bboxes[s]
        return img, sub_images, bboxes

    @staticmethod
    def collate_fn(batch: list[tuple[numpy.ndarray, numpy.ndarray]]):
        images = []
        sub_images_list = []  # 目标小图
        target: dict[int, list[numpy.ndarray] | numpy.ndarray] = {}  # cls到bbox的映射
        sub_images_cls = []

        for i, (im, sub_images, label) in enumerate(batch):
            images.append(im)
            sub_images_list.append(sub_images)

            for l in label:
                cls = int(l[0])
                sub_images_cls.append(cls)
                bbox_list = target.setdefault(cls, [])
                bbox_list.append(numpy.array([i, *l[1:]], dtype=numpy.float32))

        batched_sub_images = numpy.concatenate(sub_images_list)
        assert len(sub_images_cls) == batched_sub_images.shape[0]

        for cls in target.keys():
            target[cls] = numpy.stack(target[cls])

        stacked_target = [target[cls] for cls in sub_images_cls]

        return numpy.stack(images), batched_sub_images, stacked_target
