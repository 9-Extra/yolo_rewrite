import functools
import os
from pathlib import Path
from typing import Callable

import cv2
import h5py
import numpy
import tqdm

from dataset.CocoDataset import CocoDataset
from dataset.PascalVOC import PascalVOC
from dataset.DroneDataset import DroneDataset
from dataset.DroneDataset import DroneTestDataset
from dataset.RawDataset import RawDataset, mix_raw_dataset, delete_all_object
from dataset.BirdVSDroneBird import BirdVSDroneBird
from dataset.CocoBird import CocoBird

from config import Config


def letterbox_fixed_size(
    im, new_shape: tuple[int, int], color=(114, 114, 114), scaleup=False
):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    height, width = im.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(new_shape[0] / height, new_shape[1] / width)
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = round(width * r), round(height * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    ratio = new_unpad[0] / width, new_unpad[1] / height

    if (width, height) != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh // 2, dh - (dh // 2)  # divide padding into 2 parts
    left, right = dw // 2, dw - (dw // 2)
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (top, left)


def letterbox_stride(im: numpy.ndarray, stride: int = 32, color=(114, 114, 114)):
    height, width, channel = im.shape
    target_width = width + stride - 1 - (width - 1) % stride
    target_height = height + stride - 1 - (height - 1) % stride
    if target_width == width and target_height == height:
        return im  # skip

    result = numpy.full_like(
        im, color, shape=(target_height, target_width, channel), subok=False
    )
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    result[top : top + height, left : left + width] = im

    return result


def crop_image(img: numpy.ndarray, bbox, size: tuple[int, int]):
    x, y, w, h = bbox
    img = img[y : y + h, x : x + w, :]
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return img


def display(img, objs, label_names):
    h, w, _ = img.shape
    for obj in objs:
        cls, x, y, width, height = obj
        cls = int(cls)
        x1 = int((x - width / 2) * w)
        y1 = int((y - height / 2) * h)
        x2 = int((x + width / 2) * w)
        y2 = int((y + height / 2) * h)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(
            img,
            label_names[cls],
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("image", img)
    cv2.waitKey(0)


def process_data(origin_img: str, objs: list, target_size: tuple[int, int]):
    origin_img: numpy.ndarray = cv2.imread(origin_img)
    img, ratio, (top, left) = letterbox_fixed_size(origin_img, target_size)
    mapped_objs = numpy.empty((len(objs), 5), dtype=numpy.float32)
    for i, obj in enumerate(objs):
        x, y, width, height = obj[1]
        # center x, center y, width, height
        x = ((x + width / 2) * ratio[0] + left) / target_size[0]
        y = ((y + height / 2) * ratio[1] + top) / target_size[1]
        width *= ratio[0] / target_size[0]
        height *= ratio[1] / target_size[1]

        mapped_objs[i] = [obj[0], x, y, width, height]

    cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL, img)
    img = img.transpose(2, 0, 1)

    return img, mapped_objs


def raw_dataset2h5(
    dist: str, data: Callable[[], RawDataset], skip_if_exist: bool = True
):
    if skip_if_exist and os.path.isfile(dist):
        with h5py.File(dist, "r") as h5f:
            if "complete" in h5f.attrs:
                return

    os.makedirs(os.path.dirname(dist), exist_ok=True)
    target_size = (640, 640)
    
    data = data() # get

    # data = data.ramdom_sample(1000)

    process = functools.partial(process_data, target_size=target_size)

    image_count = len(data)
    bbox_count = sum(len(d.objs) for d in data)
    with h5py.File(dist, "w") as h5f:
        images: h5py.Dataset = h5f.create_dataset(
            "image", (image_count, 3, *target_size), dtype=numpy.uint8
        )
        bbox_idx: h5py.Dataset = h5f.create_dataset(
            "bbox_idx", (image_count, 2), dtype=numpy.uint32
        )
        bbox: h5py.Dataset = h5f.create_dataset(
            "bbox", (bbox_count, 5), dtype=numpy.float32
        )

        h5f.create_dataset(
            "obj_name", data=data.get_label_names(), dtype=h5py.special_dtype(vlen=str)
        )

        bbox_idx_offset = 0
        for i, d in enumerate(tqdm.tqdm(data, total=image_count, desc="预处理图像")):
            img, mapped_objs = process(d.img, d.objs)
            bbox_num = mapped_objs.shape[0]
            assert bbox_num != 0

            images.write_direct(numpy.ascontiguousarray(img), dest_sel=i)

            slice_ = slice(bbox_idx_offset, bbox_idx_offset + bbox_num)
            bbox_idx.write_direct(
                numpy.array([slice_.start, slice_.stop], dtype=numpy.uint32), dest_sel=i
            )
            bbox.write_direct(mapped_objs, dest_sel=slice_)
            bbox_idx_offset += bbox_num

        h5f.attrs["complete"] = 1  # 标记为完成

    pass


if __name__ == "__main__":
    config = Config.from_profile("./profiles/pure_drone.toml")
    # datasets path
    dataset_path_drone_train: Path = Path(r"/mnt/panpan/datasets/DroneTrainDataset/")
    dataset_path_drone_test: Path = Path(r"/mnt/panpan/datasets/DroneTestDataset/")
    dataset_path_coco: Path = Path(r"/mnt/panpan/datasets/coco2017")
    dataset_path_drone_vs_bird: Path = Path(r"/mnt/panpan/datasets/BirdVsDrone")
    dataset_path_pascal_vos: Path = Path(r"/mnt/panpan/datasets/VOC2012")

    os.makedirs(config.cache_path, exist_ok=True)

    @functools.cache
    def get_drone_train():
        drone_train = DroneDataset(dataset_path_drone_train, split="train")
        drone_train.summary()
        return drone_train

    # raw_dataset2h5(config.cache_path / "train_pure_drone.h5", get_drone_train)

    @functools.cache
    def get_coco_train():
        coco_train = mix_raw_dataset(
            get_drone_train(),
            CocoDataset(
                dataset_path_coco / "train2017",
                dataset_path_coco / "annotations_trainval2017/instances_train2017.json",
            )
            .delete_object("bird", "person")
            .ramdom_sample(len(get_drone_train()), 42),
        )
        coco_train.summary()
        open(config.run_path / "train_label.txt", "w").write(
            repr(coco_train.get_label_names())
        )
        return coco_train

    raw_dataset2h5(config.cache_path / "train_mixed.h5", get_coco_train)
    
    @functools.cache
    def get_voc_train():
        print("VOC训练集")
        voc_train = PascalVOC(dataset_path_pascal_vos)
        voc_train.summary()
        return voc_train

    raw_dataset2h5(config.cache_path / "train_voc.h5", get_voc_train)
    
    @functools.cache
    def get_voc_val():
        print("VOC验证集")
        voc_val = PascalVOC(dataset_path_pascal_vos, split="val")
        voc_val.summary()
        return voc_val

    raw_dataset2h5(config.cache_path / "val_voc.h5", get_voc_val)

    @functools.cache
    def get_drone_val():
        drone_val = DroneDataset(dataset_path_drone_train, split="val")
        return drone_val

    raw_dataset2h5(config.cache_path / "val_pure_drone.h5", get_drone_val)

    @functools.cache
    def get_coco_val():
        print("验证集")
        coco_val = mix_raw_dataset(
            get_drone_val(),
            CocoDataset(
                dataset_path_coco / "val2017",
                dataset_path_coco / "annotations_trainval2017/instances_val2017.json",
            ).delete_object("bird", "person"),
        )
        coco_val.summary()
        return coco_val

    raw_dataset2h5(config.cache_path / "val_mixed.h5", get_coco_val)

    # assert coco_train.get_label_names() == coco_val.get_label_names()

    @functools.cache
    def get_drone_test():
        print("原测试集")
        drone_test = DroneTestDataset(dataset_path_drone_test)
        drone_test.summary()
        return drone_test

    raw_dataset2h5(config.cache_path / "test_pure_drone.h5", get_drone_test)

    @functools.cache
    def get_coco_bird():
        print("测试集+coco中的鸟图像")
        coco_bird = mix_raw_dataset(
            get_drone_test(),
            CocoBird(
                dataset_path_coco / "val2017",
                dataset_path_coco / "annotations_trainval2017/instances_val2017.json",
            ),
        )
        coco_bird.summary()
        return coco_bird

    raw_dataset2h5(config.cache_path / "test_drone_with_coco_bird.h5", get_coco_bird)

    @functools.cache
    def get_bird():
        bird = BirdVSDroneBird(os.path.join(dataset_path_drone_vs_bird, "Birds"))
        print("鸟图像数=", len(bird))
        return bird

    raw_dataset2h5(
        config.cache_path / "test_drone_with_bird.h5",
        lambda: mix_raw_dataset(get_drone_test(), get_coco_bird()),
    )
